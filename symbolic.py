import numpy as np
import torch
import sympy as sp
import re
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# Sanitized Symbolic Primitives
from pure_symbolic_functions import create_safe_functions
safe_sqrt, safe_log, safe_div, safe_inv = create_safe_functions()
from balanced_features import BalancedFeatureTransformer as FeatureTransformer

class OptimizedExpressionProgram:
    def __init__(self, expr_str, original_program=None):
        self.expr_str = expr_str
        self.original_program = original_program
        try:
            # Use the robust gp_to_sympy logic for consistency
            self.sympy_expr = gp_to_sympy(expr_str)
            self.symbols = sorted(list(self.sympy_expr.free_symbols), key=lambda s: str(s))
            self.func = sp.lambdify(self.symbols, self.sympy_expr, modules=['numpy'])
            self.length_ = len(self.sympy_expr.args) if self.sympy_expr.is_Add else 1
        except Exception as e:
            # print(f"DEBUG: OptimizedExpressionProgram failed for {expr_str}: {e}")
            self.func = None
            self.length_ = getattr(original_program, 'length_', 1) if original_program else 1

    def execute(self, X):
        if self.func:
            try:
                if X.ndim == 1: X = X.reshape(1, -1)
                args = []
                for sym in self.symbols:
                    match = re.search(r'(\d+)', str(sym))
                    idx = int(match.group(1)) if match else 0
                    args.append(X[:, idx] if idx < X.shape[1] else np.zeros(X.shape[0]))
                res = np.asarray(self.func(*args))
                if res.ndim == 0: return np.full(X.shape[0], res)
                return res.flatten()[:X.shape[0]]
            except: pass
        return self.original_program.execute(X) if self.original_program else np.zeros(X.shape[0])

    def __str__(self): return self.expr_str

class SymbolicDistiller:
    def __init__(self, populations=500, generations=10, stopping_criteria=0.001, max_features=8):
        self.populations = populations
        self.generations = generations
        self.stopping_criteria = stopping_criteria
        self.max_features = max_features
        self.transformer = None

    def get_complexity(self, program):
        """Returns the length of the program as a measure of complexity."""
        return getattr(program, 'length_', 1)

    def _get_regressor(self, pop, gen, parsimony=0.02):
        # Safe square function that prevents overflow
        def safe_square(x):
            x_clipped = np.clip(x, -1e6, 1e6)  # Prevent large values before squaring
            return np.clip(x_clipped**2, -1e12, 1e12)  # Also clip result

        square = make_function(function=safe_square, name='square', arity=1)

        # Safe inverse square function that prevents overflow
        def safe_inv_square(x):
            x_squared = np.clip(x**2, 1e-12, 1e12)  # Prevent underflow/overflow in x^2
            result = 1.0 / (x_squared + 1e-9)
            return np.clip(result, -1e12, 1e12)  # Clip final result

        inv_square = make_function(function=safe_inv_square, name='inv_square', arity=1)

        # Use n_jobs=-1 to utilize all available CPU cores for GP
        # Added max_samples=0.9 to speed up fitness evaluation and reduce memory
        return SymbolicRegressor(population_size=pop, generations=gen, parsimony_coefficient=parsimony,
                                 function_set=('add', 'sub', 'mul', safe_div, safe_sqrt, safe_log, 'abs', 'neg', safe_inv, square, inv_square),
                                 max_samples=0.9, n_jobs=-1, random_state=42, verbose=0)

    def _select_features(self, X, y):
        # Sample data if it's too large for mutual_info_regression
        if X.shape[0] > 1000:
            indices = np.random.choice(X.shape[0], 1000, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y

        f_scores, _ = f_regression(X_sample, y_sample)
        
        # mutual_info_regression is slow; using sampled data improves performance significantly
        mi_scores = mutual_info_regression(X_sample, y_sample, random_state=42)
        
        combined = 0.5 * (f_scores / (f_scores.max() + 1e-9)) + 0.5 * (mi_scores / (mi_scores.max() + 1e-9))
        mask = np.zeros(X.shape[1], dtype=bool)
        mask[np.argsort(combined)[-self.max_features:]] = True
        return mask

    def validate_stability(self, program, X_sample):
        """Simple stability check: ensure no NaNs or Infs for a sample."""
        try:
            if hasattr(X_sample, 'detach'): X_sample = X_sample.detach().cpu().numpy()
            y = program.execute(X_sample.reshape(1, -1))
            return np.all(np.isfinite(y))
        except:
            return False

    def _distill_single_target(self, i, X, y, targets_shape_1=None, latent_states_shape_1=None, is_hamiltonian=False):
        print(f"    -> [Target {i}] Selecting top {self.max_features} features...")
        mask = self._select_features(X, y)
        X_sel = X[:, mask]
        
        print(f"    -> [Target {i}] Fitting LassoCV for baseline...")
        lasso = LassoCV(cv=5).fit(X_sel, y)
        lasso_score = lasso.score(X_sel, y)
        
        # PHYSICALITY GATE: Check if Lasso found a simple identity
        is_identity = False
        if lasso_score > 0:
            coeffs = np.abs(lasso.coef_)
            if np.sum(coeffs > 1e-3) == 1 and np.max(coeffs) > 0.9:
                is_identity = True
        
        if lasso_score > 0.999:
            print(f"    -> [Target {i}] Lasso found near-perfect fit (R2={lasso_score:.4f}). Skipping GP.")
            return OptimizedExpressionProgram(str(sp.simplify(sum(c*sp.Symbol(f'X{np.where(mask)[0][j]}') for j, c in enumerate(lasso.coef_)) + lasso.intercept_))), mask, lasso_score
            
        # Initial GP search
        print(f"    -> [Target {i}] Starting initial GP search (Pop: {self.populations}, Gen: {self.generations}, Samples: 90%)...")
        est = self._get_regressor(self.populations, self.generations).fit(X_sel, y)
        score = est.score(X_sel, y)
        program_str = str(est._program)
        print(f"    -> [Target {i}] Initial GP R2: {score:.4f}")
        
        # PHYSICALITY GATE: Trigger Deep Search if score is low or result is trivial
        is_gp_trivial = re.match(r'^X\d+$', program_str) or re.match(r'^neg\(X\d+\)$', program_str) or len(program_str) < 5
        
        if (score < 0.8) or (is_gp_trivial and score < 0.95) or is_identity:
            p_deep = min(self.populations * 2, 5000) # Capped at 5000 to save RAM
            g_deep = min(self.generations * 2, 100)  # Capped at 100 to save time
            print(f"    -> [Target {i}] Physicality Gate triggered. Starting Deep Search (Pop: {p_deep}, Gen: {g_deep})...")
            est = self._get_regressor(p_deep, g_deep, parsimony=0.01).fit(X_sel, y)
            score = est.score(X_sel, y)
            program_str = str(est._program)
            print(f"    -> [Target {i}] Deep Search R2: {score:.4f}")
            
        return OptimizedExpressionProgram(program_str), mask, score

    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None, hamiltonian=False):
        self.transformer = FeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        self.transformer.fit(latent_states, targets)
        X_norm = self.transformer.normalize_x(self.transformer.transform(latent_states))
        Y_norm = self.transformer.normalize_y(targets)
        
        if hamiltonian:
            # Regress a SINGLE target: the scalar Hamiltonian H
            # We assume the first column of targets is H if hamiltonian=True is passed
            # or that extract_latent_data was called with include_hamiltonian=True
            res, mask, score = self._distill_single_target(0, X_norm, Y_norm[:, 0], 1, latent_dim, is_hamiltonian=True)
            return [res]
        
        results = Parallel(n_jobs=-1)(delayed(self._distill_single_target)(i, X_norm, Y_norm[:, i], targets.shape[1], latent_dim) for i in range(targets.shape[1]))
        return [r[0] for r in results]

def extract_latent_data(model, dataset, dt, include_hamiltonian=False):
    model.eval()
    states, derivs = [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x = data.x.to(device)
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                ei = data.edge_index.to(device)
            else:
                ei = torch.empty((2, 0), dtype=torch.long, device=device)

            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

            try:
                z, _, _, _ = model.encode(x, ei, batch)

                # Check for NaN in z before proceeding
                if torch.isnan(z).any():
                    print(f"DEBUG: NaN detected in encoded z at step {i}")
                    continue

                z_flat = z.reshape(1, -1)  # Use reshape instead of view to avoid stride issues

                states.append(z_flat[0].cpu().numpy())

                if include_hamiltonian and hasattr(model.ode_func, 'H_net'):
                    # Extract scalar Hamiltonian
                    ode_device = next(model.ode_func.parameters()).device
                    H = model.ode_func.H_net(z_flat.to(ode_device))

                    # Check for NaN in H
                    if torch.isnan(H).any():
                        print(f"DEBUG: NaN detected in H at step {i}")
                        continue

                    derivs.append(H.cpu().numpy().flatten())
                else:
                    # Extract dynamics derivative
                    ode_device = next(model.ode_func.parameters()).device
                    dz = model.ode_func(torch.tensor([i*dt], device=ode_device), z_flat.to(ode_device))

                    # Check for NaN in dz
                    if torch.isnan(dz).any():
                        print(f"DEBUG: NaN detected in dz at step {i}")
                        # Replace NaN values with 0
                        dz = torch.nan_to_num(dz, nan=0.0, posinf=1e2, neginf=-1e2)

                    derivs.append(dz[0].cpu().numpy())
            except Exception as e:
                print(f"Exception at step {i}: {e}")
                continue

    if not states:
        return np.array([]), np.array([]), np.array([])

    return np.array(states), np.array(derivs), np.linspace(0, len(states)*dt, len(states))

def gp_to_sympy(expr_str, *args, **kwargs):
    local_dict = {
        'add': lambda x,y: x+y,
        'sub': lambda x,y: x-y,
        'mul': lambda x,y: x*y,
        'div': lambda x,y: x/(y+1e-9),
        'sqrt': lambda x: sp.sqrt(sp.Abs(x)),
        'log': lambda x: sp.log(sp.Abs(x)+1e-9),
        'abs': sp.Abs,
        'neg': lambda x: -x,
        'inv': lambda x: 1.0/(x+1e-9),
        'square': lambda x: sp.Pow(x, 2),  # Using SymPy power function for better handling
        'inv_square': lambda x: 1.0/(sp.Pow(x, 2)+1e-9)  # Using SymPy power function for better handling
    }
    return sp.sympify(expr_str, locals=local_dict)

