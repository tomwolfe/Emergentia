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

class FeatureTransformer:
    """
    Consolidated Feature Transformer with physics-inspired features and normalization.
    """
    def __init__(self, n_super_nodes, latent_dim, include_dists=True, box_size=None):
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.include_dists = include_dists
        self.box_size = box_size
        self.x_poly_mean = None
        self.x_poly_std = None
        self.target_mean = None
        self.target_std = None
        self.selector = None

    def fit(self, latent_states, targets):
        X_poly = self.transform(latent_states)
        self.x_poly_mean = X_poly.mean(axis=0)
        self.x_poly_std = X_poly.std(axis=0) + 1e-6
        self.target_mean = targets.mean(axis=0)
        self.target_std = targets.std(axis=0) + 1e-6

    def transform(self, z_flat):
        if torch.is_tensor(z_flat):
            z_flat = z_flat.detach().cpu().numpy()
        if z_flat.ndim == 1:
            z_flat = z_flat.reshape(1, -1)
            
        batch_size = z_flat.shape[0]
        n_nodes = self.n_super_nodes
        n_dims = self.latent_dim
        z_flat = np.clip(z_flat, -1e6, 1e6)
        z_nodes = z_flat.reshape(batch_size, n_nodes, n_dims)

        features = [z_flat]
        if self.include_dists:
            pos = z_nodes[:, :, :2]
            diffs = pos[:, :, None, :] - pos[:, None, :, :]
            if self.box_size is not None:
                box = np.array(self.box_size)
                diffs = diffs - box * np.round(diffs / box)
            
            dists_matrix = np.linalg.norm(diffs, axis=-1)
            i_idx, j_idx = np.triu_indices(n_nodes, k=1)
            dists_flat = dists_matrix[:, i_idx, j_idx]
            dists_flat = np.maximum(dists_flat, 1e-3)
            
            exp_dist = np.exp(-np.clip(dists_flat, 0, 20))
            features.extend([
                dists_flat, 1.0/(dists_flat + 0.1), 1.0/(dists_flat**2 + 0.1),
                exp_dist, exp_dist/(dists_flat + 0.1), np.log(dists_flat + 1.0)
            ])
            
            half_d = n_dims // 2
            p_vars = z_nodes[:, :, half_d:]
            features.append(np.sum(p_vars**2, axis=-1))

        X = np.hstack(features)
        X = np.clip(X, -1e3, 1e3)
        poly_features = [X]
        max_poly = min(20, n_nodes * n_dims)
        for i in range(max_poly):
            v_idx = i % (n_nodes * n_dims)
            poly_features.append(X[:, v_idx:v_idx+1]**2)
            node_idx = v_idx // n_dims
            for j in range(v_idx + 1, min(v_idx + 4, (node_idx + 1) * n_dims)):
                poly_features.append(X[:, v_idx:v_idx+1] * X[:, j:j+1])
        
        X_out = np.clip(np.hstack(poly_features), -1e6, 1e6)
        
        # Feature Pruning: remove near-constant or highly redundant features
        if self.selector is not None:
            try:
                X_out = self.selector.transform(X_out)
            except:
                pass # Fallback to raw features if selector fails
        
        return X_out

    def normalize_x(self, X_poly): return (X_poly - self.x_poly_mean) / self.x_poly_std
    def normalize_y(self, Y): return (Y - self.target_mean) / self.target_std
    def denormalize_y(self, Y_norm): return Y_norm * self.target_std + self.target_mean

class OptimizedExpressionProgram:
    def __init__(self, expr_str, original_program=None):
        self.expr_str = expr_str
        self.original_program = original_program
        try:
            self.sympy_expr = sp.sympify(expr_str)
            self.symbols = sorted(list(self.sympy_expr.free_symbols), key=lambda s: str(s))
            self.func = sp.lambdify(self.symbols, self.sympy_expr, modules=['numpy'])
            self.length_ = len(self.sympy_expr.args) if self.sympy_expr.is_Add else 1
        except:
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
        square = make_function(function=lambda x: x**2, name='square', arity=1)
        return SymbolicRegressor(population_size=pop, generations=gen, parsimony_coefficient=parsimony,
                                 function_set=('add', 'sub', 'mul', safe_div, safe_sqrt, safe_log, 'abs', 'neg', safe_inv, square),
                                 n_jobs=1, random_state=42, verbose=0)

    def _select_features(self, X, y):
        f_scores, _ = f_regression(X, y)
        mi_scores = mutual_info_regression(X, y, random_state=42)
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
        mask = self._select_features(X, y)
        X_sel = X[:, mask]
        
        lasso = LassoCV(cv=5).fit(X_sel, y)
        lasso_score = lasso.score(X_sel, y)
        
        # PHYSICALITY GATE: Check if Lasso found a simple identity
        is_identity = False
        if lasso_score > 0:
            coeffs = np.abs(lasso.coef_)
            if np.sum(coeffs > 1e-3) == 1 and np.max(coeffs) > 0.9:
                is_identity = True
        
        if lasso_score > 0.999:
            return OptimizedExpressionProgram(str(sp.simplify(sum(c*sp.Symbol(f'X{np.where(mask)[0][j]}') for j, c in enumerate(lasso.coef_)) + lasso.intercept_))), mask, lasso_score
            
        # Initial GP search
        est = self._get_regressor(self.populations, self.generations).fit(X_sel, y)
        score = est.score(X_sel, y)
        program_str = str(est._program)
        
        # PHYSICALITY GATE: Trigger Deep Search if score is low or result is trivial
        is_gp_trivial = re.match(r'^X\d+$', program_str) or re.match(r'^neg\(X\d+\)$', program_str) or len(program_str) < 5
        
        if (score < 0.8) or (is_gp_trivial and score < 0.95) or is_identity:
            # print(f"Target {i}: Physicality Gate triggered. Score {score:.4f} or trivial result. Starting Deep Search...")
            est = self._get_regressor(self.populations * 3, self.generations * 2, parsimony=0.01).fit(X_sel, y)
            score = est.score(X_sel, y)
            program_str = str(est._program)
            
        return OptimizedExpressionProgram(program_str), mask, score

    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None, hamiltonian=False):
        self.transformer = FeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        
        # If pruning is needed, initialize selector
        X_raw = self.transformer.transform(latent_states)
        if latent_states.shape[0] < X_raw.shape[1]:
            k = max(1, latent_states.shape[0] // 2)
            self.transformer.selector = SelectKBest(f_regression, k=k).fit(X_raw, targets[:, 0])
        
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
                z_flat = z.view(1, -1)
                
                states.append(z_flat[0].cpu().numpy())
                
                if include_hamiltonian and hasattr(model.ode_func, 'H_net'):
                    # Extract scalar Hamiltonian
                    ode_device = next(model.ode_func.parameters()).device
                    H = model.ode_func.H_net(z_flat.to(ode_device))
                    derivs.append(H.cpu().numpy().flatten())
                else:
                    # Extract dynamics derivative
                    ode_device = next(model.ode_func.parameters()).device
                    dz = model.ode_func(torch.tensor([i*dt], device=ode_device), z_flat.to(ode_device))
                    derivs.append(dz[0].cpu().numpy())
            except Exception as e:
                continue
                
    if not states:
        return np.array([]), np.array([]), np.array([])
        
    return np.array(states), np.array(derivs), np.linspace(0, len(states)*dt, len(states))

def gp_to_sympy(expr_str, *args, **kwargs):
    local_dict = {'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'div': lambda x,y: x/(y+1e-9), 'sqrt': lambda x: sp.sqrt(sp.Abs(x)), 'log': lambda x: sp.log(sp.Abs(x)+1e-9), 'abs': sp.Abs, 'neg': lambda x: -x, 'inv': lambda x: 1.0/(x+1e-9), 'square': lambda x: x**2}
    return sp.sympify(expr_str, locals=local_dict)

