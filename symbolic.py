from gplearn.genetic import SymbolicRegressor
import numpy as np
import torch
import sympy as sp
import re
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from scipy.optimize import minimize

class FeatureTransformer:
    """
    Encapsulates feature engineering logic: polynomial expansions,
    relative distances, and normalization. Ensures consistency between
    training and inference (integration).
    """
    def __init__(self, n_super_nodes, latent_dim, include_dists=True, box_size=None):
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.include_dists = include_dists
        self.box_size = box_size
        self.z_mean = None
        self.z_std = None
        self.x_poly_mean = None
        self.x_poly_std = None
        self.target_mean = None
        self.target_std = None

    def fit(self, latent_states, targets):
        self.z_mean = latent_states.mean(axis=0)
        self.z_std = latent_states.std(axis=0) + 1e-6
        X_poly = self.transform(latent_states)
        self.x_poly_mean = X_poly.mean(axis=0)
        self.x_poly_std = X_poly.std(axis=0) + 1e-6
        self.target_mean = targets.mean(axis=0)
        self.target_std = targets.std(axis=0) + 1e-6

    def transform(self, z_flat):
        # Clamp input for stability
        z_flat = np.clip(z_flat, -1e6, 1e6)
        batch_size = z_flat.shape[0]
        z_nodes = z_flat.reshape(batch_size, self.n_super_nodes, self.latent_dim)
        features = [z_flat]
        
        if self.include_dists:
            # Vectorized distance calculation
            pos = z_nodes[:, :, :2] # [Batch, K, 2]
            # Expansion to [Batch, K, K, 2]
            diffs = pos[:, :, None, :] - pos[:, None, :, :]
            
            if self.box_size is not None:
                box = np.array(self.box_size)
                diffs = diffs - box * np.round(diffs / box)
            
            # Compute norms [Batch, K, K]
            dists_matrix = np.linalg.norm(diffs, axis=-1)
            
            # Extract upper triangle indices (i < j)
            i_idx, j_idx = np.triu_indices(self.n_super_nodes, k=1)
            
            dists_flat = dists_matrix[:, i_idx, j_idx] # [Batch, K*(K-1)/2]
            # Clamp distances to avoid 1/0
            dists_flat = np.maximum(dists_flat, 1e-3)
            
            inv_dists_flat = 1.0 / (dists_flat + 0.1)
            inv_sq_dists_flat = 1.0 / (dists_flat**2 + 0.1)
            
            # New physics-informed basis functions
            exp_dist = np.exp(-np.clip(dists_flat, 0, 20))
            screened_coulomb = exp_dist / (dists_flat + 0.1)
            log_dist = np.log(dists_flat + 1.0)
            
            features.extend([dists_flat, inv_dists_flat, inv_sq_dists_flat, exp_dist, screened_coulomb, log_dist])

        X = np.hstack(features)
        # Final clamp of linear features before polynomial expansion
        X = np.clip(X, -1e3, 1e3) 
        
        poly_features = [X]
        n_latents = self.n_super_nodes * self.latent_dim

        for i in range(n_latents):
            poly_features.append((X[:, i:i+1]**2))
            node_idx = i // self.latent_dim
            dim_idx = i % self.latent_dim
            for j in range(i + 1, (node_idx + 1) * self.latent_dim):
                poly_features.append(X[:, i:i+1] * X[:, j:j+1])
            for other_node in range(node_idx + 1, self.n_super_nodes):
                other_idx = other_node * self.latent_dim + dim_idx
                poly_features.append(X[:, i:i+1] * X[:, other_idx:other_idx+1])

        res = np.hstack(poly_features)
        return np.clip(res, -1e6, 1e6) # Final safety clamp

    def normalize_x(self, X_poly):
        return (X_poly - self.x_poly_mean) / self.x_poly_std

    def normalize_y(self, Y):
        return (Y - self.target_mean) / self.target_std

    def denormalize_y(self, Y_norm):
        return Y_norm * self.target_std + self.target_mean

    def transform_jacobian(self, z_flat):
        """
        Computes the Jacobian dX/dz of the transformation.
        X = transform(z)
        Returns a tensor of shape [n_features, n_latents]
        """
        n_latents = self.n_super_nodes * self.latent_dim
        z_nodes = z_flat.reshape(self.n_super_nodes, self.latent_dim)
        
        # We need to track which features were generated to compute their derivatives
        # 1. Raw latents: d(z)/dz = Identity [n_latents, n_latents]
        jacobians = [np.eye(n_latents)]
        
        # 2. Distances and their inverses
        if self.include_dists:
            for i in range(self.n_super_nodes):
                for j in range(i + 1, self.n_super_nodes):
                    # Position dims (assumed first 2)
                    diff = z_nodes[i, :2] - z_nodes[j, :2]
                    if self.box_size is not None:
                        for dim_idx in range(2):
                            diff[dim_idx] -= self.box_size[dim_idx] * np.round(diff[dim_idx] / self.box_size[dim_idx])
                    
                    d = np.linalg.norm(diff) + 1e-9
                    
                    # d(d)/dz
                    jd = np.zeros(n_latents)
                    jd[i*self.latent_dim : i*self.latent_dim+2] = diff / d
                    jd[j*self.latent_dim : j*self.latent_dim+2] = -diff / d
                    jacobians.append(jd.reshape(1, -1))
                    
                    # d(1/(d+0.1))/dz = -1/(d+0.1)^2 * d(d)/dz
                    jacobians.append((-1.0 / (d + 0.1)**2 * jd).reshape(1, -1))
                    
                    # d(1/(d^2+0.1))/dz = -2d/(d^2+0.1)^2 * d(d)/dz
                    jacobians.append((-2.0 * d / (d**2 + 0.1)**2 * jd).reshape(1, -1))

                    # d(exp(-d))/dz = -exp(-d) * d(d)/dz
                    jacobians.append((-np.exp(-d) * jd).reshape(1, -1))

                    # d(exp(-d)/(d+0.1))/dz = (-exp(-d)/(d+0.1) - np.exp(-d)/(d+0.1)**2) * d(d)/dz
                    j_screened = (-np.exp(-d)/(d + 0.1) - np.exp(-d)/(d + 0.1)**2) * jd
                    jacobians.append(j_screened.reshape(1, -1))

                    # d(log(d+1))/dz = 1/(d+1) * d(d)/dz
                    j_log = (1.0 / (d + 1.0) * jd)
                    jacobians.append(j_log.reshape(1, -1))

        # 3. Polynomial terms (Squares and cross-terms)
        # poly_features.append((X[:, i:i+1]**2))
        # This X here is the [z, dists, ...] vector.
        # Let X_linear be the vector [z, dists, ...]
        # d(X_linear_k^2)/dz = 2 * X_linear_k * d(X_linear_k)/dz
        
        X_linear_jac = np.vstack(jacobians)
        X_linear = np.concatenate([z_flat] + [
            np.array([
                np.linalg.norm(z_nodes[i, :2] - z_nodes[j, :2] + 
                              (0 if self.box_size is None else -np.array(self.box_size) * np.round((z_nodes[i, :2] - z_nodes[j, :2])/np.array(self.box_size)))),
                1.0 / (np.linalg.norm(z_nodes[i, :2] - z_nodes[j, :2]) + 0.1),
                1.0 / (np.linalg.norm(z_nodes[i, :2] - z_nodes[j, :2])**2 + 0.1),
                np.exp(-np.linalg.norm(z_nodes[i, :2] - z_nodes[j, :2])),
                np.exp(-np.linalg.norm(z_nodes[i, :2] - z_nodes[j, :2])) / (np.linalg.norm(z_nodes[i, :2] - z_nodes[j, :2]) + 0.1),
                np.log(np.linalg.norm(z_nodes[i, :2] - z_nodes[j, :2]) + 1.0)
            ])
            for i in range(self.n_super_nodes) for j in range(i + 1, self.n_super_nodes)
        ]) if self.include_dists else z_flat
        
        poly_jacobians = [X_linear_jac]
        
        # Raw latent squares and cross terms
        for i in range(n_latents):
            # d(zi^2)/dz = 2 * zi * d(zi)/dz
            jd_sq = 2 * z_flat[i] * X_linear_jac[i]
            poly_jacobians.append(jd_sq.reshape(1, -1))
            
            node_idx = i // self.latent_dim
            # Cross-terms same node
            for j in range(i + 1, (node_idx + 1) * self.latent_dim):
                # d(zi*zj)/dz = zi*d(zj)/dz + zj*d(zi)/dz
                jd_cross = z_flat[i] * X_linear_jac[j] + z_flat[j] * X_linear_jac[i]
                poly_jacobians.append(jd_cross.reshape(1, -1))
            
            # Cross-terms same dim other nodes
            dim_idx = i % self.latent_dim
            for other_node in range(node_idx + 1, self.n_super_nodes):
                other_idx = other_node * self.latent_dim + dim_idx
                jd_cross = z_flat[i] * X_linear_jac[other_idx] + z_flat[other_idx] * X_linear_jac[i]
                poly_jacobians.append(jd_cross.reshape(1, -1))
                
        return np.vstack(poly_jacobians)

def gp_to_sympy(expr_str, n_features=None):
    """
    Robustly converts a gplearn expression string to a SymPy expression.
    Handles prefix notation, custom functions, and variable mapping without brittle regex.
    """
    if expr_str is None:
        return sp.Float(0.0)
    
    # If it's already a SymPy expression, return it
    if isinstance(expr_str, sp.Expr):
        return expr_str

    try:
        # Define local functions for gplearn standard and potential custom functions
        # Use a lambda that handles the 'X' variables dynamically
        class VariableMapper(dict):
            def __getitem__(self, key):
                if key.startswith('X') and key[1:].isdigit():
                    return sp.Symbol(f'x{key[1:]}')
                return super().__getitem__(key)
            
            def __contains__(self, key):
                if key.startswith('X') and key[1:].isdigit():
                    return True
                return super().__contains__(key)

        local_dict = VariableMapper({
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': lambda x, y: x / (y + 1e-9),
            'sqrt': lambda x: sp.sqrt(sp.Abs(x)),
            'log': lambda x: sp.log(sp.Abs(x) + 1e-9),
            'abs': sp.Abs,
            'neg': lambda x: -x,
            'inv': lambda x: 1.0 / (x + 1e-9),
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'sig': lambda x: 1 / (1 + sp.exp(-x)),
            'sigmoid': lambda x: 1 / (1 + sp.exp(-x)),
            'gauss': lambda x: sp.exp(-x**2),
            'exp': sp.exp,
        })

        # 3. Parse using sympify
        # We use evaluate=False to avoid premature simplification that might hide variables
        expr = sp.sympify(expr_str, locals=local_dict)
        
        # Simplify if not too complex
        if expr.count_ops() < 200:
            expr = sp.simplify(expr)
            
        return expr
    except Exception as e:
        # Fallback for simple numeric strings
        try:
            return sp.Float(float(expr_str))
        except:
            return sp.Float(0.0)

class OptimizedExpressionProgram:
    """
    Wrapper for expressions with optimized constants.
    """
    def __init__(self, expr_str, original_program):
        self.expr_str = expr_str
        self.original_program = original_program
        self.length_ = getattr(original_program, 'length_', 1)
        try:
            import re
            feat_indices = [int(x) for x in re.findall(r'X(\d+)', expr_str)]
            self.max_feat_idx = max(feat_indices) if feat_indices else 0
            
            # Use the robust converter
            self.sympy_expr = gp_to_sympy(expr_str, n_features=self.max_feat_idx + 1)
            
            # Identify which variables are actually used
            all_symbols = sorted(list(self.sympy_expr.free_symbols), key=lambda s: s.name)
            self.used_feat_indices = [int(s.name[1:]) for s in all_symbols if s.name.startswith('x')]
            
            # Lambdify for performance
            feat_vars = [sp.Symbol(f'x{i}') for i in range(self.max_feat_idx + 1)]
            self.func = sp.lambdify(feat_vars, self.sympy_expr, modules=['numpy'])
        except Exception as e:
            print(f"Warning: OptimizedExpressionProgram compilation failed: {e}")
            self.func = None

    def execute(self, X):
        if self.func is not None:
            try:
                if X.ndim == 1: X = X.reshape(1, -1)
                # Prepare arguments (only up to what the function expects)
                args = [X[:, i] if i < X.shape[1] else np.zeros(X.shape[0])
                        for i in range(self.max_feat_idx + 1)]
                result = self.func(*args)

                if np.isscalar(result):
                    return np.full(X.shape[0], result)
                result = np.asarray(result)
                # Ensure the result has the right shape - same number of samples as input
                if result.ndim == 1 and result.shape[0] == 1 and X.shape[0] > 1:
                    # If we have a single result but multiple input samples, broadcast it
                    result = np.full(X.shape[0], result[0])
                elif result.ndim == 0:  # Scalar result
                    result = np.full(X.shape[0], result)
                elif result.ndim == 1 and result.shape[0] != X.shape[0]:
                    # If result has different length than expected, pad or truncate
                    if result.shape[0] == 1:
                        result = np.full(X.shape[0], result[0])
                    else:
                        # Truncate or pad to match input size
                        if result.shape[0] < X.shape[0]:
                            result = np.pad(result, (0, X.shape[0] - result.shape[0]), mode='edge')
                        else:
                            result = result[:X.shape[0]]
                return result
            except:
                return self.original_program.execute(X)
        return self.original_program.execute(X)

    def __str__(self):
        return self.expr_str

class SymbolicDistiller:
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12):
        self.max_pop = populations
        self.max_gen = generations
        self.stopping_criteria = stopping_criteria
        self.max_features = max_features
        self.feature_masks = []
        self.transformer = None
        self.confidences = []

    def _get_regressor(self, pop, gen, parsimony=0.01):
        from gplearn.functions import make_function
        # Define protected exp function to prevent overflow
        def _protected_exp(x):
            with np.errstate(over='ignore'):
                return np.clip(np.exp(np.clip(x, -100, 100)), -1e10, 1e10)
        
        exp_func = make_function(function=_protected_exp, name='exp', arity=1)
        
        return SymbolicRegressor(population_size=pop,
                                 generations=gen, 
                                 stopping_criteria=self.stopping_criteria,
                                 function_set=('add', 'sub', 'mul', 'div', 'neg', 'sin', 'cos', 'log', 'sqrt', 'abs', exp_func),
                                 p_crossover=0.7, p_subtree_mutation=0.1,
                                 p_hoist_mutation=0.05, p_point_mutation=0.1,
                                 max_samples=0.8, verbose=0, 
                                 parsimony_coefficient=parsimony,
                                 n_jobs=1,
                                 random_state=42)

    def get_complexity(self, program):
        return program.length_

    def validate_stability(self, program, X_start, dt=0.01, steps=20):
        curr_x = X_start.copy()
        for _ in range(steps):
            try:
                deriv = program.execute(curr_x.reshape(1, -1))
                curr_x += deriv * dt
                if np.any(np.isnan(curr_x)) or np.any(np.isinf(curr_x)) or np.any(np.abs(curr_x) > 1e6):
                    return False
            except:
                return False
        return True

    def _select_features(self, X, y):
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        lasso = LassoCV(cv=5, max_iter=3000)
        lasso.fit(X, y)
        combined_score = np.sqrt(rf.feature_importances_ * (np.abs(lasso.coef_) + 1e-9))
        threshold = np.sort(combined_score)[-self.max_features] if len(combined_score) > self.max_features else 0
        return combined_score >= threshold

    def _refine_constants(self, candidate, X, y_true):
        try:
            program = candidate['prog']
            original_score = candidate['score']
            expr_str = str(program)
            
            # gplearn uses X0, X1, etc.
            n_features = X.shape[1]
            feat_vars = [sp.Symbol(f'x{i}') for i in range(n_features)]
            
            # Define local functions for gplearn standard
            local_dict = {
                'add': lambda x, y: x + y,
                'sub': lambda x, y: x - y,
                'mul': lambda x, y: x * y,
                'div': lambda x, y: x / (y + 1e-9),
                'sqrt': lambda x: sp.sqrt(sp.Abs(x)),
                'log': lambda x: sp.log(sp.Abs(x) + 1e-9),
                'abs': sp.Abs,
                'neg': lambda x: -x,
                'inv': lambda x: 1.0 / (x + 1e-9),
                'sin': sp.sin,
                'cos': sp.cos,
                'exp': sp.exp,
            }
            for i in range(n_features):
                local_dict[f'X{i}'] = feat_vars[i]
            
            # 1. Parse into SymPy
            full_expr = sp.sympify(expr_str, locals=local_dict)
            
            # 2. Extract all numeric constants using SymPy's atoms
            all_atoms = full_expr.atoms(sp.Number)
            # Filter out small integers and powers that might be structural (like in x**2)
            # Actually, gplearn puts constants as separate nodes.
            constants = sorted([float(a) for a in all_atoms if not a.is_Integer or abs(a) > 5])
            
            if not constants: return candidate

            # 3. Create a map from constants to parameters
            const_vars = [sp.Symbol(f'c{i}') for i in range(len(constants))]
            subs_map = {sp.Float(c): cv for c, cv in zip(constants, const_vars)}
            
            # Parametrized expression for optimization
            param_expr = full_expr.subs(subs_map)
            f_lamb = sp.lambdify(const_vars + feat_vars, param_expr, modules=['numpy'])

            def objective(const_vals):
                try:
                    y_pred = f_lamb(*const_vals, *[X[:, i] for i in range(n_features)])
                    return np.mean((y_true - y_pred)**2)
                except: return 1e10

            res = minimize(objective, constants, method='L-BFGS-B')
            if res.success:
                opt_consts = res.x
                
                # Physics-Inspired Simplification Step
                simplified_consts = opt_consts.copy()
                for j in range(len(simplified_consts)):
                    val = simplified_consts[j]
                    for base in [1.0, 0.5, 0.25]:
                        rounded = round(val / base) * base
                        if abs(val - rounded) < 0.15:
                            simplified_consts[j] = rounded
                            break
                
                mse_opt = objective(opt_consts)
                mse_simple = objective(simplified_consts)
                final_consts = simplified_consts if mse_simple < 1.05 * mse_opt else opt_consts
                
                # Create final optimized expression
                final_subs = {cv: sp.Float(val) for cv, val in zip(const_vars, final_consts)}
                optimized_expr = param_expr.subs(final_subs)
                
                # Evaluate and update if improved
                y_pred_opt = f_lamb(*final_consts, *[X[:, i] for i in range(n_features)])
                opt_score = 1 - np.sum((y_true - y_pred_opt)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-9)
                
                if opt_score > original_score:
                    # Convert back to gplearn-style string if possible, or use the OptimizedExpressionProgram
                    # OptimizedExpressionProgram takes an expr_str
                    # We can use sp.srepr or just str() and then fix it up
                    # But actually OptimizedExpressionProgram uses gp_to_sympy, so we can just pass the SymPy string
                    candidate.update({'prog': OptimizedExpressionProgram(str(optimized_expr), program), 'score': opt_score})
            return candidate
        except Exception as e:
            return candidate

    def _distill_single_target(self, i, X_norm, Y_norm, targets_shape_1, latent_states_shape_1):
        # Filter out any rows with NaNs or Infs
        mask_finite = np.isfinite(X_norm).all(axis=1) & np.isfinite(Y_norm[:, i])
        if not np.any(mask_finite):
            print(f"  -> Target_{i}: No finite data points available for distillation.")
            return None, np.zeros(X_norm.shape[1], dtype=bool), 0.0
            
        X_norm = X_norm[mask_finite]
        Y_norm_i = Y_norm[mask_finite, i]

        variances = np.var(X_norm, axis=0)
        valid_indices = np.where(variances > 1e-6)[0]
        X_pruned = X_norm[:, valid_indices]
        
        # Aggressive feature selection
        mask_pruned = self._select_features(X_pruned, Y_norm_i)
        full_mask = np.zeros(X_norm.shape[1], dtype=bool)
        full_mask[valid_indices[mask_pruned]] = True
        X_selected = X_norm[:, full_mask]

        if X_selected.shape[1] == 0: 
            return np.zeros((X_norm.shape[0], 1)), full_mask, 0.0

        # Shortcut 1: Single feature correlation (Physics often has this)
        corrs = np.array([np.abs(np.corrcoef(X_selected[:, j], Y_norm_i)[0, 1]) for j in range(X_selected.shape[1])])
        if np.any(corrs > 0.99):
            best_feat_idx = np.argmax(corrs)
            from sklearn.linear_model import LinearRegression
            lr = LinearRegression().fit(X_selected[:, [best_feat_idx]], Y_norm_i)
            class SingleFeatProg:
                def __init__(self, m, idx): self.m, self.idx, self.length_ = m, idx, 1
                def execute(self, X): return self.m.predict(X[:, [self.idx]])
                def __str__(self): return f"Linear(X{self.idx})"
            return SingleFeatProg(lr, best_feat_idx), full_mask, lr.score(X_selected[:, [best_feat_idx]], Y_norm_i)

        # Shortcut 2: Linear model (Ridge)
        ridge = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        ridge.fit(X_selected, Y_norm_i)
        r2 = ridge.score(X_selected, Y_norm_i)
        if r2 > 0.9999: # Increased from 0.95 to 0.9999
            class LinearProg:
                def __init__(self, m): self.m, self.length_ = m, 1
                def execute(self, X): return self.m.predict(X)
                def __str__(self): 
                    coeffs = self.m.coef_
                    terms = [f"{coeffs[j]:.3f}*X{j}" for j in range(len(coeffs)) if abs(coeffs[j]) > 1e-3]
                    return " + ".join(terms) + f" + {self.m.intercept_:.3f}"
            return LinearProg(ridge), full_mask, r2

        # Parallel GP candidates
        candidates = []
        # Reduce population/generations if we have many features to speed up
        pop_scale = 1.0 if X_selected.shape[1] < 20 else 0.5
        
        for p in [0.01, 0.05]: # Reduced from 3 trials to 2
            est = self._get_regressor(int(self.max_pop * pop_scale), self.max_gen, parsimony=p)
            try:
                est.fit(X_selected, Y_norm_i)
                prog = est._program
                score = est.score(X_selected, Y_norm_i)
                if targets_shape_1 == latent_states_shape_1 and not self.validate_stability(prog, X_selected[0]): 
                    continue
                candidates.append({'prog': prog, 'score': score, 'complexity': prog.length_, 'p': p, 'pareto': score - 0.02 * prog.length_})
                
                # Early exit if we found a near-perfect model
                if score > 0.99: break
            except: continue

        if not candidates: return None, full_mask, 0.0
        best = sorted(candidates, key=lambda x: x['pareto'], reverse=True)[0]
        best = self._refine_constants(best, X_selected, Y_norm_i)
        return best['prog'], full_mask, max(0, best['score'] - 0.02 * best['complexity'])

    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None):
        # Final safety check for NaNs/Infs in raw input
        finite_mask = np.isfinite(latent_states).all(axis=1) & np.isfinite(targets).all(axis=1)
        latent_states = latent_states[finite_mask]
        targets = targets[finite_mask]
        
        # Clamp targets to avoid extreme values in GP fitting
        targets = np.clip(targets, -1e6, 1e6)
        
        if len(latent_states) < 10:
            print("Warning: Insufficient finite data points for symbolic distillation.")
            return [None] * targets.shape[1]

        self.transformer = FeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        self.transformer.fit(latent_states, targets)
        X_norm = self.transformer.normalize_x(self.transformer.transform(latent_states))
        Y_norm = self.transformer.normalize_y(targets)

        results = Parallel(n_jobs=-1)(
            delayed(self._distill_single_target)(i, X_norm, Y_norm, targets.shape[1], latent_states.shape[1])
            for i in range(targets.shape[1])
        )
        equations, self.feature_masks, self.confidences = zip(*results)
        return list(equations)

    def evaluate_on_test(self, programs, latent_states, targets):
        X_norm = self.transformer.normalize_x(self.transformer.transform(latent_states))
        Y_norm = self.transformer.normalize_y(targets)
        scores = []
        for i, (prog, mask) in enumerate(zip(programs, self.feature_masks)):
            if prog is None:
                scores.append(0.0)
                continue
            y_pred = prog.execute(X_norm[:, mask])
            scores.append(1 - np.sum((Y_norm[:, i] - y_pred)**2) / (np.sum((Y_norm[:, i] - Y_norm[:, i].mean())**2) + 1e-9))
        return np.array(scores)

def extract_latent_data(model, dataset, dt, include_hamiltonian=False):
    model.eval()
    states, derivs, times, hams = [], [], [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, data in enumerate(dataset):
            x, ei = data.x.to(device), data.edge_index.to(device)
            z, _, _, _ = model.encode(x, ei, torch.zeros(x.size(0), dtype=torch.long, device=device))
            z_flat = z.view(1, -1)
            
            # Extract dynamics derivative
            ode_dev = next(model.ode_func.parameters()).device
            try:
                dz = model.ode_func(torch.tensor([i*dt], device=ode_dev), z_flat.to(ode_dev))
                
                # Check for NaNs or massive values in dz
                if torch.isnan(dz).any() or torch.isinf(dz).any() or torch.max(torch.abs(dz)) > 1e4:
                    continue
                    
                states.append(z_flat[0].cpu().numpy())
                derivs.append(dz[0].cpu().numpy())
                times.append(i*dt)
                
                if include_hamiltonian and hasattr(model.ode_func, 'H_net'):
                    h_val = model.ode_func.H_net(z_flat.to(ode_dev))[0]
                    hams.append(h_val.cpu().numpy())
            except Exception as e:
                # Silently skip integration errors for data extraction
                continue
                
    if len(states) == 0:
        # Fallback if everything was NaNs
        return (np.zeros((0, model.encoder.n_super_nodes * model.encoder.latent_dim)), 
                np.zeros((0, model.encoder.n_super_nodes * model.encoder.latent_dim)), 
                np.zeros(0))
                
    res = [np.array(states), np.array(derivs), np.array(times)]
    if include_hamiltonian: res.append(np.array(hams))
    return tuple(res)