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
        z_nodes = z_flat.reshape(-1, self.n_super_nodes, self.latent_dim)
        features = [z_flat]
        
        if self.include_dists:
            dists = []
            inv_dists = []
            inv_sq_dists = []

            for i in range(self.n_super_nodes):
                for j in range(i + 1, self.n_super_nodes):
                    diff = z_nodes[:, i, :2] - z_nodes[:, j, :2]
                    if self.box_size is not None:
                        for dim_idx in range(2):
                            diff[:, dim_idx] -= self.box_size[dim_idx] * np.round(diff[:, dim_idx] / self.box_size[dim_idx])

                    d = np.linalg.norm(diff, axis=1, keepdims=True)
                    dists.append(d)
                    inv_dists.append(1.0 / (d + 0.1))
                    inv_sq_dists.append(1.0 / (d**2 + 0.1))

            if dists:
                features.extend([np.hstack(dists), np.hstack(inv_dists), np.hstack(inv_sq_dists)])

        X = np.hstack(features)
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

        return np.hstack(poly_features)

    def normalize_x(self, X_poly):
        return (X_poly - self.x_poly_mean) / self.x_poly_std

    def normalize_y(self, Y):
        return (Y - self.target_mean) / self.target_std

    def denormalize_y(self, Y_norm):
        return Y_norm * self.target_std + self.target_mean

class OptimizedExpressionProgram:
    """
    Wrapper for expressions with optimized constants.
    """
    def __init__(self, expr_str, original_program):
        self.expr_str = expr_str
        self.original_program = original_program
        self.length_ = getattr(original_program, 'length_', 1)
        try:
            feat_indices = [int(x) for x in re.findall(r'X(\d+)', expr_str)]
            self.max_feat_idx = max(feat_indices) if feat_indices else 0
            n_features = self.max_feat_idx + 1
            feat_vars = [sp.Symbol(f'x{i}') for i in range(n_features)]
            var_mapping = {f'x{i}': var for i, var in enumerate(feat_vars)}
            sympy_expr_str = expr_str
            for i in range(n_features):
                sympy_expr_str = sympy_expr_str.replace(f'X{i}', f'x{i}')
            self.sympy_expr = sp.sympify(sympy_expr_str, locals=var_mapping)
            self.func = sp.lambdify(feat_vars, self.sympy_expr, modules=['numpy'])
        except:
            self.func = None

    def execute(self, X):
        try:
            if self.func is not None:
                if X.ndim == 1: X = X.reshape(1, -1)
                args = [X[:, i] if i < X.shape[1] else np.zeros(X.shape[0]) for i in range(self.max_feat_idx + 1)]
                result = self.func(*args)
                if np.isscalar(result): return np.full(X.shape[0], result)
                return np.asarray(result).flatten() if result.ndim > 1 else np.asarray(result)
            return self.original_program.execute(X)
        except:
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
        return SymbolicRegressor(population_size=pop,
                                 generations=gen, 
                                 stopping_criteria=self.stopping_criteria,
                                 function_set=('add', 'sub', 'mul', 'div', 'neg', 'sin', 'cos'),
                                 p_crossover=0.7, p_subtree_mutation=0.1,
                                 p_hoist_mutation=0.05, p_point_mutation=0.1,
                                 max_samples=0.8, verbose=0, 
                                 parsimony_coefficient=parsimony,
                                 n_jobs=-1,
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
            constants = re.findall(r'\d+\.\d+', expr_str)
            if not constants: constants = re.findall(r'\b\d+\b', expr_str)
            if not constants: return candidate

            const_vars = [sp.Symbol(f'c{i}') for i in range(len(constants))]
            modified_expr_str = expr_str
            for i, c in enumerate(constants):
                modified_expr_str = modified_expr_str.replace(c, f'c{i}', 1)

            n_features = X.shape[1]
            feat_vars = [sp.Symbol(f'x{i}') for i in range(n_features)]
            var_mapping = {f'c{i}': v for i, v in enumerate(const_vars)}
            for i in range(n_features): var_mapping[f'x{i}'] = feat_vars[i]
            
            # gplearn uses X0, X1, etc.
            for i in range(n_features): modified_expr_str = modified_expr_str.replace(f'X{i}', f'x{i}')
            
            full_expr = sp.sympify(modified_expr_str, locals=var_mapping)
            f_lamb = sp.lambdify(const_vars + feat_vars, full_expr, modules=['numpy'])

            def objective(const_vals):
                try:
                    y_pred = f_lamb(*const_vals, *[X[:, i] for i in range(n_features)])
                    return np.mean((y_true - y_pred)**2)
                except: return 1e10

            res = minimize(objective, [float(c) for c in constants], method='L-BFGS-B')
            if res.success:
                optimized_expr_str = expr_str
                for i, val in enumerate(res.x):
                    optimized_expr_str = optimized_expr_str.replace(constants[i], f'{val:.6f}', 1)
                
                # Check if actually improved
                y_pred_opt = f_lamb(*res.x, *[X[:, i] for i in range(n_features)])
                opt_score = 1 - np.sum((y_true - y_pred_opt)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-9)
                
                if opt_score > original_score:
                    candidate.update({'prog': OptimizedExpressionProgram(optimized_expr_str, program), 'score': opt_score})
            return candidate
        except: return candidate

    def _distill_single_target(self, i, X_norm, Y_norm, targets_shape_1, latent_states_shape_1):
        variances = np.var(X_norm, axis=0)
        valid_indices = np.where(variances > 1e-6)[0]
        X_pruned = X_norm[:, valid_indices]
        mask_pruned = self._select_features(X_pruned, Y_norm[:, i])
        full_mask = np.zeros(X_norm.shape[1], dtype=bool)
        full_mask[valid_indices[mask_pruned]] = True
        X_selected = X_norm[:, full_mask]

        if X_selected.shape[1] == 0: return np.zeros((X_norm.shape[0], 1)), full_mask, 0.0

        ridge = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        ridge.fit(X_selected, Y_norm[:, i])
        if ridge.score(X_selected, Y_norm[:, i]) > 0.985:
            class LinearProg:
                def __init__(self, m): self.m, self.length_ = m, 1
                def execute(self, X): return self.m.predict(X)
                def __str__(self): return "LinearModel"
            return LinearProg(ridge), full_mask, ridge.score(X_selected, Y_norm[:, i])

        candidates = []
        for p in [0.005, 0.02, 0.1]:
            est = self._get_regressor(self.max_pop // 2, self.max_gen // 2, parsimony=p)
            try:
                est.fit(X_selected, Y_norm[:, i])
                prog = est._program
                score = est.score(X_selected, Y_norm[:, i])
                if targets_shape_1 == latent_states_shape_1 and not self.validate_stability(prog, X_selected[0]): continue
                candidates.append({'prog': prog, 'score': score, 'complexity': prog.length_, 'p': p, 'pareto': score - 0.02 * prog.length_})
            except: continue

        if not candidates: return None, full_mask, 0.0
        best = sorted(candidates, key=lambda x: x['pareto'], reverse=True)[0]
        best = self._refine_constants(best, X_selected, Y_norm[:, i])
        return best['prog'], full_mask, max(0, best['score'] - 0.02 * best['complexity'])

    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None):
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
            ode_dev = next(model.ode_func.parameters()).device
            dz = model.ode_func(torch.tensor([i*dt], device=ode_dev), z_flat.to(ode_dev))
            states.append(z_flat[0].cpu().numpy())
            derivs.append(dz[0].cpu().numpy())
            times.append(i*dt)
            if include_hamiltonian and hasattr(model.ode_func, 'H_net'):
                hams.append(model.ode_func.H_net(z_flat.to(ode_dev))[0].cpu().numpy())
    res = [np.array(states), np.array(derivs), np.array(times)]
    if include_hamiltonian: res.append(np.array(hams))
    return tuple(res)