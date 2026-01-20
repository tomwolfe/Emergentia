from gplearn.genetic import SymbolicRegressor
import numpy as np
import torch
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from joblib import Parallel, delayed

class FeatureTransformer:
    """
    Encapsulates feature engineering logic: polynomial expansions, 
    relative distances, and normalization. Ensures consistency between 
    training and inference (integration).
    """
    def __init__(self, n_super_nodes, latent_dim, include_dists=True):
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.include_dists = include_dists
        self.z_mean = None
        self.z_std = None
        self.x_poly_mean = None
        self.x_poly_std = None
        self.target_mean = None
        self.target_std = None

    def fit(self, latent_states, targets):
        # 1. Fit raw latent normalization
        self.z_mean = latent_states.mean(axis=0)
        self.z_std = latent_states.std(axis=0) + 1e-6
        
        # 2. Transform to poly features
        X_poly = self.transform(latent_states)
        
        # 3. Fit poly feature normalization
        self.x_poly_mean = X_poly.mean(axis=0)
        self.x_poly_std = X_poly.std(axis=0) + 1e-6
        
        # 4. Fit target normalization
        self.target_mean = targets.mean(axis=0)
        self.target_std = targets.std(axis=0) + 1e-6

    def transform(self, z_flat):
        # z_flat: [Batch, n_super_nodes * latent_dim]
        z_nodes = z_flat.reshape(-1, self.n_super_nodes, self.latent_dim)
        
        features = [z_flat]
        if self.include_dists:
            dists = []
            inv_dists = []
            inv_sq_dists = []
            
            for i in range(self.n_super_nodes):
                for j in range(i + 1, self.n_super_nodes):
                    # Relative distance between super-nodes (using first 2 dims as positions)
                    # This assumes the align_loss worked and z[:, :2] is CoM
                    diff = z_nodes[:, i, :2] - z_nodes[:, j, :2]
                    d = np.linalg.norm(diff, axis=1, keepdims=True)
                    dists.append(d)
                    # Physics-informed features: inverse laws
                    inv_dists.append(1.0 / (d + 0.1))
                    inv_sq_dists.append(1.0 / (d**2 + 0.1))
            
            if dists:
                features.extend([np.hstack(dists), np.hstack(inv_dists), np.hstack(inv_sq_dists)])
        
        X = np.hstack(features)
        
        # Polynomial expansion (Linear + Quadratic)
        n_raw = X.shape[1]
        poly_features = [X]
        # Only square the raw latent variables to avoid feature explosion
        n_latents = self.n_super_nodes * self.latent_dim
        for i in range(n_latents):
            poly_features.append((X[:, i:i+1]**2))
            
            # Smart cross-terms: only between dimensions of the SAME super-node
            # or same dimensions across DIFFERENT super-nodes
            node_idx = i // self.latent_dim
            dim_idx = i % self.latent_dim
            
            # Cross-terms within same node
            for j in range(i + 1, (node_idx + 1) * self.latent_dim):
                poly_features.append(X[:, i:i+1] * X[:, j:j+1])
        
        return np.hstack(poly_features)

    def normalize_x(self, X_poly):
        return (X_poly - self.x_poly_mean) / self.x_poly_std

    def normalize_y(self, Y):
        return (Y - self.target_mean) / self.target_std

    def denormalize_y(self, Y_norm):
        return Y_norm * self.target_std + self.target_mean

class SymbolicDistiller:
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12):
        self.max_pop = populations
        self.max_gen = generations
        self.stopping_criteria = stopping_criteria
        self.max_features = max_features
        self.feature_masks = []
        self.transformer = None

    def _get_regressor(self, pop, gen, parsimony=0.01):
        # Allow dynamic parsimony to find the 'knee' of the Pareto front
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
        """
        Hybrid selection: Random Forest for non-linear importance 
        + Lasso for linear/sparse consistency.
        """
        # 1. Non-linear importance (Random Forest)
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = rf.feature_importances_
        
        # 2. Linear importance (LassoCV)
        lasso = LassoCV(cv=5, max_iter=3000)
        lasso.fit(X, y)
        lasso_importance = np.abs(lasso.coef_)
        
        # Combined score (geometric mean to find features that are important in both)
        combined_score = np.sqrt(rf_importance * (lasso_importance + 1e-9))
        
        # Select top-k
        threshold = np.sort(combined_score)[-self.max_features] if len(combined_score) > self.max_features else 0
        mask = combined_score >= threshold
        return mask

    def _distill_single_target(self, i, X_norm, Y_norm, targets_shape_1, latent_states_shape_1):
        print(f"Selecting features for target_{i} (Input dim: {X_norm.shape[1]})...")
        
        variances = np.var(X_norm, axis=0)
        valid_indices = np.where(variances > 1e-6)[0]
        X_pruned = X_norm[:, valid_indices]
        
        mask_pruned = self._select_features(X_pruned, Y_norm[:, i])
        
        full_mask = np.zeros(X_norm.shape[1], dtype=bool)
        full_mask[valid_indices[mask_pruned]] = True
        
        X_selected = X_norm[:, full_mask]
        print(f"  -> Target_{i}: Reduced to {X_selected.shape[1]} informative variables.")

        if X_selected.shape[1] == 0:
            return np.zeros((X_norm.shape[0], 1)), full_mask, 0.0

        ridge = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        ridge.fit(X_selected, Y_norm[:, i])
        linear_score = ridge.score(X_selected, Y_norm[:, i])
        
        if linear_score > 0.985:
            print(f"  -> Target_{i}: High linear fit (R2={linear_score:.3f}). Using linear model.")
            class LinearProgram:
                def __init__(self, model): self.model = model; self.length_ = 1
                def execute(self, X): 
                    if X.ndim == 1: X = X.reshape(1, -1)
                    return self.model.predict(X)
            return LinearProgram(ridge), full_mask, linear_score

        parsimony_levels = [0.001, 0.005, 0.01]
        complexity_factor = max(1.0, 2.0 * (1.0 - linear_score))
        scaled_pop = int(self.max_pop * complexity_factor)
        scaled_pop = min(scaled_pop, 5000)
        
        candidates = []
        for p_coeff in parsimony_levels:
            est = self._get_regressor(scaled_pop, self.max_gen // 2, parsimony=p_coeff)
            try:
                est.fit(X_selected, Y_norm[:, i])
                # gplearn's _program is the best one.
                # We can also get the hall of fame or best from each generation, 
                # but let's stick to the best for each parsimony level.
                prog = est._program
                score = est.score(X_selected, Y_norm[:, i])
                
                is_stable = True
                if targets_shape_1 == latent_states_shape_1:
                    is_stable = self.validate_stability(prog, X_selected[0])
                
                if is_stable:
                    candidates.append({'prog': prog, 'score': score, 'complexity': self.get_complexity(prog), 'p': p_coeff})
            except Exception as e:
                print(f"    ! GP search failed for p={p_coeff}: {e}")
        
        if not candidates:
            return None, full_mask, 0.0

        candidates.sort(key=lambda x: x['score'], reverse=True)
        best_candidate = candidates[0]
        
        # Calculate "Uncertainty" based on agreement of top candidates' scores and complexities
        # High confidence if multiple parsimony levels find similar quality models
        top_scores = [c['score'] for c in candidates[:3]]
        score_std = np.std(top_scores) if len(top_scores) > 1 else 0.1
        confidence = best_candidate['score'] * (1.0 - score_std)

        for c in candidates[1:]:
            if (best_candidate['score'] - c['score']) < 0.03 and c['complexity'] < 0.6 * best_candidate['complexity']:
                best_candidate = c
                
        if best_candidate['score'] < 0.85:
            print(f"  -> Escalating distillation for target_{i}...")
            est = self._get_regressor(self.max_pop, self.max_gen, parsimony=best_candidate['p'])
            est.fit(X_selected, Y_norm[:, i])
            best_candidate = {'prog': est._program, 'score': est.score(X_selected, Y_norm[:, i])}
            confidence = best_candidate['score'] # Simplified update
            
        return best_candidate['prog'], full_mask, confidence

    def distill(self, latent_states, targets, n_super_nodes, latent_dim):
        self.transformer = FeatureTransformer(n_super_nodes, latent_dim)
        self.transformer.fit(latent_states, targets)
        
        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)
        Y_norm = self.transformer.normalize_y(targets)

        results = Parallel(n_jobs=-1)(
            delayed(self._distill_single_target)(i, X_norm, Y_norm, targets.shape[1], latent_states.shape[1])
            for i in range(targets.shape[1])
        )
        
        equations = [r[0] for r in results]
        self.feature_masks = [r[1] for r in results]
        self.confidences = [r[2] for r in results]
                
        return equations

    def evaluate_on_test(self, programs, latent_states, targets):
        if self.transformer is None: return None
        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)
        Y_norm = self.transformer.normalize_y(targets)
        
        scores = []
        for i, (prog, mask) in enumerate(zip(programs, self.feature_masks)):
            y_pred = prog.execute(X_norm[:, mask])
            score = 1 - ((Y_norm[:, i] - y_pred)**2).sum() / (((Y_norm[:, i] - Y_norm[:, i].mean())**2).sum() + 1e-9)
            scores.append(score)
        return np.array(scores)

def extract_latent_data(model, dataset, dt, include_hamiltonian=False):
    model.eval()
    latent_states, latent_derivs, hamiltonians, times = [], [], [], []
    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            x, edge_index = data.x.to(device), data.edge_index.to(device)
            z, _, _, _ = model.encode(x, edge_index, torch.zeros(x.size(0), dtype=torch.long, device=device))
            z_flat = z.view(1, -1)
            
            ode_device = next(model.ode_func.parameters()).device
            dz = model.ode_func(torch.tensor([i*dt], device=ode_device), z_flat.to(ode_device))
            
            latent_states.append(z_flat[0].cpu().numpy())
            latent_derivs.append(dz[0].cpu().numpy())
            times.append(i*dt)
            if include_hamiltonian and hasattr(model.ode_func, 'H_net'):
                hamiltonians.append(model.ode_func.H_net(z_flat.to(ode_device))[0].cpu().numpy())

    res = [np.array(latent_states), np.array(latent_derivs), np.array(times)]
    if include_hamiltonian: res.append(np.array(hamiltonians))
    return tuple(res)

