from gplearn.genetic import SymbolicRegressor
import numpy as np
import torch
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

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
            for i in range(self.n_super_nodes):
                for j in range(i + 1, self.n_super_nodes):
                    # Efficiently compute distances for the batch
                    d = np.linalg.norm(z_nodes[:, i] - z_nodes[:, j], axis=1, keepdims=True)
                    dists.append(d)
            if dists:
                features.append(np.hstack(dists))
        
        X = np.hstack(features)
        
        # Polynomial expansion (Linear + Quadratic)
        n_raw = X.shape[1]
        poly_features = [X]
        for i in range(n_raw):
            poly_features.append((X[:, i:i+1]**2))
            # Cross-terms (limited to reduce explosion)
            for j in range(i + 1, min(i + 5, n_raw)): 
                poly_features.append(X[:, i:i+1] * X[:, j:j+1])
        
        return np.hstack(poly_features)

    def normalize_x(self, X_poly):
        return (X_poly - self.x_poly_mean) / self.x_poly_std

    def normalize_y(self, Y):
        return (Y - self.target_mean) / self.target_std

    def denormalize_y(self, Y_norm):
        return Y_norm * self.target_std + self.target_mean

class SymbolicDistiller:
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=15):
        self.max_pop = populations
        self.max_gen = generations
        self.stopping_criteria = stopping_criteria
        self.max_features = max_features
        self.feature_masks = []
        self.transformer = None

    def _get_regressor(self, pop, gen):
        # Increased parsimony_coefficient to 0.05 (from 0.01) to penalize complex junk terms
        # Simplified function_set: removed 'log', 'sqrt', 'abs' which are often sources of instability
        return SymbolicRegressor(population_size=pop,
                                 generations=gen, 
                                 stopping_criteria=self.stopping_criteria,
                                 function_set=('add', 'sub', 'mul', 'div', 'neg', 'sin', 'cos'),
                                 p_crossover=0.7, p_subtree_mutation=0.1,
                                 p_hoist_mutation=0.05, p_point_mutation=0.1,
                                 max_samples=0.9, verbose=0, 
                                 parsimony_coefficient=0.05,
                                 random_state=42)

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

    def distill(self, latent_states, targets, n_super_nodes, latent_dim):
        """
        Generic distillation of targets from latent states.
        targets can be dZ/dt or Hamiltonian H.
        """
        self.transformer = FeatureTransformer(n_super_nodes, latent_dim)
        self.transformer.fit(latent_states, targets)
        
        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)
        Y_norm = self.transformer.normalize_y(targets)

        equations = []
        self.feature_masks = []

        for i in range(targets.shape[1]):
            print(f"Selecting features for target_{i} (Input dim: {X_norm.shape[1]})...")
            selector = SelectFromModel(LassoCV(cv=5, max_iter=2000), max_features=self.max_features)
            selector.fit(X_norm, Y_norm[:, i])
            mask = selector.get_support()
            self.feature_masks.append(mask)
            X_selected = X_norm[:, mask]
            
            print(f"  -> Reduced features to {X_selected.shape[1]} informative variables.")

            coarse_pop = self.max_pop // 4
            coarse_gen = self.max_gen // 2
            
            print(f"Distilling target_{i} (Coarse search)...")
            est = self._get_regressor(coarse_pop, coarse_gen)
            est.fit(X_selected, Y_norm[:, i])
            
            best_prog = est._program
            # Stability check only makes sense for dZ/dt, not for H
            # But we'll keep it as a general check if targets.shape[1] == latent_states.shape[1]
            is_stable = True
            if targets.shape[1] == latent_states.shape[1]:
                is_stable = self.validate_stability(best_prog, X_selected[0])
            
            if est.score(X_selected, Y_norm[:, i]) > 0.95 and is_stable:
                print(f"  -> Good fit found in coarse search (R^2={est.score(X_selected, Y_norm[:, i]):.3f})")
                equations.append(best_prog)
            else:
                print(f"  -> Escalating to Fine search (pop={self.max_pop}, gen={self.max_gen})...")
                est = self._get_regressor(self.max_pop, self.max_gen)
                est.fit(X_selected, Y_norm[:, i])
                
                # Final check: is the fine search actually better?
                # Sometimes GP just finds a complex identity that overfits.
                # We prefer simpler programs if the score is 'good enough'.
                if hasattr(est, '_program'):
                    equations.append(est._program)
                else:
                    equations.append(best_prog) # Fallback to coarse if fine failed
                
        return equations

    def evaluate_on_test(self, programs, latent_states, targets):
        """
        Evaluates discovered programs on a hold-out set to detect overfitting.
        """
        if self.transformer is None:
            return None
            
        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)
        Y_norm = self.transformer.normalize_y(targets)
        
        scores = []
        for i, (prog, mask) in enumerate(zip(programs, self.feature_masks)):
            X_selected = X_norm[:, mask]
            y_pred = prog.execute(X_selected)
            # R^2 score
            u = ((Y_norm[:, i] - y_pred) ** 2).sum()
            v = ((Y_norm[:, i] - Y_norm[:, i].mean()) ** 2).sum()
            score = 1 - u / (v + 1e-9)
            scores.append(score)
            
        return np.array(scores)

def extract_latent_data(model, dataset, dt, include_hamiltonian=False):
    model.eval()
    latent_states = []
    latent_derivs = []
    hamiltonians = []
    times = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            current_t = i * dt

            z, s, _, _ = model.encode(x, edge_index, torch.zeros(x.size(0), dtype=torch.long, device=device))
            z_flat = z.view(1, -1)
            
            # Latent Derivative
            ode_device = next(model.ode_func.parameters()).device
            t_ode = torch.tensor([current_t], dtype=torch.float32, device=ode_device)
            z_for_ode = z_flat.to(ode_device)
            dz = model.ode_func(t_ode, z_for_ode)
            
            latent_states.append(z_flat[0].cpu().numpy())
            latent_derivs.append(dz[0].cpu().numpy())
            times.append(current_t)

            if include_hamiltonian and hasattr(model.ode_func, 'H_net'):
                H = model.ode_func.H_net(z_for_ode)
                hamiltonians.append(H[0].cpu().numpy())

    res = [np.array(latent_states), np.array(latent_derivs), np.array(times)]
    if include_hamiltonian:
        res.append(np.array(hamiltonians))
    return tuple(res)

