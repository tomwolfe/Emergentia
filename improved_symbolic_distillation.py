"""
Improved Optimized Symbolic Distillation Module

This module implements performance-optimized symbolic regression to speed up
the symbolic distillation process while maintaining accuracy and improving
feature selection and equation discovery.
"""

import numpy as np
import torch
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from symbolic import SymbolicDistiller, FeatureTransformer, gp_to_sympy
from balanced_features import BalancedFeatureTransformer
from hamiltonian_symbolic import HamiltonianSymbolicDistiller
import sympy as sp
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class ImprovedFeatureTransformer(FeatureTransformer):
    """
    Improved feature transformer with better physics-inspired features and reduced computational overhead.
    """
    def __init__(self, n_super_nodes, latent_dim, include_dists=True, box_size=None):
        super().__init__(n_super_nodes, latent_dim, include_dists, box_size)

    def transform(self, z_flat):
        """
        Improved feature transformation with better physics-inspired features.
        Input: z_flat [batch_size, n_features] where n_features = n_nodes * n_dims
        """
        # Ensure z_flat is a numpy array
        if torch.is_tensor(z_flat):
            z_flat = z_flat.detach().cpu().numpy()
            
        # Handle 1D input by reshaping to 2D with batch_size=1
        if z_flat.ndim == 1:
            # Reshape 1D to 2D with batch_size=1
            z_flat = z_flat.reshape(1, -1)
        
        batch_size = z_flat.shape[0]
        n_nodes = self.n_super_nodes
        n_dims = self.latent_dim
        
        # Check if the input dimension matches expected (n_nodes * n_dims)
        if z_flat.shape[1] != n_nodes * n_dims:
            # If mismatch, we might be receiving a single sample incorrectly shaped
            # Try to force it to the expected total size if possible
            expected_total = n_nodes * n_dims
            if z_flat.size == expected_total:
                z_flat = z_flat.reshape(1, expected_total)
                batch_size = 1
            else:
                # If we really have a size mismatch, log it and try to continue with padding/truncating
                # though this is a sign of an upstream issue
                if z_flat.shape[1] < expected_total:
                    z_flat = np.pad(z_flat, ((0, 0), (0, expected_total - z_flat.shape[1])), mode='constant')
                else:
                    z_flat = z_flat[:, :expected_total]

        # Clamp input for stability
        z_flat = np.clip(z_flat, -1e6, 1e6)

        # Reshape to [batch_size, n_nodes, n_dims]
        z_nodes = z_flat.reshape(batch_size, n_nodes, n_dims)

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

            # NEW: Physics-informed basis functions
            exp_dist = np.exp(-np.clip(dists_flat, 0, 20))
            screened_coulomb = exp_dist / (dists_flat + 0.1)
            log_dist = np.log(dists_flat + 1.0)
            
            # NEW: Additional physics-inspired features
            sin_dist = np.sin(dists_flat)
            cos_dist = np.cos(dists_flat)
            sqrt_dist = np.sqrt(dists_flat)
            
            # NEW: Combined features
            dist_times_inv = dists_flat * inv_dists_flat
            dist_plus_inv = dists_flat + inv_dists_flat

            features.extend([dists_flat, inv_dists_flat, inv_sq_dists_flat, 
                           exp_dist, screened_coulomb, log_dist,
                           sin_dist, cos_dist, sqrt_dist,
                           dist_times_inv, dist_plus_inv])
            
            # NEW: Explicit kinetic energy terms (p^2) for each node
            # Assuming p is in the second half of the latent dimensions
            half_d = n_dims // 2
            p_vars = z_nodes[:, :, half_d:]
            ke_terms = np.sum(p_vars**2, axis=-1) # [Batch, K]
            features.append(ke_terms)

        X = np.hstack(features)
        # Final clamp of linear features before polynomial expansion
        X = np.clip(X, -1e3, 1e3)

        # NEW: Improved polynomial features with better physics-inspired combinations
        poly_features = [X]
        n_latents = self.n_super_nodes * self.latent_dim

        # NEW: More sophisticated polynomial features
        max_features_for_poly = min(20, n_latents)  # Increased limit for better representation

        for i in range(max_features_for_poly):
            valid_idx = i % n_latents
            # Quadratic terms
            poly_features.append((X[:, valid_idx:valid_idx+1]**2))
            
            node_idx = valid_idx // self.latent_dim
            dim_idx = valid_idx % self.latent_dim
            
            # Cross terms with expanded range
            for j in range(valid_idx + 1, min(valid_idx + 6, (node_idx + 1) * self.latent_dim)):  # Increased cross terms per feature
                poly_features.append(X[:, valid_idx:valid_idx+1] * X[:, j:j+1])
                
            # Cross terms with other nodes
            for other_node in range(node_idx + 1, min(node_idx + 5, self.n_super_nodes)):  # Increased other nodes
                other_idx = other_node * self.latent_dim + dim_idx
                poly_features.append(X[:, valid_idx:valid_idx+1] * X[:, other_idx:other_idx+1])
                
            # NEW: Cubic terms for important features
            if valid_idx % 3 == 0:  # Every third feature gets cubic term
                poly_features.append((X[:, valid_idx:valid_idx+1]**3))

        res = np.hstack(poly_features)

        # Always return 2D result for consistency with sklearn-like transformers
        return np.clip(res, -1e6, 1e6)

class ImprovedSymbolicDistiller(SymbolicDistiller):
    """
    Improved Symbolic Distiller with enhanced feature selection and equation discovery.
    """

    def __init__(self, populations=1500, generations=40, stopping_criteria=0.001, max_features=15,
                 secondary_optimization=True, opt_method='L-BFGS-B', opt_iterations=100,
                 use_sindy_pruning=True, sindy_threshold=0.001,  # Reduced from 0.01 to prevent over-pruning
                 enhanced_feature_selection=True, physics_informed=True):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.secondary_optimization = secondary_optimization
        self.opt_method = opt_method
        self.opt_iterations = opt_iterations
        self.use_sindy_pruning = use_sindy_pruning
        self.sindy_threshold = sindy_threshold
        self.enhanced_feature_selection = enhanced_feature_selection
        self.physics_informed = physics_informed

    def _get_regressor(self, population_size, generations, parsimony=0.001):
        """
        Create an improved symbolic regressor with better function sets and parameters.
        """
        from gplearn.genetic import SymbolicRegressor
        from gplearn.functions import make_function

        # Define square and inv functions for gplearn
        def _square(x):
            return x**2
        square = make_function(function=_square, name='square', arity=1)
        
        def _inv(x):
            return 1.0 / (x + 1e-9)
        inv = make_function(function=_inv, name='inv', arity=1)

        # Restricted function set: removed tan, sin, cos; added square and inv
        function_set = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', inv, square)

        est = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            stopping_criteria=self.stopping_criteria,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=0,
            parsimony_coefficient=parsimony,
            const_range=(-2., 2.),  # Wider range for constants
            init_depth=(2, 8),      # Deeper trees for more complex expressions
            init_method='half and half',
            function_set=function_set,
            metric='mean absolute error',
            tournament_size=25,     # Larger tournaments for better selection
            n_jobs=1                # Use single thread to avoid overhead
        )
        return est

    def _enhanced_feature_selection(self, X, y, max_features=None):
        """
        Enhanced feature selection using multiple methods and ensemble approach.
        """
        if max_features is None:
            max_features = self.max_features

        n_features = X.shape[1]
        if n_features <= max_features:
            return np.arange(n_features, dtype=int)

        # NEW: Multiple feature selection methods
        from sklearn.feature_selection import f_regression, mutual_info_regression
        from sklearn.feature_selection import SelectKBest
        
        # Method 1: F-regression
        f_scores, _ = f_regression(X, y)
        
        # Method 2: Mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Method 3: Variance threshold
        variances = np.var(X, axis=0)
        
        # Ensemble scores
        normalized_f = f_scores / (np.max(f_scores) + 1e-9)
        normalized_mi = mi_scores / (np.max(mi_scores) + 1e-9)
        normalized_var = variances / (np.max(variances) + 1e-9)
        
        # Weighted combination (physics-informed weights)
        combined_scores = 0.4 * normalized_f + 0.4 * normalized_mi + 0.2 * normalized_var
        
        # Select top features
        top_indices = np.argsort(combined_scores)[-max_features:]
        top_indices = np.sort(top_indices)

        # Create boolean mask
        mask = np.zeros(n_features, dtype=bool)
        mask[top_indices] = True
        return mask

    def _select_features(self, X, y, max_features=None):
        """
        Optimized feature selection with reduced computational overhead.
        """
        if self.enhanced_feature_selection:
            return self._enhanced_feature_selection(X, y, max_features)
        else:
            if max_features is None:
                max_features = self.max_features

            n_features = X.shape[1]
            if n_features <= max_features:
                return np.arange(n_features, dtype=int)

            # Use a faster feature selection method
            from sklearn.feature_selection import f_regression

            # Compute F-statistic for each feature
            f_scores, _ = f_regression(X, y)

            # Select top features
            top_indices = np.argsort(f_scores)[-max_features:]
            top_indices = np.sort(top_indices)

            # Create boolean mask
            mask = np.zeros(n_features, dtype=bool)
            mask[top_indices] = True
            return mask

    def _sindy_select(self, X, y, threshold=0.005, max_iter=15):  # Reduced default threshold to be less aggressive
        """
        Enhanced Sequential Thresholded Least Squares (STLSQ) for SINDy-style pruning.
        With adaptive threshold fallback to prevent over-pruning.
        """
        from sklearn.linear_model import ElasticNet
        n_features = X.shape[1]
        
        y_std = np.std(y)
        # Use relative threshold based on signal standard deviation, but with a floor
        curr_threshold = max(threshold * y_std, 1e-4)
        best_mask = np.ones(n_features, dtype=bool)
        
        # Adaptive loop: if we prune too much, try again with smaller threshold
        for attempt in range(5): # Up to 5 attempts with decreasing threshold
            mask = np.ones(n_features, dtype=bool)
            y_var = np.var(y)
            adaptive_base_threshold = max(curr_threshold, 0.0001 * np.sqrt(y_var))

            for iteration in range(max_iter):
                if not np.any(mask):
                    break

                model = ElasticNet(alpha=1e-5, l1_ratio=0.9, max_iter=1000)
                X_active = X[:, mask]

                if X_active.shape[1] == 0:
                    break

                model.fit(X_active, y)

                new_mask = np.zeros(n_features, dtype=bool)
                active_coeffs = np.abs(model.coef_)

                if len(active_coeffs) > 0:
                    # Adaptive thresholding: combination of base threshold and percentile
                    percentile_threshold = np.percentile(active_coeffs, 40)
                    adaptive_threshold = max(adaptive_base_threshold, percentile_threshold)
                    new_mask[mask] = active_coeffs > adaptive_threshold
                else:
                    new_mask[mask] = True

                if np.array_equal(mask, new_mask):
                    break
                mask = new_mask
            
            # Check if we have enough features
            num_selected = np.sum(mask)
            if num_selected >= 2: # At least 2 features
                best_mask = mask
                break
            
            # If not enough features, reduce threshold and try again
            curr_threshold *= 0.5
            best_mask = mask # Keep the best so far just in case

        # FEATURE FALLBACK: If SINDy prunes to 0 features, force a fallback
        if np.sum(best_mask) == 0:
            print("  -> SINDy pruned all features. Falling back to top 5 features by Mutual Information.")
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y, random_state=42)
            top_indices = np.argsort(mi_scores)[-min(5, n_features):]
            best_mask = np.zeros(n_features, dtype=bool)
            best_mask[top_indices] = True
            
        return best_mask

    def _validate_physics_consistency(self, program, X_sample):
        """
        NEW: Validate that the discovered equation is physically consistent.
        """
        try:
            # Check for numerical stability
            y_pred = program.execute(X_sample)
            if not np.all(np.isfinite(y_pred)):
                return False
            
            # NEW: Check for conservation properties if applicable
            # This is a simplified check - could be expanded based on domain knowledge
            if len(y_pred) > 1:
                # Check for extreme variations that might indicate instability
                variation = np.std(y_pred) / (np.mean(np.abs(y_pred)) + 1e-9)
                if variation > 100:  # Too much variation
                    return False
                    
            return True
        except:
            return False

    def get_complexity(self, program):
        """
        Custom complexity measure with higher penalty for transcendental functions.
        """
        expr_str = str(program)
        # Base complexity is the number of nodes (length)
        base_complexity = getattr(program, 'length_', len(expr_str.split('(')))
        
        # Additional penalty for transcendental functions (Parsimony Boost)
        # Polynomial terms are preferred for Hamiltonians
        transcendental_penalty = 0.0
        transcendental_penalty += expr_str.count('log') * 3.0
        transcendental_penalty += expr_str.count('sqrt') * 2.0
        transcendental_penalty += expr_str.count('abs') * 1.0
        transcendental_penalty += expr_str.count('inv') * 1.0
        
        return base_complexity + transcendental_penalty

    def _distill_single_target(self, i, X_norm, Y_norm, targets_shape_1, latent_states_shape_1, is_hamiltonian=False):
        """
        Improved single target distillation with enhanced validation and selection.
        """
        print(f"Selecting features for target_{i} (Input dim: {X_norm.shape[1]})...")
        
        # Increase parsimony if we are distilling a Hamiltonian
        parsimony_multiplier = 10.0 if is_hamiltonian else 1.0

        # NEW: Advanced variance-based filtering
        variances = np.var(X_norm, axis=0)
        # NEW: Also remove features with very low variance (close to constant)
        valid_indices = np.where((variances > 1e-6) & (variances < 1e6))[0]
        X_pruned = X_norm[:, valid_indices]

        # Use SINDy-style pruning if enabled
        if self.use_sindy_pruning:
            sindy_mask = self._sindy_select(X_pruned, Y_norm[:, i], threshold=self.sindy_threshold)
            print(f"  -> SINDy pruned {len(valid_indices)} to {np.sum(sindy_mask)} features.")

            if np.sum(sindy_mask) < 2:
                print("  -> SINDy too aggressive, using standard selection.")
                mask_pruned = self._select_features(X_pruned, Y_norm[:, i])

                # Ensure we have at least some features selected
                if not np.any(mask_pruned):
                    # If standard selection also failed, pick the top 2 features by variance
                    variances = np.var(X_pruned, axis=0)
                    top_indices = np.argsort(variances)[-min(2, len(variances)):]
                    mask_pruned = np.zeros(len(valid_indices), dtype=bool)
                    mask_pruned[top_indices] = True
            else:
                X_sindy = X_pruned[:, sindy_mask]
                # Further refine with standard feature selector to reach max_features
                refinement_mask = self._select_features(X_sindy, Y_norm[:, i])

                # Ensure we have at least some features after refinement
                if not np.any(refinement_mask):
                    # If refinement failed, use all sindy-selected features
                    refinement_mask = np.ones(len(sindy_mask), dtype=bool)

                mask_pruned = np.zeros(len(valid_indices), dtype=bool)
                mask_pruned[np.where(sindy_mask)[0][refinement_mask]] = True
        else:
            mask_pruned = self._select_features(X_pruned, Y_norm[:, i])

            # Ensure we have at least some features selected
            if not np.any(mask_pruned):
                # If standard selection failed, pick the top 2 features by variance
                variances = np.var(X_pruned, axis=0)
                top_indices = np.argsort(variances)[-min(2, len(variances)):]
                mask_pruned = np.zeros(len(valid_indices), dtype=bool)
                mask_pruned[top_indices] = True

        full_mask = np.zeros(X_norm.shape[1], dtype=bool)
        full_mask[valid_indices[mask_pruned]] = True

        X_selected = X_norm[:, full_mask]
        print(f"  -> Target_{i}: Reduced to {X_selected.shape[1]} informative variables.")

        if X_selected.shape[1] == 0:
            return np.zeros((X_norm.shape[0], 1)), full_mask, 0.0

        # NEW: Enhanced linear fit check with multiple metrics
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        
        # Use LassoCV for better parsimony (L1 regularization)
        lasso = LassoCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1], cv=min(5, len(X_selected)))
        lasso.fit(X_selected, Y_norm[:, i])
        linear_score = lasso.score(X_selected, Y_norm[:, i])
        
        # NEW: Try a simple quadratic model as well with Lasso
        try:
            quad_model = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('lasso', LassoCV(alphas=[1e-6, 1e-4, 1e-2, 1], cv=min(5, len(X_selected))))
            ])
            quad_model.fit(X_selected, Y_norm[:, i])
            quad_score = quad_model.score(X_selected, Y_norm[:, i])
        except:
            quad_score = linear_score  # Fallback if quadratic fails

        best_linear_score = max(linear_score, quad_score)

        # NEW: Much more stringent threshold for linear/quadratic shortcuts (0.995)
        # and enforce parsimony (max 15 terms)
        if best_linear_score > 0.995:
            if quad_score > linear_score:
                class QuadraticProgram:
                    def __init__(self, model, feature_indices):
                        self.model = model
                        self.feature_indices = feature_indices

                        # Create a human-readable string representation
                        try:
                            lasso_model = model.named_steps['lasso']
                            poly_features = model.named_steps['poly']
                            
                            coeffs = lasso_model.coef_
                            intercept = lasso_model.intercept_
                            
                            # Get feature names from poly_features
                            raw_names = poly_features.get_feature_names_out()
                            
                            terms = []
                            if abs(intercept) > 1e-6:
                                terms.append(f"{intercept:.6f}")
                                
                            for idx, coef in enumerate(coeffs):
                                if abs(coef) > 1e-6:
                                    name = raw_names[idx]
                                    import re
                                    def replace_name(match):
                                        i = int(match.group(1))
                                        return f"X{feature_indices[i]}"
                                    
                                    clean_name = re.sub(r'x(\d+)', replace_name, name)
                                    
                                    if ' ' in clean_name: # Cross term: X1 X2
                                        parts = clean_name.split(' ')
                                        term = f"mul({coef:.6f}, mul({parts[0]}, {parts[1]}))"
                                    elif '^2' in clean_name: # Quadratic term: X1^2
                                        base = clean_name.replace('^2', '')
                                        term = f"mul({coef:.6f}, mul({base}, {base}))"
                                    else: # Linear term: X1
                                        term = f"mul({coef:.6f}, {clean_name})"
                                    
                                    terms.append(term)
                                    
                            self.length_ = len(terms)
                            if not terms:
                                self.expr_str = f"{intercept:.6e}"
                            elif len(terms) == 1:
                                self.expr_str = terms[0]
                            else:
                                self.expr_str = terms[0]
                                for term in terms[1:]:
                                    self.expr_str = f"add({self.expr_str}, {term})"
                        except Exception as e:
                            self.expr_str = f"QuadraticModel(R2={quad_score:.3f})"
                            self.length_ = 1

                    def execute(self, X):
                        if X.ndim == 1: X = X.reshape(1, -1)
                        result = self.model.predict(X)
                        result = np.asarray(result)
                        if result.ndim == 0:
                            result = np.full(X.shape[0], result)
                        elif result.ndim == 1:
                            if result.shape[0] != X.shape[0]:
                                if result.shape[0] == 1:
                                    result = np.full(X.shape[0], result[0])
                                else:
                                    result = result[:X.shape[0]]
                                    if result.shape[0] < X.shape[0]:
                                        result = np.pad(result, (0, X.shape[0] - result.shape[0]), mode='edge')
                        else:
                            if result.shape[1] == 1 and len(result.shape) == 2:
                                result = result.flatten()
                            else:
                                result = result.flatten()
                                if result.shape[0] != X.shape[0]:
                                    if result.shape[0] == 1:
                                        result = np.full(X.shape[0], result[0])
                                    else:
                                        result = result[:X.shape[0]]
                                        if result.shape[0] < X.shape[0]:
                                            result = np.pad(result, (0, X.shape[0] - result.shape[0]), mode='edge')
                        return result

                    def __str__(self):
                        return self.expr_str

                selected_indices = np.where(full_mask)[0]
                prog = QuadraticProgram(quad_model, selected_indices)
                if prog.length_ <= 15:
                    print(f"  -> Target_{i}: Good quadratic fit (R2={quad_score:.3f}, terms={prog.length_}). Using quadratic model.")
                    return prog, full_mask, quad_score
                else:
                    print(f"  -> Target_{i}: Quadratic fit too complex ({prog.length_} terms). Falling back to GP.")
            else:
                class LinearProgram:
                    def __init__(self, model, feature_indices):
                        self.model = model
                        self.feature_indices = feature_indices
                        terms = []
                        if abs(model.intercept_) > 1e-6:
                            terms.append(f"{model.intercept_:.6f}")

                        coeffs = model.coef_
                        for idx, coef in enumerate(coeffs):
                            if abs(coef) > 1e-6:
                                orig_idx = feature_indices[idx]
                                terms.append(f"mul({coef:.6f}, X{orig_idx})")

                        self.length_ = len(terms)
                        if not terms:
                            self.expr_str = f"{model.intercept_:.6e}"
                        elif len(terms) == 1:
                            self.expr_str = terms[0]
                        else:
                            self.expr_str = terms[0]
                            for term in terms[1:]:
                                self.expr_str = f"add({self.expr_str}, {term})"

                    def execute(self, X):
                        if X.ndim == 1: X = X.reshape(1, -1)
                        result = self.model.predict(X)
                        result = np.asarray(result)
                        if result.ndim == 0:
                            result = np.full(X.shape[0], result)
                        elif result.ndim == 1:
                            if result.shape[0] != X.shape[0]:
                                if result.shape[0] == 1:
                                    result = np.full(X.shape[0], result[0])
                                else:
                                    result = result[:X.shape[0]]
                                    if result.shape[0] < X.shape[0]:
                                        result = np.pad(result, (0, X.shape[0] - result.shape[0]), mode='edge')
                        else:
                            if result.shape[1] == 1 and len(result.shape) == 2:
                                result = result.flatten()
                            else:
                                result = result.flatten()
                                if result.shape[0] != X.shape[0]:
                                    if result.shape[0] == 1:
                                        result = np.full(X.shape[0], result[0])
                                    else:
                                        result = result[:X.shape[0]]
                                        if result.shape[0] < X.shape[0]:
                                            result = np.pad(result, (0, X.shape[0] - result.shape[0]), mode='edge')
                        return result

                    def __str__(self):
                        return self.expr_str

                selected_indices = np.where(full_mask)[0]
                prog = LinearProgram(lasso, selected_indices)
                if prog.length_ <= 15:
                    print(f"  -> Target_{i}: Good linear fit (R2={linear_score:.3f}, terms={prog.length_}). Using linear model.")
                    return prog, full_mask, linear_score
                else:
                    print(f"  -> Target_{i}: Linear fit too complex ({prog.length_} terms). Falling back to GP.")

        # NEW: Enhanced GP search with better parameters
        parsimony_levels = [0.0001 * parsimony_multiplier, 0.001 * parsimony_multiplier, 0.01 * parsimony_multiplier]
        complexity_factor = max(1.0, 3.0 * (1.0 - best_linear_score))  # Increased factor
        scaled_pop = int(self.max_pop * complexity_factor)
        scaled_pop = min(scaled_pop, 8000)  # Increased max population

        candidates = []
        for p_coeff in parsimony_levels:
            est = self._get_regressor(scaled_pop//len(parsimony_levels), 
                                     max(self.max_gen//2, 10), parsimony=p_coeff)  # At least 10 generations
            try:
                # Force float64 for stability
                X_gp = X_selected.astype(np.float64)
                y_gp = Y_norm[:, i].astype(np.float64)

                est.fit(X_gp, y_gp)
                prog = est._program

                # NEW: Robust scoring with multiple metrics
                y_pred = est.predict(X_gp)
                if not np.all(np.isfinite(y_pred)):
                    # Penalize unstable models heavily
                    score = -10.0
                else:
                    # NEW: Use multiple metrics for better evaluation
                    r2 = r2_score(y_gp, y_pred)
                    mse = mean_squared_error(y_gp, y_pred)
                    
                    # NEW: Adjust score based on complexity and stability
                    complexity_penalty = 0.001 * self.get_complexity(prog)
                    stability_penalty = 0.0 if self._validate_physics_consistency(est, X_gp[:min(10, len(X_gp))]) else 5.0
                    
                    score = r2 - complexity_penalty - stability_penalty

                is_stable = True
                if targets_shape_1 == latent_states_shape_1:
                    is_stable = self.validate_stability(prog, X_gp[0])

                if is_stable:
                    candidates.append({'prog': prog, 'score': score, 'complexity': self.get_complexity(prog), 'p': p_coeff})
            except Exception as e:
                print(f"    ! GP search failed for p={p_coeff}: {e}")

        if not candidates:
            return None, full_mask, 0.0

        # NEW: Enhanced Pareto Frontier Selection with better balance
        for c in candidates:
            # NEW: Adjusted score with better balance between accuracy and complexity
            c['pareto_score'] = c['score'] - 0.001 * c['complexity']

        candidates.sort(key=lambda x: x['pareto_score'], reverse=True)
        best_candidate = candidates[0]

        # NEW: Triviality check - if H is just a single variable (e.g., H = -X1), 
        # it's likely a failure to find the true dynamics.
        best_prog_str = str(best_candidate['prog'])
        import re
        is_trivial = len(re.findall(r'X\d+', best_prog_str)) <= 1 and best_candidate['complexity'] < 5
        
        if is_trivial and is_hamiltonian:
            print(f"  -> Detected trivial Hamiltonian: {best_prog_str}. Re-running with high parsimony...")
            high_parsimony = 0.1 * parsimony_multiplier
            est = self._get_regressor(scaled_pop, self.max_gen, parsimony=high_parsimony)
            try:
                est.fit(X_selected.astype(np.float64), Y_norm[:, i].astype(np.float64))
                if est._program.length_ > best_candidate['prog'].length_:
                    print(f"  -> Found better non-trivial alternative: {est._program}")
                    best_candidate['prog'] = est._program
                    best_candidate['score'] = r2_score(Y_norm[:, i], est.predict(X_selected))
                    best_candidate['complexity'] = self.get_complexity(est._program)
            except:
                pass

        # Apply secondary optimization if enabled
        if self.secondary_optimization:
            optimized_prog = self._optimize_constants(best_candidate['prog'], X_selected, Y_norm[:, i])
            if optimized_prog:
                # Evaluate the optimized program
                try:
                    y_pred = optimized_prog.execute(X_selected)
                    if np.all(np.isfinite(y_pred)):
                        opt_score = r2_score(Y_norm[:, i], y_pred)
                        
                        # NEW: Check if optimization improved the score and maintained stability
                        is_stable = self._validate_physics_consistency(optimized_prog, X_selected[:min(10, len(X_selected))])
                        
                        # NEW: Only accept if both score improved and stability maintained
                        if opt_score > best_candidate['score'] and is_stable:
                            print(f"  -> Secondary optimization improved score from {best_candidate['score']:.3f} to {opt_score:.3f}")
                            best_candidate['prog'] = optimized_prog
                            best_candidate['score'] = opt_score
                except:
                    # If optimization failed, keep the original
                    pass

        # NEW: Improved confidence calculation with multiple factors
        base_confidence = max(0, best_candidate['score'])
        complexity_factor = 1.0 / (1.0 + 0.001 * best_candidate['complexity'])
        confidence = base_confidence * complexity_factor

        return best_candidate['prog'], full_mask, confidence

    def _optimize_constants(self, program, X, y_true):
        """
        Enhanced constant refinement with better physics-informed approaches.
        """
        try:
            expr_str = str(program)
            n_features = X.shape[1]
            feat_vars = [sp.Symbol(f'x{i}') for i in range(n_features)]

            # gplearn uses X0, X1, etc.
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
                'square': lambda x: x**2,
                'cube': lambda x: x**3,
                'max': lambda x, y: sp.Max(x, y),
                'min': lambda x, y: sp.Min(x, y),
                'tan': sp.tan,
            }
            for i in range(n_features):
                local_dict[f'X{i}'] = feat_vars[i]

            # 1. Parse into SymPy
            full_expr = sp.sympify(expr_str, locals=local_dict)

            # 2. Extract numeric constants
            all_atoms = full_expr.atoms(sp.Number)
            # NEW: Filter constants with better criteria
            constants = sorted([float(a) for a in all_atoms if not a.is_Integer or abs(a) > 1e-6])

            if not constants:
                return program

            # 3. Parametrize constants (limit to first 15 constants to avoid huge optimization problems)
            constants = constants[:15]  # Increased limit for better optimization
            const_vars = [sp.Symbol(f'c{i}') for i in range(len(constants))]
            subs_map = {sp.Float(c): cv for c, cv in zip(constants, const_vars)}
            param_expr = full_expr.subs(subs_map)

            # 4. NEW: Create a more robust lambdify function
            try:
                f_lamb = sp.lambdify(const_vars + feat_vars, param_expr, modules=['numpy', 'sympy'])
            except:
                # Fallback to numpy only
                f_lamb = sp.lambdify(const_vars + feat_vars, param_expr, modules=['numpy'])

            def eval_expr(const_vals):
                try:
                    # NEW: Add bounds checking to prevent overflow
                    if np.any(np.abs(const_vals) > 1e6):
                        return float('inf')
                    
                    y_pred = f_lamb(*const_vals, *[X[:, i] for i in range(n_features)])
                    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                        return float('inf')
                    return mean_squared_error(y_true, y_pred)
                except:
                    return float('inf')

            # NEW: Enhanced optimization with multiple starting points
            best_result = None
            best_cost = float('inf')
            
            # Try multiple initializations
            for trial in range(3):  # Try 3 different starting points
                if trial == 0:
                    # Original constants
                    start_vals = constants
                elif trial == 1:
                    # Perturbed constants
                    start_vals = [c + np.random.normal(0, 0.1*max(abs(c), 0.1)) for c in constants]
                else:
                    # Random initialization around original values
                    start_vals = [np.random.uniform(max(c-1, -2), min(c+1, 2)) for c in constants]
                
                # NEW: Use different optimization methods
                for method in ['L-BFGS-B', 'Powell', 'Nelder-Mead']:
                    try:
                        result = minimize(eval_expr, start_vals, method=method,
                                         options={'maxiter': min(self.opt_iterations//3, 50)})  # Divide iterations among methods
                        
                        if result.success and result.fun < best_cost:
                            best_result = result
                            best_cost = result.fun
                            
                            # NEW: Early stopping if we find a very good solution
                            if result.fun < 1e-6:
                                break
                    except:
                        continue
                
                if best_cost < 1e-6:  # Very good solution found
                    break

            if best_result is not None and best_result.success:
                opt_consts = best_result.x

                # NEW: Physics-Inspired Simplification with more rules
                simplified_consts = opt_consts.copy()
                for j in range(len(simplified_consts)):
                    val = simplified_consts[j]
                    # NEW: More simplification rules
                    for base in [1.0, 0.5, 0.25, 2.0, 0.1, 0.01, 0.125, 0.75]:
                        rounded = round(val / base) * base
                        if abs(val - rounded) < 0.05 * max(abs(base), 0.1):
                            simplified_consts[j] = rounded
                            break
                    # NEW: Check for common mathematical constants
                    for math_const, name in [(np.pi, "pi"), (np.e, "e"), (1/np.pi, "inv_pi"), (np.sqrt(2), "sqrt2")]:
                        if abs(val - math_const) < 0.05:
                            simplified_consts[j] = math_const
                            break

                mse_opt = eval_expr(opt_consts)
                mse_simple = eval_expr(simplified_consts)
                
                # NEW: Accept simplified version if it's close enough or better
                final_consts = simplified_consts if mse_simple <= 1.05 * mse_opt else opt_consts

                # Create final optimized expression
                final_subs = {cv: sp.Float(val) for cv, val in zip(const_vars, final_consts)}
                optimized_expr = param_expr.subs(final_subs)

                return OptimizedExpressionWrapper(str(optimized_expr), program)
            else:
                return program

        except Exception as e:
            print(f"  -> Secondary optimization failed: {e}")
            return program

    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None):
        """
        Improved distillation with better validation and physics-informed features.
        """
        # Use improved transformer
        self.transformer = ImprovedFeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        self.transformer.fit(latent_states, targets)

        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)
        Y_norm = self.transformer.normalize_y(targets)

        # Process targets sequentially to reduce memory usage
        equations = []
        self.feature_masks = []
        self.confidences = []

        # Check if we are distilling a Hamiltonian (targets.shape[1] == 1 usually means Hamiltonian)
        # Robust check for Hamiltonian distillation (energy is a scalar)
        is_hamiltonian_distill = (targets.ndim == 1) or (targets.shape[-1] == 1)
        if targets.ndim == 1:
            Y_norm = Y_norm.reshape(-1, 1)

        for i in range(Y_norm.shape[1]):
            eq, mask, conf = self._distill_single_target(i, X_norm, Y_norm, Y_norm.shape[1], latent_states.shape[1], is_hamiltonian=is_hamiltonian_distill)
            
            # HARD PHYSICALITY GATING for Hamiltonian
            if is_hamiltonian_distill and eq is not None:
                # Get the set of features used in the equation
                eq_str = str(eq)
                import re
                # Match both X and x prefixes for robustness
                used_features = [int(f) for f in re.findall(r'[Xx](\d+)', eq_str)]
                
                # Check if it contains at least one q and one p from the latent space
                half_d = latent_dim // 2
                q_indices = set()
                p_indices = set()
                for k in range(n_super_nodes):
                    for d in range(half_d):
                        q_indices.add(k * latent_dim + d)
                    for d in range(half_d, latent_dim):
                        p_indices.add(k * latent_dim + d)
                
                has_q = any(f in q_indices for f in used_features)
                has_p = any(f in p_indices for f in used_features)
                
                if not (has_q and has_p):
                    print(f"  -> PHYSICALITY GATE FAILED for Hamiltonian H(z) = {eq_str}. Lacks q or p dependency.")
                    print(f"  -> TRIGGERING DEEP SEARCH (3x Population, 2x Generations)...")
                    
                    # Store original settings
                    orig_pop = self.max_pop
                    orig_gen = self.max_gen
                    
                    # Increase resources for deep search
                    self.max_pop = orig_pop * 3
                    self.max_gen = orig_gen * 2
                    
                    # Re-run single target distillation
                    eq, mask, conf = self._distill_single_target(i, X_norm, Y_norm, Y_norm.shape[1], latent_states.shape[1], is_hamiltonian=is_hamiltonian_distill)
                    
                    # Restore original settings
                    self.max_pop = orig_pop
                    self.max_gen = orig_gen
                    
                    # Final check after deep search
                    eq_str = str(eq)
                    used_features = [int(f) for f in re.findall(r'[Xx](\d+)', eq_str)]
                    has_q = any(f in q_indices for f in used_features)
                    has_p = any(f in p_indices for f in used_features)
                    
                    if not (has_q and has_p):
                        print(f"  -> DEEP SEARCH also failed physicality gate. Invalidating result.")
                        conf = 0.0
            
            equations.append(eq)
            self.feature_masks.append(mask)
            self.confidences.append(conf)

        return equations


class OptimizedExpressionWrapper:
    """
    Wrapper for optimized expressions that maintains compatibility with gplearn programs.
    """
    def __init__(self, expr_str, original_program):
        self.expr_str = expr_str
        self.original_program = original_program
        
        # Parse the expression for execution
        try:
            import sympy as sp
            self._sympy_expr = sp.sympify(expr_str)
            # Dynamically identify all symbols like X0, X1 or x0, x1
            self._symbols = sorted(list(self._sympy_expr.free_symbols), key=lambda s: str(s))
            self._lambdified = sp.lambdify(self._symbols, self._sympy_expr, modules=['numpy'])
            
            # Count actual terms for length_
            if hasattr(self._sympy_expr, 'args'):
                self.length_ = len(self._sympy_expr.args) if self._sympy_expr.is_Add else 1
            else:
                self.length_ = 1
        except Exception as e:
            self._sympy_expr = None
            self._lambdified = None
            self._symbols = []
            self.length_ = getattr(original_program, 'length_', 1)
        
        self.depth_ = getattr(original_program, 'depth_', 2)

    def execute(self, X):
        if self._lambdified is not None:
            try:
                # Ensure X is 2D for consistent processing
                if X.ndim == 1:
                    X = X.reshape(1, -1)

                # Map the required symbols to the columns of X
                # Symbols are expected to be named X{i} or x{i}
                import re
                args = []
                for sym in self._symbols:
                    match = re.search(r'(\d+)', str(sym))
                    if match:
                        idx = int(match.group(1))
                        if idx < X.shape[1]:
                            args.append(X[:, idx])
                        else:
                            args.append(np.zeros(X.shape[0]))
                    else:
                        args.append(np.zeros(X.shape[0]))

                result = self._lambdified(*args)
                result = np.asarray(result)

                # Ensure result is always a 1D numpy array with same length as input samples
                result = np.asarray(result)

                # If result is a scalar, convert to 1D array with same length as input samples
                if result.ndim == 0:
                    result = np.full(X.shape[0], result)
                elif result.ndim == 1:
                    # If result has different length than expected, handle appropriately
                    if result.shape[0] != X.shape[0]:
                        if result.shape[0] == 1:
                            # If we have a single result but multiple input samples, broadcast it
                            result = np.full(X.shape[0], result[0])
                        else:
                            # If result has more elements than expected, truncate
                            result = result[:X.shape[0]]
                            # If result has fewer elements than expected, pad
                            if result.shape[0] < X.shape[0]:
                                result = np.pad(result, (0, X.shape[0] - result.shape[0]), mode='edge')
                else:
                    # If result is multi-dimensional, flatten to 1D and handle length
                    # This is important: if result has shape (n_samples, 1), flatten to (n_samples,)
                    if result.shape[1] == 1 and len(result.shape) == 2:
                        result = result.flatten()
                    else:
                        result = result.flatten()
                        if result.shape[0] != X.shape[0]:
                            if result.shape[0] == 1:
                                result = np.full(X.shape[0], result[0])
                            else:
                                result = result[:X.shape[0]]
                                if result.shape[0] < X.shape[0]:
                                    result = np.pad(result, (0, X.shape[0] - result.shape[0]), mode='edge')

                return result
            except Exception as e:
                # Fallback to original program if execution fails
                print(f"OptimizedExpressionWrapper execute failed: {e}")
                return self.original_program.execute(X)
        else:
            return self.original_program.execute(X)

    def __str__(self):
        return self.expr_str