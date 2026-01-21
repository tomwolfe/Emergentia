"""
Optimized Symbolic Distillation Module

This module implements performance-optimized symbolic regression to speed up
the symbolic distillation process while maintaining accuracy.
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
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class OptimizedFeatureTransformer(FeatureTransformer):
    """
    Optimized feature transformer with reduced computational overhead.
    """
    def __init__(self, n_super_nodes, latent_dim, include_dists=True, box_size=None):
        super().__init__(n_super_nodes, latent_dim, include_dists, box_size)
        
    def transform(self, z_flat):
        """
        Optimized feature transformation with reduced computation.
        Input: z_flat [batch_size, n_features] where n_features = n_nodes * n_dims
        """
        # Clamp input for stability
        z_flat = np.clip(z_flat, -1e6, 1e6)
        batch_size = z_flat.shape[0]

        # Calculate n_nodes and n_dims from the flattened shape
        total_features = z_flat.shape[1]
        n_nodes = self.n_super_nodes
        n_dims = self.latent_dim

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

            # New physics-informed basis functions
            exp_dist = np.exp(-np.clip(dists_flat, 0, 20))
            screened_coulomb = exp_dist / (dists_flat + 0.1)
            log_dist = np.log(dists_flat + 1.0)

            features.extend([dists_flat, inv_dists_flat, inv_sq_dists_flat, exp_dist, screened_coulomb, log_dist])

        X = np.hstack(features)
        # Final clamp of linear features before polynomial expansion
        X = np.clip(X, -1e3, 1e3)

        # Limit polynomial features to reduce computation
        poly_features = [X]
        n_latents = self.n_super_nodes * self.latent_dim

        # Only compute quadratic and cross terms for a subset of features
        # to reduce combinatorial explosion
        max_features_for_poly = min(15, n_latents)  # Limit to first 15 features

        for i in range(max_features_for_poly):
            poly_features.append((X[:, i:i+1]**2))
            node_idx = i // self.latent_dim
            dim_idx = i % self.latent_dim
            # Cross terms with limited range to reduce computation
            for j in range(i + 1, min(i + 4, (node_idx + 1) * self.latent_dim)):  # Only 3 cross terms per feature in same node
                poly_features.append(X[:, i:i+1] * X[:, j:j+1])
            for other_node in range(node_idx + 1, min(node_idx + 3, self.n_super_nodes)):  # Only 2 other nodes
                other_idx = other_node * self.latent_dim + dim_idx
                poly_features.append(X[:, i:i+1] * X[:, other_idx:other_idx+1])

        res = np.hstack(poly_features)
        return np.clip(res, -1e6, 1e6) # Final safety clamp


class OptimizedSymbolicDistiller(SymbolicDistiller):
    """
    Optimized Symbolic Distiller with reduced computational complexity.
    """
    
    def __init__(self, populations=1000, generations=20, stopping_criteria=0.01, max_features=8,
                 secondary_optimization=True, opt_method='L-BFGS-B', opt_iterations=50,
                 use_sindy_pruning=True, sindy_threshold=0.1):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.secondary_optimization = secondary_optimization
        self.opt_method = opt_method
        self.opt_iterations = opt_iterations
        self.use_sindy_pruning = use_sindy_pruning
        self.sindy_threshold = sindy_threshold

    def _get_regressor(self, population_size, generations, parsimony=0.001):
        """
        Create an optimized symbolic regressor with reduced complexity.
        """
        from gplearn.genetic import SymbolicRegressor
        
        # Reduced function set to speed up evolution
        function_set = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'max', 'min')
        
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
            const_range=(-1., 1.),
            init_depth=(2, 6),
            init_method='half and half',
            function_set=function_set,
            metric='mean absolute error',
            extra_trees=10,  # Reduced from default
            tournament_size=20,  # Reduced from default
            n_jobs=1  # Use single thread to avoid overhead
        )
        return est

    def _select_features(self, X, y, max_features=None):
        """
        Optimized feature selection with reduced computational overhead.
        """
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

    def _sindy_select(self, X, y, threshold=0.05, max_iter=10):
        """
        Sequential Thresholded Least Squares (STLSQ) for SINDy-style pruning.
        """
        from sklearn.linear_model import Ridge
        n_features = X.shape[1]
        mask = np.ones(n_features, dtype=bool)

        for _ in range(max_iter):
            if not np.any(mask): break
            # Solve least squares on active features
            model = Ridge(alpha=1e-5)
            model.fit(X[:, mask], y)

            # Update mask: threshold coefficients
            new_mask = np.zeros(n_features, dtype=bool)
            active_coeffs = np.abs(model.coef_)
            # Normalize coeffs by their max to make threshold relative or use absolute
            # Absolute threshold is more standard in SINDy
            new_mask[mask] = active_coeffs > threshold

            if np.array_equal(mask, new_mask): break
            mask = new_mask

        return mask

    def _distill_single_target(self, i, X_norm, Y_norm, targets_shape_1, latent_states_shape_1):
        """
        Optimized single target distillation with reduced computation.
        """
        print(f"Selecting features for target_{i} (Input dim: {X_norm.shape[1]})...")

        # Quick variance-based filtering
        variances = np.var(X_norm, axis=0)
        valid_indices = np.where(variances > 1e-6)[0]
        X_pruned = X_norm[:, valid_indices]

        # Use SINDy-style pruning if enabled
        if self.use_sindy_pruning:
            sindy_mask = self._sindy_select(X_pruned, Y_norm[:, i], threshold=self.sindy_threshold)
            print(f"  -> SINDy pruned {len(valid_indices)} to {np.sum(sindy_mask)} features.")

            if np.sum(sindy_mask) < 2:
                print("  -> SINDy too aggressive, using standard selection.")
                mask_pruned = self._select_features(X_pruned, Y_norm[:, i])
            else:
                X_sindy = X_pruned[:, sindy_mask]
                # Further refine with standard feature selector to reach max_features
                refinement_mask = self._select_features(X_sindy, Y_norm[:, i])
                mask_pruned = np.zeros(len(valid_indices), dtype=bool)
                mask_pruned[np.where(sindy_mask)[0][refinement_mask]] = True
        else:
            mask_pruned = self._select_features(X_pruned, Y_norm[:, i])

        full_mask = np.zeros(X_norm.shape[1], dtype=bool)
        full_mask[valid_indices[mask_pruned]] = True

        X_selected = X_norm[:, full_mask]
        print(f"  -> Target_{i}: Reduced to {X_selected.shape[1]} informative variables.")

        if X_selected.shape[1] == 0:
            return np.zeros((X_norm.shape[0], 1)), full_mask, 0.0

        # Quick linear fit to check if linear model is sufficient
        from sklearn.linear_model import RidgeCV
        ridge = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        ridge.fit(X_selected, Y_norm[:, i])
        linear_score = ridge.score(X_selected, Y_norm[:, i])

        if linear_score > 0.95:  # Lowered threshold for early exit
            print(f"  -> Target_{i}: High linear fit (R2={linear_score:.3f}). Using linear model.")
            class LinearProgram:
                def __init__(self, model, feature_indices):
                    self.model = model
                    self.length_ = 1
                    self.feature_indices = feature_indices
                    # Create a string representation
                    terms = []
                    # Lower threshold for terms to 1e-6
                    if abs(model.intercept_) > 1e-6:
                        terms.append(f"{model.intercept_:.6f}")

                    # Sort coefficients to find most important ones if many are small
                    coeffs = model.coef_
                    for idx, coef in enumerate(coeffs):
                        if abs(coef) > 1e-6:
                            # Map back to original feature index
                            orig_idx = feature_indices[idx]
                            terms.append(f"mul({coef:.6f}, X{orig_idx})")

                    if not terms:
                        # If everything is extremely small, check if intercept is actually non-zero
                        self.expr_str = f"{model.intercept_:.6e}"
                    elif len(terms) == 1:
                        self.expr_str = terms[0]
                    else:
                        self.expr_str = terms[0]
                        for term in terms[1:]:
                            self.expr_str = f"add({self.expr_str}, {term})"

                def execute(self, X):
                    if X.ndim == 1: X = X.reshape(1, -1)
                    return self.model.predict(X)

                def __str__(self):
                    return self.expr_str

            # Find the indices of selected features from the full_mask
            selected_indices = np.where(full_mask)[0]
            return LinearProgram(ridge, selected_indices), full_mask, linear_score

        # Reduced complexity for GP search
        parsimony_levels = [0.001, 0.01]  # Reduced from 4 levels to 2
        complexity_factor = max(1.0, 2.0 * (1.0 - linear_score))  # Reduced factor
        scaled_pop = int(self.max_pop * complexity_factor)
        scaled_pop = min(scaled_pop, 5000)  # Reduced max population

        candidates = []
        for p_coeff in parsimony_levels:
            est = self._get_regressor(scaled_pop//2, self.max_gen//2, parsimony=p_coeff)  # Reduced both pop and gen
            try:
                # Force float64 for stability
                X_gp = X_selected.astype(np.float64)
                y_gp = Y_norm[:, i].astype(np.float64)

                est.fit(X_gp, y_gp)
                prog = est._program

                # Robust scoring: check for NaNs in prediction
                y_pred = est.predict(X_gp)
                if not np.all(np.isfinite(y_pred)):
                    # Penalize unstable models
                    score = -1.0
                else:
                    from sklearn.metrics import r2_score
                    score = r2_score(y_gp, y_pred)

                is_stable = True
                if targets_shape_1 == latent_states_shape_1:
                    is_stable = self.validate_stability(prog, X_gp[0])

                if is_stable:
                    candidates.append({'prog': prog, 'score': score, 'complexity': self.get_complexity(prog), 'p': p_coeff})
            except Exception as e:
                print(f"    ! GP search failed for p={p_coeff}: {e}")

        if not candidates:
            return None, full_mask, 0.0

        # Pareto Frontier Selection: Less aggressive penalty on complexity
        for c in candidates:
            # Adjusted score: R2 penalized by complexity (length of the expression)
            c['pareto_score'] = c['score'] - 0.01 * c['complexity']

        candidates.sort(key=lambda x: x['pareto_score'], reverse=True)
        best_candidate = candidates[0]

        # Apply secondary optimization if enabled
        if self.secondary_optimization:
            optimized_prog = self._optimize_constants(best_candidate['prog'], X_selected, Y_norm[:, i])
            if optimized_prog:
                # Evaluate the optimized program
                try:
                    y_pred = optimized_prog.execute(X_selected)
                    opt_score = 1 - ((Y_norm[:, i] - y_pred)**2).sum() / (((Y_norm[:, i] - Y_norm[:, i].mean())**2).sum() + 1e-9)

                    # Check if optimization improved the score
                    if opt_score > best_candidate['score']:
                        print(f"  -> Secondary optimization improved score from {best_candidate['score']:.3f} to {opt_score:.3f}")
                        best_candidate['prog'] = optimized_prog
                        best_candidate['score'] = opt_score
                except:
                    # If optimization failed, keep the original
                    pass

        # Confidence now accounts for both accuracy and parsimony
        confidence = max(0, best_candidate['score'] - 0.005 * best_candidate['complexity'])

        return best_candidate['prog'], full_mask, confidence

    def _optimize_constants(self, program, X, y_true):
        """
        Optimized constant refinement with reduced computational overhead.
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
            }
            for i in range(n_features):
                local_dict[f'X{i}'] = feat_vars[i]

            # 1. Parse into SymPy
            full_expr = sp.sympify(expr_str, locals=local_dict)

            # 2. Extract numeric constants
            all_atoms = full_expr.atoms(sp.Number)
            # Filter constants that are likely parameters and not indices/small integers
            constants = sorted([float(a) for a in all_atoms if not a.is_Integer or abs(a) > 5])

            if not constants:
                return program

            # 3. Parametrize constants (limit to first 10 constants to avoid huge optimization problems)
            constants = constants[:10]  # Limit to first 10 constants
            const_vars = [sp.Symbol(f'c{i}') for i in range(len(constants))]
            subs_map = {sp.Float(c): cv for c, cv in zip(constants, const_vars)}
            param_expr = full_expr.subs(subs_map)

            # 4. Lambdify for optimization
            f_lamb = sp.lambdify(const_vars + feat_vars, param_expr, modules=['numpy'])

            def eval_expr(const_vals):
                try:
                    y_pred = f_lamb(*const_vals, *[X[:, i] for i in range(n_features)])
                    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                        return float('inf')
                    return mean_squared_error(y_true, y_pred)
                except:
                    return float('inf')

            # Perform optimization with reduced iterations
            result = minimize(eval_expr, constants, method=self.opt_method,
                             options={'maxiter': min(self.opt_iterations, 20)})  # Reduced iterations

            if result.success:
                opt_consts = result.x

                # Physics-Inspired Simplification
                simplified_consts = opt_consts.copy()
                for j in range(len(simplified_consts)):
                    val = simplified_consts[j]
                    for base in [1.0, 0.5, 0.25]:
                        rounded = round(val / base) * base
                        if abs(val - rounded) < 0.12:
                            simplified_consts[j] = rounded
                            break

                mse_opt = eval_expr(opt_consts)
                mse_simple = eval_expr(simplified_consts)
                final_consts = simplified_consts if mse_simple < 1.05 * mse_opt else opt_consts

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
        Optimized distillation with reduced computational complexity.
        """
        # Use optimized transformer
        self.transformer = OptimizedFeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        self.transformer.fit(latent_states, targets)

        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)
        Y_norm = self.transformer.normalize_y(targets)

        # Process targets sequentially to reduce memory usage
        equations = []
        self.feature_masks = []
        self.confidences = []

        for i in range(targets.shape[1]):
            eq, mask, conf = self._distill_single_target(i, X_norm, Y_norm, targets.shape[1], latent_states.shape[1])
            equations.append(eq)
            self.feature_masks.append(mask)
            self.confidences.append(conf)

        return equations