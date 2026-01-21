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

        X = np.hstack(features)
        # Final clamp of linear features before polynomial expansion
        X = np.clip(X, -1e3, 1e3)

        # NEW: Improved polynomial features with better physics-inspired combinations
        poly_features = [X]
        n_latents = self.n_super_nodes * self.latent_dim

        # NEW: More sophisticated polynomial features
        max_features_for_poly = min(20, n_latents)  # Increased limit for better representation

        for i in range(max_features_for_poly):
            # Quadratic terms
            poly_features.append((X[:, i:i+1]**2))
            
            node_idx = i // self.latent_dim
            dim_idx = i % self.latent_dim
            
            # Cross terms with expanded range
            for j in range(i + 1, min(i + 6, (node_idx + 1) * self.latent_dim)):  # Increased cross terms per feature
                poly_features.append(X[:, i:i+1] * X[:, j:j+1])
                
            # Cross terms with other nodes
            for other_node in range(node_idx + 1, min(node_idx + 5, self.n_super_nodes)):  # Increased other nodes
                other_idx = other_node * self.latent_dim + dim_idx
                poly_features.append(X[:, i:i+1] * X[:, other_idx:other_idx+1])
                
            # NEW: Cubic terms for important features
            if i % 3 == 0:  # Every third feature gets cubic term
                poly_features.append((X[:, i:i+1]**3))

        res = np.hstack(poly_features)
        return np.clip(res, -1e6, 1e6) # Final safety clamp


class ImprovedSymbolicDistiller(SymbolicDistiller):
    """
    Improved Symbolic Distiller with enhanced feature selection and equation discovery.
    """

    def __init__(self, populations=1500, generations=40, stopping_criteria=0.001, max_features=15,
                 secondary_optimization=True, opt_method='L-BFGS-B', opt_iterations=100,
                 use_sindy_pruning=True, sindy_threshold=0.01, 
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

        # NEW: Expanded function set with physics-relevant functions
        function_set = ('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 
                        'max', 'min', 'sin', 'cos', 'tan', 'exp', 'square', 'cube')

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
            extra_trees=15,         # More trees for better exploration
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

    def _sindy_select(self, X, y, threshold=0.05, max_iter=15):
        """
        Enhanced Sequential Thresholded Least Squares (STLSQ) for SINDy-style pruning.
        """
        from sklearn.linear_model import Ridge, ElasticNet
        n_features = X.shape[1]
        mask = np.ones(n_features, dtype=bool)

        for iteration in range(max_iter):
            if not np.any(mask): 
                break
                
            # NEW: Use ElasticNet instead of Ridge for better sparsity
            model = ElasticNet(alpha=1e-5, l1_ratio=0.9, max_iter=1000)
            X_active = X[:, mask]
            
            if X_active.shape[1] == 0:
                break
                
            model.fit(X_active, y)

            # Update mask: threshold coefficients
            new_mask = np.zeros(n_features, dtype=bool)
            active_coeffs = np.abs(model.coef_)
            
            # NEW: Adaptive thresholding based on coefficient distribution
            if len(active_coeffs) > 0:
                adaptive_threshold = max(threshold, np.percentile(active_coeffs, 70))  # Top 30% kept
                new_mask[mask] = active_coeffs > adaptive_threshold
            else:
                new_mask[mask] = True

            if np.array_equal(mask, new_mask): 
                break
            mask = new_mask

        return mask

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

    def _distill_single_target(self, i, X_norm, Y_norm, targets_shape_1, latent_states_shape_1):
        """
        Improved single target distillation with enhanced validation and selection.
        """
        print(f"Selecting features for target_{i} (Input dim: {X_norm.shape[1]})...")

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

        # NEW: Enhanced linear fit check with multiple metrics
        from sklearn.linear_model import RidgeCV, LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        
        # Try both simple linear and polynomial fits
        ridge = RidgeCV(alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        ridge.fit(X_selected, Y_norm[:, i])
        linear_score = ridge.score(X_selected, Y_norm[:, i])
        
        # NEW: Try a simple quadratic model as well
        try:
            quad_model = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('ridge', RidgeCV(alphas=[1e-6, 1e-4, 1e-2, 1]))
            ])
            quad_model.fit(X_selected, Y_norm[:, i])
            quad_score = quad_model.score(X_selected, Y_norm[:, i])
        except:
            quad_score = linear_score  # Fallback if quadratic fails

        best_linear_score = max(linear_score, quad_score)
        
        # NEW: More flexible threshold for linear models
        if best_linear_score > 0.90:  # More flexible threshold
            print(f"  -> Target_{i}: Good linear fit (R2={best_linear_score:.3f}). Using linear model.")
            
            # NEW: Return the best performing model (linear or quadratic)
            if quad_score > linear_score:
                class QuadraticProgram:
                    def __init__(self, model, feature_indices):
                        self.model = model
                        self.length_ = 1
                        self.feature_indices = feature_indices
                        
                        # Create a string representation (simplified)
                        self.expr_str = f"QuadraticModel_{i}"

                    def execute(self, X):
                        if X.ndim == 1: X = X.reshape(1, -1)
                        return self.model.predict(X)

                    def __str__(self):
                        return self.expr_str
                        
                selected_indices = np.where(full_mask)[0]
                return QuadraticProgram(quad_model, selected_indices), full_mask, quad_score
            else:
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

        # NEW: Enhanced GP search with better parameters
        parsimony_levels = [0.0001, 0.001, 0.01]  # More granular levels
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

        for i in range(targets.shape[1]):
            eq, mask, conf = self._distill_single_target(i, X_norm, Y_norm, targets.shape[1], latent_states.shape[1])
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
        self.length_ = getattr(original_program, 'length_', len(expr_str.split()))
        self.depth_ = getattr(original_program, 'depth_', 2)
        
        # Parse the expression for execution
        try:
            import sympy as sp
            self._sympy_expr = sp.sympify(expr_str)
            self._lambdified = sp.lambdify([sp.Symbol(f'X{i}') for i in range(50)], 
                                          self._sympy_expr, modules=['numpy'])
        except:
            self._sympy_expr = None
            self._lambdified = None

    def execute(self, X):
        if self._lambdified is not None:
            try:
                # Prepare arguments for the lambdified function
                args = []
                for i in range(min(X.shape[1], 50)):  # Up to 50 features
                    if X.ndim == 1:
                        args.append(X[i])
                    else:
                        args.append(X[:, i])
                
                # Pad with zeros if needed
                for i in range(len(args), 50):
                    if X.ndim == 1:
                        args.append(0.0)
                    else:
                        args.append(np.zeros(X.shape[0]))
                
                result = self._lambdified(*args)
                return np.asarray(result)
            except:
                # Fallback to original program if execution fails
                return self.original_program.execute(X)
        else:
            return self.original_program.execute(X)

    def __str__(self):
        return self.expr_str