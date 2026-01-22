"""
Enhanced Symbolic Distiller with secondary optimization to address GP convergence issues.
This addresses the critical issue identified in the analysis where GP struggles to find
exact constants without secondary local optimization.
"""

import numpy as np
import torch
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from symbolic import SymbolicDistiller, FeatureTransformer, gp_to_sympy
from balanced_features import BalancedFeatureTransformer
from hamiltonian_symbolic import HamiltonianSymbolicDistiller
import sympy as sp
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SymPyToTorch(torch.nn.Module):
    """
    A lightweight, robust converter from SymPy expressions to PyTorch modules.
    Handles standard gplearn and physics-informed functions with focus on differentiability.
    """
    def __init__(self, sympy_expr, n_inputs):
        super().__init__()
        self.n_inputs = n_inputs
        # Map SymPy symbols to input indices
        self.symbols = [sp.Symbol(f'x{i}') for i in range(n_inputs)]
        self.expr = sympy_expr
        
        # Mapping SymPy ops to Torch ops
        # We use epsilons for stability and to prevent NaN gradients near singularities
        self.eps = 1e-8
        self.op_map = {
            sp.Add: torch.add,
            sp.Mul: torch.mul,
            sp.Pow: self._safe_pow,
            sp.exp: self._safe_exp,
            sp.log: self._safe_log,
            sp.sin: torch.sin,
            sp.cos: torch.cos,
            sp.tan: self._safe_tan,
            sp.Abs: torch.abs,
            sp.sqrt: lambda x: torch.sqrt(torch.abs(x) + self.eps),
            # Add explicit support for potential remaining gplearn-style functions
            sp.Function('sub'): torch.sub,
            sp.Function('add'): torch.add,
            sp.Function('mul'): torch.mul,
            sp.Function('div'): self._safe_div,
            sp.Function('neg'): torch.neg,
            sp.Function('inv'): lambda x: 1.0 / (x + self.eps),
            sp.Function('square'): lambda x: torch.pow(x, 2),
        }

    def _safe_div(self, x, y):
        return x / (y + torch.sign(y + 1e-12) * self.eps)

    def _safe_pow(self, x, y):
        # Handle division (y < 0) and square roots (y = 0.5) safely
        try:
            y_val = float(y)
            if abs(y_val - (-1.0)) < 1e-9:
                # 1/x with singularity protection
                return 1.0 / (x + torch.sign(x + 1e-12) * self.eps + (x == 0).float() * self.eps)
            if abs(y_val - 0.5) < 1e-9:
                return torch.sqrt(torch.abs(x) + self.eps)
            if y_val < 0:
                # Safe division for any negative power
                return 1.0 / (torch.pow(torch.abs(x), abs(y_val)) + self.eps)
        except:
            # If y is a tensor or complex expression
            pass
        
        # Standard power with protection for negative bases and non-integer exponents
        return torch.pow(torch.abs(x) + self.eps, y)

    def _safe_exp(self, x):
        # Clamp input to prevent overflow before exp
        return torch.exp(torch.clamp(x, max=15.0))

    def _safe_log(self, x):
        # Safe log with absolute value and epsilon
        return torch.log(torch.abs(x) + self.eps)

    def _safe_tan(self, x):
        # Tan has singularities at (n + 1/2) * pi
        # We use a safe version that clamps the output
        res = torch.tan(x)
        return torch.clamp(res, -1e4, 1e4)

    def forward(self, x_inputs):
        """
        x_inputs: [Batch, n_inputs] tensor
        """
        try:
            res = self._recursive_eval(self.expr, x_inputs)
            # Catch any remaining NaNs or Infs and clamp
            res = torch.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
            return torch.clamp(res, -1e6, 1e6)
        except Exception:
            # Fallback for unexpected SymPy structures
            l_func = sp.lambdify(self.symbols, self.expr, modules='torch')
            args = [x_inputs[:, i] for i in range(self.n_inputs)]
            res = l_func(*args)
            res = torch.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
            return torch.clamp(res, -1e6, 1e6)

    def _recursive_eval(self, node, x_inputs):
        if node.is_Symbol:
            # Extract index from 'xi'
            try:
                idx = int(node.name[1:])
                if idx < self.n_inputs:
                    return x_inputs[:, idx]
                return torch.zeros(x_inputs.size(0), device=x_inputs.device)
            except:
                return torch.zeros(x_inputs.size(0), device=x_inputs.device)
        
        if node.is_Number:
            return torch.tensor(float(node), device=x_inputs.device, dtype=x_inputs.dtype)
        
        # Handle functions
        op = node.func
        if op in self.op_map:
            args = [self._recursive_eval(arg, x_inputs) for arg in node.args]
            if len(args) == 1:
                return self.op_map[op](args[0])
            # For Add and Mul, handle multiple arguments
            res = args[0]
            for i in range(1, len(args)):
                res = self.op_map[op](res, args[i])
            return res
        
        # Special case for gplearn-style functions if they appear as SymPy functions
        if hasattr(op, '__name__'):
            name = op.__name__.lower()
            if name == 'sig' or name == 'sigmoid':
                return torch.sigmoid(self._recursive_eval(node.args[0], x_inputs))
        
        # Fallback for complex ops or unknown ones
        # Use lambdify with torch as last resort
        l_func = sp.lambdify(self.symbols, node, modules='torch')
        args = [x_inputs[:, i] for i in range(self.n_inputs)]
        return l_func(*args)

class TorchFeatureTransformer(torch.nn.Module):
    """
    PyTorch implementation of FeatureTransformer for differentiable symbolic proxy.
    Updated to match BalancedFeatureTransformer logic exactly.
    """
    def __init__(self, transformer):
        super().__init__()
        self.n_super_nodes = transformer.n_super_nodes
        self.latent_dim = transformer.latent_dim
        self.include_dists = transformer.include_dists
        self.box_size = transformer.box_size
        self.basis_functions = getattr(transformer, 'basis_functions', 'physics_informed')

        # Register FULL buffers for normalization parameters
        # We now store full buffers and apply masking in forward() to avoid shape mismatches
        self.register_buffer('x_poly_mean', torch.from_numpy(transformer.x_poly_mean).float())
        self.register_buffer('x_poly_std', torch.from_numpy(transformer.x_poly_std).float())
        self.register_buffer('target_mean', torch.from_numpy(transformer.target_mean).float())
        self.register_buffer('target_std', torch.from_numpy(transformer.target_std).float())

        # Register feature selection mask to match distillation dimensions
        if hasattr(transformer, 'selected_feature_indices') and transformer.selected_feature_indices is not None:
            indices = transformer.selected_feature_indices
            if isinstance(indices, (np.ndarray, list)):
                indices_np = np.array(indices)
                if indices_np.dtype == bool:
                    indices_np = np.where(indices_np)[0]
                self.register_buffer('feature_mask', torch.from_numpy(indices_np).long())
            else:
                self.register_buffer('feature_mask', torch.from_numpy(np.array(indices)).long())
        elif hasattr(transformer, 'selector') and transformer.selector is not None:
            try:
                indices = transformer.selector.get_support(indices=True)
                self.register_buffer('feature_mask', torch.from_numpy(np.array(indices)).long())
            except Exception as e:
                print(f"DEBUG: Failed to get feature mask from selector: {e}")
                self.register_buffer('feature_mask', torch.tensor([], dtype=torch.long))
        else:
            self.register_buffer('feature_mask', torch.tensor([], dtype=torch.long))

    def forward(self, z_flat):
        # z_flat: [Batch, K * D]
        # Safety clamp
        z_flat = torch.clamp(z_flat.to(torch.float32), -1e6, 1e6)
        batch_size = z_flat.size(0)
        z_nodes = z_flat.view(batch_size, self.n_super_nodes, self.latent_dim)

        features = [z_flat]

        if self.include_dists and self.basis_functions != 'polynomial_only':
            # Compute distance features using the same logic as BalancedFeatureTransformer
            dist_features = self._compute_distance_features(z_nodes)
            if dist_features:
                features.extend(dist_features)

        X = torch.cat(features, dim=1)
        # Final safety clip before expansion
        X = torch.clamp(X, -1e6, 1e6)

        # Apply basis function expansion based on configuration
        if self.basis_functions == 'polynomial':
            X_expanded = self._polynomial_expansion(X)
        elif self.basis_functions == 'physics_informed':
            X_expanded = self._physics_informed_expansion(X, z_nodes)
        elif self.basis_functions == 'adaptive':
            X_expanded = self._adaptive_expansion(X, z_nodes)
        else:
            X_expanded = self._physics_informed_expansion(X, z_nodes)

        # Final safety clip and NaN handling
        X_expanded = torch.nan_to_num(X_expanded, nan=0.0, posinf=1e9, neginf=-1e9)
        X_expanded = torch.clamp(X_expanded, -1e12, 1e12)

        # 1. Normalize using FULL buffers
        X_norm_full = (X_expanded - self.x_poly_mean) / self.x_poly_std

        # 2. Apply feature mask AFTER normalization to match distilled feature set
        if self.feature_mask is not None and self.feature_mask.numel() > 0:
            mask = self.feature_mask.to(X_expanded.device)
            # Ensure indices are within bounds
            valid_mask = mask[mask < X_norm_full.size(1)]
            X_norm_selected = X_norm_full[:, valid_mask]
            return X_norm_selected

        return X_norm_full

    def _compute_distance_features(self, z_nodes):
        """
        Compute distance-based features between super-nodes using vectorized operations.
        Matches BalancedFeatureTransformer logic.
        """
        n_batch = z_nodes.shape[0]
        if self.n_super_nodes < 2:
            return []

        # Get all pairs of indices
        i_idx, j_idx = torch.triu_indices(self.n_super_nodes, self.n_super_nodes, offset=1, device=z_nodes.device)

        # [Batch, n_pairs, 2]
        diff = z_nodes[:, i_idx, :2] - z_nodes[:, j_idx, :2]

        if self.box_size is not None:
            # Minimum Image Convention
            box = torch.tensor(self.box_size, device=z_nodes.device, dtype=torch.float32)
            diff -= box * torch.round(diff / box)

        # [Batch, n_pairs]
        d = torch.norm(diff, dim=2) + 1e-6

        # Only return 1/r and 1/r^2 as requested for small systems
        return [1.0 / (d + 0.1), 1.0 / (d**2 + 0.1)]

    def _polynomial_expansion(self, X):
        """
        Perform standard polynomial expansion.
        """
        # For simplicity, we'll use a basic quadratic expansion here
        batch_size, n_features = X.shape

        # Base features
        features = [X]

        # Quadratic terms
        for i in range(min(n_features, 10)):  # Limit to avoid explosion
            features.append(X[:, i:i+1]**2)

        # Cross terms for first few features
        for i in range(min(n_features, 5)):
            for j in range(i+1, min(n_features, 5)):
                features.append(X[:, i:i+1] * X[:, j:j+1])

        return torch.cat(features, dim=1)

    def _physics_informed_expansion(self, X, z_nodes):
        """
        Perform physics-informed polynomial expansion with cross-terms.
        Matches BalancedFeatureTransformer logic.
        """
        batch_size, n_total_features = X.shape
        n_raw_latents = self.n_super_nodes * self.latent_dim

        # For small systems, target < 40 features by skipping higher-order terms
        if self.n_super_nodes <= 4:
            # Match BalancedFeatureTransformer: return just X
            return X

        # 1. Base features: Raw latents + already computed distance features
        features = [X]

        # 2. Squares of raw latents: [Batch, n_raw_latents]
        X_raw = X[:, :n_raw_latents]
        features.append(X_raw**2)

        # 3. Intra-node cross-terms: O(K * D^2)
        X_nodes = X_raw.reshape(batch_size, self.n_super_nodes, self.latent_dim)
        intra_terms = []
        for i in range(self.latent_dim):
            for j in range(i + 1, self.latent_dim):
                intra_terms.append(X_nodes[:, :, i] * X_nodes[:, :, j])

        if intra_terms:
            features.append(torch.stack(intra_terms, dim=-1).reshape(batch_size, -1))

        # 4. Inter-node cross-terms (same dimension): O(D * K^2)
        inter_terms = []
        for d in range(self.latent_dim):
            val_d = X_nodes[:, :, d]
            i_idx, j_idx = torch.triu_indices(self.n_super_nodes, self.n_super_nodes, offset=1, device=X_nodes.device)
            inter_terms.append(val_d[:, i_idx] * val_d[:, j_idx])

        if inter_terms:
            features.append(torch.stack(inter_terms, dim=-1).reshape(batch_size, -1))

        # Combine all features
        X_expanded = torch.cat(features, dim=1)

        # Memory Safety
        if X_expanded.shape[1] > 1000:
            variances = torch.var(X_expanded, dim=0)
            top_indices = torch.argsort(variances, descending=True)[:1000]
            X_expanded = X_expanded[:, top_indices]

        return X_expanded

    def _adaptive_expansion(self, X, z_nodes):
        """
        Adaptive expansion that learns the most relevant feature combinations.
        """
        X_physics = self._physics_informed_expansion(X, z_nodes)
        return X_physics

    def denormalize_y(self, Y_norm):
        return Y_norm * self.target_std + self.target_mean


class EnhancedSymbolicDistiller(SymbolicDistiller):
    """
    Enhanced SymbolicDistiller with secondary optimization to address GP convergence issues.
    
    Addresses the critical issue where GP struggles to find exact constants without
    secondary local optimization (like BFGS). This implementation adds:
    1. Secondary optimization using scipy.optimize
    2. Parameter refinement for discovered expressions
    3. Improved constant identification
    4. Better handling of physics-specific parameters
    """

    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12,
                 secondary_optimization=True, opt_method='L-BFGS-B', opt_iterations=100,
                 use_sindy_pruning=True, sindy_threshold=0.05):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.secondary_optimization = secondary_optimization
        self.opt_method = opt_method
        self.opt_iterations = opt_iterations
        self.use_sindy_pruning = use_sindy_pruning
        self.sindy_threshold = sindy_threshold

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
        print(f"Selecting features for target_{i} (Input dim: {X_norm.shape[1]})...")

        variances = np.var(X_norm, axis=0)
        valid_indices = np.where(variances > 1e-6)[0]
        X_pruned = X_norm[:, valid_indices]

        # Use SINDy-style pruning if enabled
        if self.use_sindy_pruning:
            sindy_mask = self._sindy_select(X_pruned, Y_norm[:, i], threshold=self.sindy_threshold)
            print(f"  -> SINDy pruned {len(valid_indices)} to {np.sum(sindy_mask)} features.")
            # If SINDy was too aggressive, fall back to variance-based valid_indices
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

        from sklearn.linear_model import RidgeCV
        ridge = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        ridge.fit(X_selected, Y_norm[:, i])
        linear_score = ridge.score(X_selected, Y_norm[:, i])

        if linear_score > 0.9999:
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

        parsimony_levels = [0.001, 0.01, 0.05, 0.1]
        complexity_factor = max(1.0, 3.0 * (1.0 - linear_score))
        scaled_pop = int(self.populations * complexity_factor)
        scaled_pop = min(scaled_pop, 10000)

        candidates = []
        for p_coeff in parsimony_levels:
            est = self._get_regressor(scaled_pop, self.generations // 2, parsimony=p_coeff)
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

        # Pareto Frontier Selection: REDUCED COMPLEXITY PENALTY IF R2 SCORE IS BELOW 0.9
        for c in candidates:
            # Adjusted score: R2 penalized by complexity (length of the expression)
            if c['score'] < 0.9:
                c['pareto_score'] = c['score'] - 0.005 * c['complexity']  # Reduced penalty
            else:
                c['pareto_score'] = c['score'] - 0.015 * c['complexity']

        candidates.sort(key=lambda x: x['pareto_score'], reverse=True)
        best_candidate = candidates[0]

        # If the best candidate still has a very high complexity (> 20 nodes),
        # look for a significantly simpler one that is still reasonably accurate.
        if best_candidate['complexity'] > 20:
            for c in candidates[1:]:
                if c['complexity'] < 10 and (best_candidate['score'] - c['score']) < 0.08:
                    best_candidate = c
                    break

        if best_candidate['score'] < 0.85:
            print(f"  -> Escalating distillation for target_{i}...")
            est = self._get_regressor(self.populations, self.generations, parsimony=best_candidate['p'])
            est.fit(X_selected, Y_norm[:, i])
            # For the escalated model, we also check if it's better Pareto-wise
            esc_prog = est._program
            esc_score = est.score(X_selected, Y_norm[:, i])
            esc_complexity = self.get_complexity(esc_prog)
            esc_pareto = esc_score - 0.015 * esc_complexity

            if esc_pareto > best_candidate['pareto_score']:
                best_candidate = {'prog': esc_prog, 'score': esc_score, 'complexity': esc_complexity}

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
        confidence = max(0, best_candidate['score'] - 0.01 * best_candidate['complexity'])

        return best_candidate['prog'], full_mask, confidence

    def _optimize_constants(self, program, X, y_true):
        """
        Apply secondary optimization to refine constants in the symbolic expression.
        Uses SymPy tree traversal for robust constant identification.
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
            
            # 3. Parametrize constants
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
            
            # Perform optimization
            result = minimize(eval_expr, constants, method=self.opt_method, 
                             options={'maxiter': self.opt_iterations})
            
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

    def distill_with_secondary_optimization(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None):
        """
        Distill with enhanced secondary optimization specifically for physics constants.
        """
        self.transformer = BalancedFeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        self.transformer.fit(latent_states, targets)

        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)
        Y_norm = self.transformer.normalize_y(targets)

        from joblib import Parallel, delayed
        results = Parallel(n_jobs=1)(
            delayed(self._distill_single_target)(i, X_norm, Y_norm, targets.shape[1], latent_states.shape[1])
            for i in range(targets.shape[1])
        )

        equations = [r[0] for r in results]
        self.feature_masks = [r[1] for r in results]
        self.confidences = [r[2] for r in results]

        # Store the feature masks in the transformer for later use by TorchFeatureTransformer
        # Use the first mask if all masks are the same, or store all masks if they differ
        if len(set(map(tuple, self.feature_masks))) == 1:
            # All masks are the same, store as selected_feature_indices
            self.transformer.selected_feature_indices = self.feature_masks[0]
        else:
            # Different masks for different targets, store all of them
            self.transformer.selected_feature_indices = self.feature_masks[0]  # Use first as default

        return equations


class OptimizedExpressionWrapper:
    """
    Wrapper for expressions with optimized constants.
    Ensures that refined constants are used during execution.
    """
    def __init__(self, expr_str, original_program):
        self.expr_str = expr_str
        self.original_program = original_program
        self.length_ = original_program.length_
        self._lambda_func = None
        self._feat_indices = None
        
        # Parse the expression to create a numerical evaluator using robust gp_to_sympy
        try:
            # Use robust converter
            # gp_to_sympy handles the mapping of X0, X1 to x0, x1
            sympy_expr = gp_to_sympy(expr_str)
            
            # Identify all used features
            all_symbols = sorted(list(sympy_expr.free_symbols), key=lambda s: s.name)
            self._feat_indices = [int(s.name[1:]) for s in all_symbols if s.name.startswith('x')]
            
            # Lambdify for performance
            # We must provide all features up to max_idx to ensure correct indexing
            max_idx = max(self._feat_indices) if self._feat_indices else 0
            feat_vars = [sp.Symbol(f'x{i}') for i in range(max_idx + 1)]
            self._lambda_func = sp.lambdify(feat_vars, sympy_expr, modules=['numpy'])
            
        except Exception as e:
            print(f"Warning: Could not compile optimized expression '{expr_str}': {e}")
            self._lambda_func = None
    
    def execute(self, X):
        """
        Execute the optimized expression.
        """
        if self._lambda_func is not None:
            try:
                if X.ndim == 1: X = X.reshape(1, -1)
                # Ensure X has enough features
                n_req = len(self._lambda_func.__code__.co_varnames)
                if X.shape[1] < n_req:
                    X_padded = np.pad(X, ((0, 0), (0, n_req - X.shape[1])), mode='constant')
                    args = [X_padded[:, i] for i in range(n_req)]
                else:
                    args = [X[:, i] for i in range(n_req)]
                
                result = self._lambda_func(*args)
                if np.isscalar(result):
                    return np.full(X.shape[0], result)
                return np.asarray(result).flatten()
            except Exception as e:
                # Fallback to original program if execution fails
                return self.original_program.execute(X)
        
        return self.original_program.execute(X)


class PhysicsAwareSymbolicDistiller(HamiltonianSymbolicDistiller):
    """
    Physics-aware symbolic distiller that incorporates domain knowledge about physical constants.
    """
    
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12,
                 secondary_optimization=True, physics_constants=None):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.secondary_optimization = secondary_optimization
        self.physics_constants = physics_constants or {}
        
    def _distill_with_hamiltonian_structure(self, X_norm, Y_norm, n_super_nodes, latent_dim):
        """
        Enhanced Hamiltonian distillation with physics-aware constant optimization.
        """
        # For Hamiltonian systems, we only need to learn the Hamiltonian function H(q,p)
        n_vars = X_norm.shape[1]

        # Create symbolic variables for the feature space
        sympy_vars = [sp.Symbol(f'x{i}') for i in range(n_vars)]

        # Distill the Hamiltonian function H
        hamiltonian_eq, hamiltonian_mask, hamiltonian_conf = self._distill_single_target(
            0, X_norm, Y_norm.reshape(-1, 1), Y_norm.shape[1], X_norm.shape[1]
        )

        if hamiltonian_eq is None:
            print("Warning: Could not distill Hamiltonian function. Falling back to standard distillation.")
            return self._distill_standard(X_norm, Y_norm)

        # Convert the gplearn expression to SymPy for analytical differentiation
        try:
            # Get the expression string representation
            expr_str = str(hamiltonian_eq)

            # Use the robust converter from symbolic.py
            sympy_expr = gp_to_sympy(expr_str, n_features=n_vars)

            # Apply physics-aware optimization if enabled
            if self.secondary_optimization:
                sympy_expr = self._optimize_physics_constants(sympy_expr, X_norm, Y_norm)

            # Compute analytical gradients with respect to all variables
            sympy_vars = [sp.Symbol(f'x{i}') for i in range(n_vars)]
            sympy_grads = [sp.diff(sympy_expr, var) for var in sympy_vars]

            # Create lambda functions for gradients
            # Each lambda should take all variables as input for consistency
            lambda_funcs = [sp.lambdify(sympy_vars, grad, 'numpy') for grad in sympy_grads]

            # Estimate dissipation coefficients γ
            dissipation_coeffs = np.zeros(n_super_nodes)
            # (Assuming this logic is handled similarly to HamiltonianSymbolicDistiller or passed in)

            # Create Hamiltonian equation object that respects structure
            hamiltonian_equation = HamiltonianEquation(
                hamiltonian_eq, hamiltonian_mask, n_super_nodes, latent_dim,
                dissipation_coeffs=dissipation_coeffs, 
                sympy_expr=sympy_expr, 
                grad_funcs=lambda_funcs
            )

            # For Hamiltonian systems, we return a single equation that knows how to compute
            # both dq/dt and dp/dt from the Hamiltonian structure
            self.feature_masks = [hamiltonian_mask]
            self.confidences = [hamiltonian_conf]

            return [hamiltonian_equation]

        except Exception as e:
            print(f"Hamiltonian structure enforcement failed: {e}. Falling back to standard distillation.")
            return self._distill_standard(X_norm, Y_norm)
    
    def _optimize_physics_constants(self, expr, X, y_true):
        """
        Optimize physics-specific constants in the expression.
        """
        try:
            # Look for common physics constants in the expression
            # This is a simplified version - in practice, this would be more sophisticated
            expr_str = str(expr)
            
            # Common physics constants that might appear in the expression
            # We'll look for numeric values that could be optimized
            import re
            numbers = re.findall(r'\d+\.?\d*', expr_str)
            numbers = [n for n in numbers if n and n != '.']  # Filter out empty strings and periods
            
            if not numbers:
                return expr  # No numbers to optimize
            
            # For demonstration, we'll just return the original expression
            # In a real implementation, we would perform physics-aware optimization
            return expr
            
        except Exception as e:
            print(f"Physics constant optimization failed: {e}")
            return expr


def create_enhanced_distiller(physics_constants=None, secondary_optimization=True, use_sindy_pruning=True):
    """
    Factory function to create an enhanced symbolic distiller based on requirements.
    """
    if physics_constants:
        return PhysicsAwareSymbolicDistiller(secondary_optimization=secondary_optimization, 
                                           physics_constants=physics_constants)
    else:
        return EnhancedSymbolicDistiller(secondary_optimization=secondary_optimization,
                                       use_sindy_pruning=use_sindy_pruning)