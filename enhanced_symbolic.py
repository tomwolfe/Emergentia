"""
Enhanced Symbolic Distiller with secondary optimization to address GP convergence issues.
This addresses the critical issue identified in the analysis where GP struggles to find
exact constants without secondary local optimization.
"""

import numpy as np
import torch
import torch.nn as nn
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from symbolic import SymbolicDistiller, FeatureTransformer, OptimizedExpressionProgram
from symbolic_utils import gp_to_sympy
from balanced_features import BalancedFeatureTransformer
from hamiltonian_symbolic import HamiltonianSymbolicDistiller
import sympy as sp
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def get_expression_dimension(expr, feature_dims):
    """
    Recursively determines the dimension of a SymPy expression.
    Returns (L, M, T) or None if inconsistent.
    """
    if expr.is_Symbol:
        try:
            # Match x0, x1, etc.
            idx = int(str(expr)[1:])
            if idx < len(feature_dims): return feature_dims[idx]
        except: pass
        return (0, 0, 0)
    if expr.is_Number:
        return (0, 0, 0)
    
    if expr.is_Add:
        dims = [get_expression_dimension(arg, feature_dims) for arg in expr.args]
        if any(d is None for d in dims): return None
        # All non-zero dimensions must match
        non_zero_dims = [d for d in dims if any(v != 0 for v in d)]
        if not non_zero_dims: return (0, 0, 0)
        first = non_zero_dims[0]
        # Check for consistency with small tolerance for float dims
        for d in non_zero_dims:
            if not all(abs(v1 - v2) < 1e-5 for v1, v2 in zip(d, first)):
                return None
        return first
    
    if expr.is_Mul:
        dims = [get_expression_dimension(arg, feature_dims) for arg in expr.args]
        if any(d is None for d in dims): return None
        return tuple(sum(x) for x in zip(*dims))
    
    if expr.is_Pow:
        base_dim = get_expression_dimension(expr.args[0], feature_dims)
        if base_dim is None: return None
        try:
            p = float(expr.args[1])
            return tuple(v * p for v in base_dim)
        except: 
            # If power is not a number, the expression must be dimensionless
            if all(v == 0 for v in base_dim): return (0, 0, 0)
            return None
        
    # Standard functions (sin, cos, exp, log) require dimensionless arguments
    if expr.func in [sp.sin, sp.cos, sp.tan, sp.exp, sp.log]:
        arg_dim = get_expression_dimension(expr.args[0], feature_dims)
        if arg_dim is None or any(v != 0 for v in arg_dim):
            return None
        return (0, 0, 0)

    return (0, 0, 0) # Fallback for unknown (usually dimensionless)

class DimensionalGuard:
    """Enforces physical unit consistency in symbolic discovery."""
    def __init__(self, feature_dims):
        self.feature_dims = feature_dims

    def check_consistency(self, sympy_expr, selected_indices=None):
        """
        Returns True if the expression is dimensionally homogeneous.
        selected_indices: if the expression uses a subset of features (X0, X1...) 
                          mapped to original indices.
        """
        try:
            if selected_indices is not None:
                candidate_feature_dims = [self.feature_dims[idx] for idx in selected_indices]
            else:
                candidate_feature_dims = self.feature_dims
                
            dim = get_expression_dimension(sympy_expr, candidate_feature_dims)
            return dim is not None
        except Exception:
            return False

    def get_penalty(self, sympy_expr, selected_indices=None, penalty_value=0.5):
        """Returns a penalty score if inconsistent."""
        return 0.0 if self.check_consistency(sympy_expr, selected_indices) else penalty_value

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
        
        # NEW: Extract and register numeric constants as parameters
        self.constants = nn.ParameterList()
        self.const_map = {} # Maps SymPy Number to parameter index
        
        # Traverse expression to find all unique numbers
        # We use a temporary modified expression where numbers are replaced by place-holders
        self.param_expr = self._replace_numbers_with_params(sympy_expr)
        
        # Mapping SymPy ops to Torch ops
        self.eps = 1e-10
        self.op_map = {
            sp.Add: torch.add,
            sp.Mul: torch.mul,
            sp.Pow: self._safe_pow,
            sp.sin: torch.sin,
            sp.cos: torch.cos,
            sp.tan: self._safe_tan,
            sp.exp: self._safe_exp,
            sp.log: self._safe_log,
            sp.Abs: torch.abs,
        }
        
        # Pre-compile the expression for faster evaluation in forward
        try:
            self.torch_func = sp.lambdify(self.symbols + list(self.const_map.keys()), 
                                          self.param_expr, modules='torch')
        except Exception:
            self.torch_func = None

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

    def _replace_numbers_with_params(self, expr):
        """Recursively replaces numbers in SymPy expr with parameter symbols."""
        if expr.is_Number:
            # Skip very small integers or indices that are likely not physical constants
            val = float(expr)
            if expr.is_Integer and abs(val) <= 1:
                return expr
            
            # Register as parameter if not already done
            const_name = f"c{len(self.constants)}"
            self.constants.append(nn.Parameter(torch.tensor(val)))
            param_sym = sp.Symbol(const_name)
            self.const_map[param_sym] = len(self.constants) - 1
            return param_sym
        
        if not expr.args:
            return expr
        
        new_args = [self._replace_numbers_with_params(arg) for arg in expr.args]
        return expr.func(*new_args)

    def forward(self, x_inputs):
        """
        x_inputs: [Batch, n_inputs] tensor
        """
        try:
            if self.torch_func is not None:
                # Prepare arguments for lambdified function
                # First n_inputs are x0, x1, ...
                # Following are constants c0, c1, ...
                args = [x_inputs[:, i] for i in range(self.n_inputs)]
                args.extend(list(self.constants))
                res = self.torch_func(*args)
            else:
                res = self._recursive_eval(self.param_expr, x_inputs)
            
            # Catch any remaining NaNs or Infs and clamp
            res = torch.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
            res = torch.clamp(res, -1e6, 1e6)
        except Exception:
            # Fallback for unexpected SymPy structures
            l_func = sp.lambdify(self.symbols, self.expr, modules='torch')
            args = [x_inputs[:, i] for i in range(self.n_inputs)]
            res = l_func(*args)
            res = torch.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6)
            res = torch.clamp(res, -1e6, 1e6)
        
        # Ensure output is [Batch, 1]
        if res.dim() == 1:
            res = res.unsqueeze(1)
        elif res.dim() == 0:
            res = res.expand(x_inputs.size(0), 1)
        
        return res

    def _recursive_eval(self, node, x_inputs):
        if node.is_Symbol:
            # Check if it's a parameter symbol
            if node in self.const_map:
                return self.constants[self.const_map[node]]
            
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
        self.include_raw_latents = getattr(transformer, 'include_raw_latents', False)
        self.sim_type = getattr(transformer, 'sim_type', 'lj')

        # Register FULL buffers for normalization parameters
        self.register_buffer('z_mean', torch.from_numpy(transformer.z_mean).float())
        self.register_buffer('z_std', torch.from_numpy(transformer.z_std).float())
        self.register_buffer('x_poly_mean', torch.from_numpy(transformer.x_poly_mean).float())
        self.register_buffer('x_poly_std', torch.from_numpy(transformer.x_poly_std).float())
        self.register_buffer('target_mean', torch.tensor(transformer.target_mean).float())
        self.register_buffer('target_std', torch.tensor(transformer.target_std).float())

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
        
        # Center and normalize raw latents matching BalancedFeatureTransformer
        z_flat_norm = (z_flat - self.z_mean) / self.z_std
        
        z_nodes = z_flat.view(batch_size, self.n_super_nodes, self.latent_dim)

        features = []
        if self.include_raw_latents:
            features.append(z_flat_norm)

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
            X_expanded = self._physics_informed_expansion(X, z_flat_norm)
        elif self.basis_functions == 'adaptive':
            X_expanded = self._adaptive_expansion(X, z_flat_norm)
        else:
            X_expanded = self._physics_informed_expansion(X, z_flat_norm)

        # Final safety clip and NaN handling
        X_expanded = torch.nan_to_num(X_expanded, nan=0.0, posinf=1e9, neginf=-1e9)
        X_expanded = torch.clamp(X_expanded, -1e12, 1e12)

        # 1. Apply feature mask IMMEDIATELY after expansion and BEFORE normalization
        if self.feature_mask is not None and self.feature_mask.numel() > 0:
            mask = self.feature_mask.to(X_expanded.device)
            # Ensure indices are within bounds
            valid_mask = mask[mask < X_expanded.size(1)]
            X_selected = X_expanded[:, valid_mask]
            
            # 2. Mask normalization buffers accordingly (ensure they are on correct device)
            mu = self.x_poly_mean.to(X_expanded.device)[valid_mask]
            std = self.x_poly_std.to(X_expanded.device)[valid_mask]
            
            # 3. Normalize selected features
            X_norm_selected = (X_selected - mu) / std
            return X_norm_selected

        # Fallback to full normalization if no mask
        return (X_expanded - self.x_poly_mean) / self.x_poly_std

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
        
        features = []
        softening = 0.001
        
        # General spectrum of features for unbiased discovery
        # 1. Basic distance and inverse distance
        features.append(d)
        features.append(1.0 / (d + softening))
        
        # 2. Spectrum of power laws: 1/r^n
        for n in [2, 3, 4, 5, 6, 8, 10, 12, 14]:
            features.append(1.0 / (torch.pow(d, n) + softening))
            
        # 3. Short-range interaction terms (Exponential/Yukawa-like)
        features.append(torch.exp(-d))
        features.append(torch.exp(-d) / (d + softening))
        
        # 4. Logarithmic interactions (2D gravity/electrostatics)
        features.append(torch.log(d + softening))

        return features

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

    def _physics_informed_expansion(self, X, z_flat_norm):
        """
        Perform physics-informed polynomial expansion with cross-terms.
        Matches BalancedFeatureTransformer logic.
        """
        batch_size, n_total_features = X.shape
        n_raw_latents = self.n_super_nodes * self.latent_dim

        # 1. Base features: (Optional raw latents) + distance features
        features = [X]

        # 2. Squares and Transcendental of raw latents
        X_raw = z_flat_norm
        features.append(X_raw**2)
        
        # Transcendental: Sin, Cos, Log, Exp
        features.append(torch.sin(X_raw))
        features.append(torch.cos(X_raw))
        features.append(torch.log(torch.abs(X_raw) + 1e-3))
        features.append(torch.exp(torch.clamp(X_raw, -10, 2)))

        # NEW: Explicit sum of squares of momentum dims (p^2) per super-node
        d_sub = self.latent_dim // 2
        X_nodes = X_raw.reshape(batch_size, self.n_super_nodes, self.latent_dim)
        p_sq_sum = (X_nodes[:, :, d_sub:]**2).sum(dim=2) # [Batch, K]
        features.append(p_sq_sum)

        # For small systems, target < 60 features by skipping cross-terms
        if self.n_super_nodes <= 4:
            return torch.cat(features, dim=1)

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

    def _adaptive_expansion(self, X, z_flat_norm):
        """
        Adaptive expansion that learns the most relevant feature combinations.
        """
        X_physics = self._physics_informed_expansion(X, z_flat_norm)
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

    def __init__(self, populations=5000, generations=60, stopping_criteria=0.0001, max_features=10,
                 secondary_optimization=True, opt_method='L-BFGS-B', opt_iterations=200,
                 use_sindy_pruning=True, sindy_threshold=0.05, **kwargs):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.secondary_optimization = secondary_optimization
        self.opt_method = opt_method
        self.opt_iterations = opt_iterations
        self.use_sindy_pruning = use_sindy_pruning
        self.sindy_threshold = sindy_threshold
        self.extra_kwargs = kwargs
        self.all_candidates = [] # Store all candidates for analysis

    def _get_regressor(self, pop, gen, parsimony=None, n_jobs=1):
        """Override to use parsimony from extra_kwargs if provided."""
        if parsimony is None:
            parsimony = self.extra_kwargs.get('parsimony', 0.05)
        return super()._get_regressor(pop, gen, parsimony=parsimony, n_jobs=n_jobs)

    def _sindy_select(self, X, y, threshold=0.05, max_iter=10):
        """
        Sequential Thresholded Least Squares (STLSQ) for SINDy-style pruning.
        Uses relative thresholding for improved robustness.
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
            
            # Use relative threshold: max(absolute_threshold, relative_threshold * max_coeff)
            max_coeff = np.max(active_coeffs) if len(active_coeffs) > 0 else 0
            effective_threshold = max(threshold, 0.05 * max_coeff)
            
            new_mask[mask] = active_coeffs > effective_threshold
            
            if np.array_equal(mask, new_mask): break
            mask = new_mask
            
        return mask

    def _distill_single_target(self, i, X_norm, Y_norm, targets_shape_1=None, latent_states_shape_1=None, is_hamiltonian=False, skip_deep_search=False, sim_type=None):
        print(f"Selecting features for target_{i} (Input dim: {X_norm.shape[1]})...")
        # Handle cases where Y_norm might be a single target already or the full target matrix
        if Y_norm.ndim > 1 and Y_norm.shape[1] > i:
            y_target = Y_norm[:, i]
        elif Y_norm.ndim > 1 and Y_norm.shape[1] == 1:
            y_target = Y_norm[:, 0]
        else:
            y_target = Y_norm

        variances = np.var(X_norm, axis=0)
        valid_indices = np.where(variances > 1e-6)[0]
        X_pruned = X_norm[:, valid_indices]

        # Get names for valid features
        valid_names = None
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'feature_names'):
            valid_names = [self.transformer.feature_names[j] for j in valid_indices]

        # Use SINDy-style pruning if enabled
        is_quick = self.populations < 500
        if self.use_sindy_pruning:
            sindy_mask = self._sindy_select(X_pruned, y_target, threshold=self.sindy_threshold)
            print(f"  -> SINDy pruned {len(valid_indices)} to {np.sum(sindy_mask)} features.")
            # If SINDy was too aggressive, fall back to variance-based valid_indices
            if np.sum(sindy_mask) < 2:
                print("  -> SINDy too aggressive, using standard selection.")
                mask_pruned = self._select_features(X_pruned, y_target, sim_type=sim_type, feature_names=valid_names, skip_mi=is_quick)
            else:
                X_sindy = X_pruned[:, sindy_mask]
                sindy_names = [valid_names[j] for j in np.where(sindy_mask)[0]] if valid_names else None
                # Further refine with standard feature selector to reach max_features
                refinement_mask = self._select_features(X_sindy, y_target, sim_type=sim_type, feature_names=sindy_names, skip_mi=is_quick)
                mask_pruned = np.zeros(len(valid_indices), dtype=bool)
                mask_pruned[np.where(sindy_mask)[0][refinement_mask]] = True
        else:
            mask_pruned = self._select_features(X_pruned, y_target, sim_type=sim_type, feature_names=valid_names, skip_mi=is_quick)

        full_mask = np.zeros(X_norm.shape[1], dtype=bool)
        full_mask[valid_indices[mask_pruned]] = True

        X_selected = X_norm[:, full_mask]
        print(f"  -> Target_{i}: Reduced to {X_selected.shape[1]} informative variables.")
        
        # Display selected feature names if available
        if hasattr(self, 'transformer') and hasattr(self.transformer, 'feature_names'):
            names = [self.transformer.feature_names[j] for j in np.where(full_mask)[0]]
            print(f"  -> Target_{i} Selected features: {names}")

        if X_selected.shape[1] == 0:
            return np.zeros((X_norm.shape[0], 1)), full_mask, 0.0

        from sklearn.linear_model import RidgeCV
        ridge = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        ridge.fit(X_selected, y_target)
        linear_score = ridge.score(X_selected, y_target)

        if linear_score > 0.9999:
            print(f"  -> Target_{i}: High linear fit (R2={linear_score:.3f}). Using linear model.")
            
            # Find the indices of selected features from the full_mask
            selected_indices = np.where(full_mask)[0]

            class LinearProgram:
                def __init__(self, model, feature_indices): 
                    self.model = model
                    self.length_ = 1
                    self.feature_indices = feature_indices
                    # Create a string representation using GLOBAL indices
                    terms = []
                    if abs(model.intercept_) > 1e-6:
                        terms.append(f"{model.intercept_:.6f}")
                    
                    coeffs = model.coef_
                    for idx, coef in enumerate(coeffs):
                        if abs(coef) > 1e-6:
                            # Use global index from feature_indices
                            global_idx = feature_indices[idx]
                            terms.append(f"mul({coef:.6f}, X{global_idx})")
                    
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
                    # If X is the full X_norm, we need to slice it using feature_indices
                    if X.shape[1] > len(self.feature_indices):
                        return self.model.predict(X[:, self.feature_indices])
                    return self.model.predict(X)
                
                def __str__(self):
                    return self.expr_str
            
            return LinearProgram(ridge, selected_indices), full_mask, linear_score

        # OPTIMIZATION: If populations is small, reduce parsimony levels and search depth
        is_quick = self.populations < 1000
        parsimony_levels = [0.05] if is_quick else [0.01, 0.05]
        
        complexity_factor = max(1.0, 2.0 * (1.0 - linear_score))
        scaled_pop = int(self.populations * complexity_factor)
        if is_quick:
            scaled_pop = self.populations # Don't scale up if quick
        else:
            scaled_pop = min(scaled_pop, 5000)

        candidates = []
        for p_coeff in parsimony_levels:
            # For quick runs, cap generations even further
            max_gen = min(self.generations // 2, 20) if not is_quick else self.generations
            est = self._get_regressor(scaled_pop, max_gen, parsimony=p_coeff)
            try:
                # Force float64 for stability
                X_gp = X_selected.astype(np.float64)
                y_gp = y_target.astype(np.float64)
                
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
                    # Use DimensionalGuard for physical consistency penalty
                    is_consistent = True
                    if hasattr(self, 'transformer') and hasattr(self.transformer, 'feature_dims'):
                        guard = DimensionalGuard(self.transformer.feature_dims)
                        selected_indices = np.where(full_mask)[0]
                        local_dict = {f'X{j}': sp.Symbol(f'x{j}') for j in range(len(selected_indices))}
                        try:
                            candidate_sympy = sp.sympify(str(prog), locals=local_dict)
                            is_consistent = guard.check_consistency(candidate_sympy, selected_indices)
                        except:
                            is_consistent = False # Unparseable is inconsistent

                    candidate = {
                        'prog': prog, 
                        'score': score, 
                        'complexity': self.get_complexity(prog), 
                        'p': p_coeff, 
                        'target_idx': i,
                        'is_consistent': is_consistent
                    }
                    candidates.append(candidate)
                    self.all_candidates.append(candidate)
            except Exception as e:
                print(f"    ! GP search failed for p={p_coeff}: {e}")

        if not candidates:
            return None, full_mask, 0.0

        # Pareto Frontier Selection: HARSHER COMPLEXITY PENALTY & HARD DIMENSIONAL CONSTRAINT
        # First, try to only keep consistent candidates
        consistent_candidates = [c for c in candidates if c.get('is_consistent', False)]
        if consistent_candidates:
            candidates = consistent_candidates
            print(f"  -> Target_{i}: Found {len(candidates)} dimensionally consistent candidates.")
        else:
            print(f"  -> Target_{i}: WARNING: No dimensionally consistent candidates found. Using inconsistent ones.")

        for c in candidates:
            # Adjusted score: R2 penalized by complexity
            # Harsher penalty if not consistent (though we prefer consistent ones)
            penalty_scale = 0.04 if c.get('is_consistent', False) else 0.1
            c['pareto_score'] = c['score'] - penalty_scale * c['complexity']

        candidates.sort(key=lambda x: x['pareto_score'], reverse=True)
        best_candidate = candidates[0]

        # If the best candidate still has a very high complexity (> 15 nodes),
        # look for a significantly simpler one that is still reasonably accurate.
        if best_candidate['complexity'] > 15:
            for c in candidates[1:]:
                if c['complexity'] < 8 and (best_candidate['score'] - c['score']) < 0.1:
                    best_candidate = c
                    break

        if best_candidate['score'] < 0.85:
            print(f"  -> Escalating distillation for target_{i}...")
            est = self._get_regressor(self.populations, self.generations, parsimony=best_candidate['p'])
            est.fit(X_selected, y_target)
            # For the escalated model, we also check if it's better Pareto-wise
            esc_prog = est._program
            esc_score = est.score(X_selected, y_target)
            esc_complexity = self.get_complexity(esc_prog)
            esc_pareto = esc_score - 0.015 * esc_complexity

            if esc_pareto > best_candidate['pareto_score']:
                best_candidate = {'prog': esc_prog, 'score': esc_score, 'complexity': esc_complexity}

        # Apply secondary optimization if enabled
        if self.secondary_optimization:
            # We need to make sure _optimize_constants knows the global indices for remapping
            selected_indices = np.where(full_mask)[0]
            optimized_prog = self._optimize_constants(best_candidate['prog'], X_selected, y_target, global_indices=selected_indices)
            if optimized_prog:
                # Evaluate the optimized program
                try:
                    y_pred = optimized_prog.execute(X_selected)
                    opt_score = 1 - ((y_target - y_pred)**2).sum() / (((y_target - y_target.mean())**2).sum() + 1e-9)
                    
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

    def _optimize_constants(self, program, X, y_true, global_indices=None):
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
            
            if not constants and global_indices is None:
                return program
            
            if constants:
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
                
                # Perform optimization: Two-stage approach for robustness
                # 1. Global search (Differential Evolution) - reduced iterations for speed
                bounds = [(c - 2.0 * abs(c) - 0.5, c + 2.0 * abs(c) + 0.5) for c in constants]
                try:
                    res_de = differential_evolution(eval_expr, bounds, maxiter=10, popsize=5)
                    initial_guess = res_de.x if res_de.success else constants
                except:
                    initial_guess = constants

                # 2. Local refinement (L-BFGS-B)
                result = minimize(eval_expr, initial_guess, method=self.opt_method, 
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
                else:
                    optimized_expr = full_expr
            else:
                optimized_expr = full_expr
                
            # REMAP local indices (X0, X1...) to global indices
            if global_indices is not None:
                # Replace x0 with x{global_indices[0]}, etc.
                # Use a temporary mapping to avoid collisions
                remap_subs = {}
                for idx in range(len(global_indices)):
                    remap_subs[sp.Symbol(f'x{idx}')] = sp.Symbol(f'x{global_indices[idx]}')
                
                final_remapped_expr = optimized_expr.subs(remap_subs)
                return OptimizedExpressionWrapper(str(final_remapped_expr), program)
            
            return OptimizedExpressionWrapper(str(optimized_expr), program)
                
        except Exception as e:
            print(f"  -> Secondary optimization failed: {e}")
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
        results = Parallel(n_jobs=-1)(
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
    def __init__(self, expr_str, original_program=None):
        self.expr_str = expr_str
        self.original_program = original_program
        self.length_ = getattr(original_program, 'length_', len(str(expr_str)))
        self._lambda_func = None
        self._feat_indices = None
        
        # Parse the expression to create a numerical evaluator using robust gp_to_sympy
        try:
            # Use robust converter
            # gp_to_sympy handles the mapping of X0, X1 to x0, x1
            self.sympy_expr = gp_to_sympy(str(expr_str))
            
            # Identify all used features
            all_symbols = sorted(list(self.sympy_expr.free_symbols), key=lambda s: s.name)
            self._feat_indices = [int(s.name[1:]) for s in all_symbols if s.name.startswith('x')]
            
            # Lambdify for performance
            # We must provide all features up to max_idx to ensure correct indexing
            max_idx = max(self._feat_indices) if self._feat_indices else 0
            feat_vars = [sp.Symbol(f'x{i}') for i in range(max_idx + 1)]
            self._lambda_func = sp.lambdify(feat_vars, self.sympy_expr, modules=['numpy'])
            
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
                if self.original_program:
                    return self.original_program.execute(X)
                return np.zeros(X.shape[0])
        
        if self.original_program:
            return self.original_program.execute(X)
        return np.zeros(X.shape[0])

    def __str__(self):
        return self.expr_str


class EnsembleSymbolicDistiller(EnhancedSymbolicDistiller):
    """
    Implements Ensemble Distillation to eliminate spurious terms.
    Runs distillation 5 times on different data shuffles and only keeps terms
    that appear in at least 4 of the 5 runs.
    """
    def __init__(self, populations=2000, generations=40, ensemble_size=5, consensus_threshold=4, **kwargs):
        super().__init__(populations=populations, generations=generations, **kwargs)
        self.ensemble_size = ensemble_size
        self.consensus_threshold = consensus_threshold

    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None, sim_type=None, enforce_separable=True, hamiltonian=False):
        """
        Run ensemble distillation across multiple targets in parallel.
        """
        # Fit transformer once
        self.transformer = BalancedFeatureTransformer(n_super_nodes, latent_dim, box_size=box_size, 
                                                     basis_functions='physics_informed', 
                                                     include_raw_latents=True, sim_type=sim_type,
                                                     feature_selection_method='bic',
                                                     hamiltonian=hamiltonian) # Use new BIC method
        self.transformer.fit(latent_states, targets)

        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)
        Y_norm = self.transformer.normalize_y(targets)

        n_targets = targets.shape[1] if targets.ndim > 1 else 1
        final_equations = []
        
        from joblib import Parallel, delayed
        
        # Inner helper to run a single ensemble member
        def run_ensemble_member(target_idx, X_norm, yi_norm):
            idx = np.random.permutation(len(X_norm))
            X_run = X_norm[idx]
            y_run = yi_norm[idx]
            
            # OPTIMIZATION: If populations is small, reduce search depth and parsimony levels
            # and set n_jobs=1 for the regressor since we parallelize at the target level
            is_very_quick = self.populations < 500
            
            prog, _, _ = self._distill_single_target(target_idx, X_run, y_run.reshape(-1, 1), 
                                                    targets_shape_1=n_targets, 
                                                    latent_states_shape_1=latent_dim * n_super_nodes,
                                                    sim_type=sim_type)
            return prog

        self.confidences = []
        
        # 1. Prepare all tasks for global parallelization
        all_tasks = []
        for i in range(n_targets):
            yi_norm = Y_norm[:, i] if Y_norm.ndim > 1 else Y_norm
            for _ in range(self.ensemble_size):
                all_tasks.append((i, yi_norm))
        
        print(f"\n[Ensemble] Distilling {n_targets} targets with ensemble size {self.ensemble_size} (Total tasks: {len(all_tasks)})")
        
        # 2. Run all tasks in parallel
        # Use n_jobs=-1 to use all cores for the target-level parallelization
        all_progs = Parallel(n_jobs=-1)(
            delayed(run_ensemble_member)(t_idx, X_norm, y_norm) for t_idx, y_norm in all_tasks
        )
        
        # 3. Group and Process Consensus
        for i in range(n_targets):
            # Extract results for this target
            run_expressions = [all_progs[j] for j, (t_idx, _) in enumerate(all_tasks) if t_idx == i]
            run_expressions = [p for p in run_expressions if p is not None]
            
            if not run_expressions:
                final_equations.append(None)
                self.confidences.append(0.0)
                continue

            if self.ensemble_size == 1:
                final_equations.append(run_expressions[0])
                self.confidences.append(0.7)
                continue

            # 2. Extract and canonicalize terms from all runs
            term_counts = {}
            term_to_sympy = {}
            
            for prog in run_expressions:
                try:
                    expr = gp_to_sympy(str(prog))
                    # Expand and split into additive terms
                    expr = sp.expand(expr)
                    terms = sp.Add.make_args(expr)
                    
                    for term in terms:
                        # Canonicalize: separate constant from functional part
                        coeff, func_part = term.as_coeff_Mul()
                        # Use string representation of functional part as key
                        key = str(func_part)
                        term_counts[key] = term_counts.get(key, 0) + 1
                        term_to_sympy[key] = func_part
                except Exception as e:
                    pass

            # 3. Filter terms by consensus threshold
            consensus_terms = [term_to_sympy[key] for key, count in term_counts.items() 
                              if count >= self.consensus_threshold]
            
            print(f"  [Ensemble] Terms found: {len(term_counts)} | Consensus (>= {self.consensus_threshold}): {len(consensus_terms)}")

            if not consensus_terms:
                # Fallback to the best single run if no consensus
                print("  [Ensemble] No consensus terms found. Falling back to best run.")
                final_equations.append(run_expressions[0])
                self.confidences.append(0.5)
                continue

            # 4. Re-fit constants for the consensus expression
            consensus_expr = sp.Add(*[sp.Symbol(f'c{j}') * consensus_terms[j] for j in range(len(consensus_terms))])
            
            # Use secondary optimization to find best constants c_j
            refined_prog = self._refine_consensus_expression(consensus_expr, consensus_terms, X_norm, yi_norm)
            final_equations.append(refined_prog)
            self.confidences.append(0.9) # High confidence for consensus

        return final_equations

    def _refine_consensus_expression(self, consensus_expr, consensus_terms, X, y_true):
        """
        Fit constants for a consensus expression using least squares.
        """
        n_features = X.shape[1]
        feat_vars = [sp.Symbol(f'x{i}') for i in range(n_features)]
        
        # Build feature matrix for regression: Phi_ij = term_j(X_i)
        phi = np.zeros((X.shape[0], len(consensus_terms)))
        for j, term in enumerate(consensus_terms):
            f_lamb = sp.lambdify(feat_vars, term, modules=['numpy'])
            try:
                # Handle terms that might only use a subset of features
                val = f_lamb(*[X[:, k] for k in range(n_features)])
                if np.isscalar(val):
                    phi[:, j] = np.full(X.shape[0], val)
                else:
                    phi[:, j] = val
            except:
                phi[:, j] = 0.0

        # Linear regression to find constants
        from sklearn.linear_model import Ridge
        reg = Ridge(alpha=1e-6, fit_intercept=True)
        reg.fit(phi, y_true)
        
        # Reconstruct final SymPy expression
        final_expr = reg.intercept_
        for j, coeff in enumerate(reg.coef_):
            if abs(coeff) > 1e-6:
                final_expr += coeff * consensus_terms[j]
        
        return OptimizedExpressionWrapper(str(final_expr))


class PhysicsAwareSymbolicDistiller(EnhancedSymbolicDistiller, HamiltonianSymbolicDistiller):
    """
    Physics-aware symbolic distiller that incorporates domain knowledge about physical constants.
    """
    
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12,
                 secondary_optimization=True, physics_constants=None):
        # We call EnhancedSymbolicDistiller's init which calls SymbolicDistiller's init
        # We don't need to call HamiltonianSymbolicDistiller's init as it doesn't have one 
        # (it uses the base class init)
        EnhancedSymbolicDistiller.__init__(self, populations, generations, stopping_criteria, max_features, 
                                          secondary_optimization=secondary_optimization)
        self.physics_constants = physics_constants or {}
        # HamiltonianSymbolicDistiller attributes
        self.enforce_hamiltonian_structure = True
        self.perform_coordinate_alignment = False
        self.estimate_dissipation = True
        
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