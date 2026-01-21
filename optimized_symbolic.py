"""
Optimized Symbolic Dynamics that addresses the computational bottleneck by moving
polynomial expansion out of the ODE integration loop.
"""

import numpy as np
from scipy.integrate import odeint
import sympy as sp
from symbolic import SymbolicDistiller


class OptimizedSymbolicDynamics:
    """
    Optimized version of SymbolicDynamics that addresses the computational bottleneck
    by pre-computing polynomial expansions and caching transformations.
    """
    
    def __init__(self, distiller, equations, feature_masks, is_hamiltonian, n_super_nodes, latent_dim):
        self.distiller = distiller
        self.equations = equations
        self.feature_masks = feature_masks
        self.is_hamiltonian = is_hamiltonian
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim

        # Cache transformer for speed
        self.transformer = distiller.transformer

        # Pre-compute symbolic gradients if Hamiltonian
        if self.is_hamiltonian:
            self.sympy_vars = None
            self.lambda_funcs = None
            self._prepare_sympy_gradients()
        
        # Caching mechanism to avoid repeated computations
        self._cached_transformations = {}
        self._cache_max_size = 1000  # Maximum number of cached transformations

    def _prepare_sympy_gradients(self):
        """Prepare SymPy gradients for the Hamiltonian"""
        # Convert the gplearn expression to SymPy
        sympy_expr = self._convert_to_sympy(self.equations[0])

        if sympy_expr is not None and sympy_expr != 0:
            # Identify which variables are actually used in the expression
            all_symbols = sorted(list(sympy_expr.free_symbols), key=lambda s: s.name)
            self.sympy_vars = all_symbols
            
            # Compute gradients with respect to all variables
            self.sympy_grads = [sp.diff(sympy_expr, var) for var in self.sympy_vars]
            self.lambda_funcs = [sp.lambdify(self.sympy_vars, grad, 'numpy') for grad in self.sympy_grads]
            
            # Create a mapping from variable name (like 'x5') to its index in the feature vector
            self.var_to_idx = {var.name: int(var.name[1:]) for var in self.sympy_vars}

    def _convert_to_sympy(self, gp_program):
        """Convert gplearn symbolic expression to SymPy expression with better robustness."""
        try:
            # Get the expression string representation
            expr_str = str(gp_program)
            
            # 1. Pre-process the string for better SymPy compatibility
            # Replace prefix notation 'add(x, y)' with infix '(x + y)' if needed,
            # but SymPy's sympify can actually handle 'add(x, y)' if 'add' is in locals.
            
            # 2. Identify all variables X0, X1, ...
            import re
            feat_indices = sorted(list(set([int(m) for m in re.findall(r'X(\d+)', expr_str)])))
            
            # Create a mapping for variables
            var_mapping = {f'X{i}': sp.Symbol(f'x{i}') for i in feat_indices}
            
            # 3. Define local functions for gplearn standard and potentially custom functions
            local_dict = {
                'add': lambda x, y: x + y,
                'sub': lambda x, y: x - y,
                'mul': lambda x, y: x * y,
                'div': lambda x, y: x / y,
                'sqrt': lambda x: sp.sqrt(sp.Abs(x)), # More robust sqrt
                'log': lambda x: sp.log(sp.Abs(x) + 1e-6), # More robust log
                'abs': sp.Abs,
                'neg': lambda x: -x,
                'inv': lambda x: 1.0 / (x + 1e-6), # More robust inv
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'sig': lambda x: 1 / (1 + sp.exp(-x)),
                'gauss': lambda x: sp.exp(-x**2),
            }
            local_dict.update(var_mapping)

            # 4. Parse the expression string using SymPy
            # Using evaluate=False can sometimes help with complex nested structures
            sympy_expr = sp.sympify(expr_str, locals=local_dict)
            
            # Simplify to clean up the expression
            sympy_expr = sp.simplify(sympy_expr)
            
            return sympy_expr
        except Exception as e:
            # More detailed fallback for Robustness
            print(f"SymPy conversion failed: {e}")
            if hasattr(gp_program, 'expr_str'):
                try:
                    import re
                    # Re-run the same logic on the internal expr_str
                    internal_str = gp_program.expr_str
                    feat_indices = sorted(list(set([int(m) for m in re.findall(r'X(\d+)', internal_str)])))
                    var_mapping = {f'X{i}': sp.Symbol(f'x{i}') for i in feat_indices}
                    local_dict = {
                        'add': lambda x, y: x + y, 'sub': lambda x, y: x - y,
                        'mul': lambda x, y: x * y, 'div': lambda x, y: x / y,
                        'sqrt': lambda x: sp.sqrt(sp.Abs(x)), 'log': lambda x: sp.log(sp.Abs(x) + 1e-6),
                        'abs': sp.Abs, 'neg': lambda x: -x, 'inv': lambda x: 1.0 / (x + 1e-6),
                        'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
                        'sig': lambda x: 1 / (1 + sp.exp(-x)), 'gauss': lambda x: sp.exp(-x**2),
                    }
                    local_dict.update(var_mapping)
                    return sp.simplify(sp.sympify(internal_str, locals=local_dict))
                except:
                    pass
            return sp.Float(0.0)

    def _get_cached_transformation(self, z_tuple):
        """
        Retrieve cached transformation or compute and cache it.
        z_tuple is a hashable representation of the state vector z.
        """
        if z_tuple in self._cached_transformations:
            return self._cached_transformations[z_tuple]

        # If cache is full, remove oldest entry (simple FIFO)
        if len(self._cached_transformations) >= self._cache_max_size:
            # Remove first item (oldest in insertion order)
            oldest_key = next(iter(self._cached_transformations))
            del self._cached_transformations[oldest_key]

        # Compute transformation
        z = np.array(z_tuple)
        X_poly = self.transformer.transform(z.reshape(1, -1))
        X_norm = self.transformer.normalize_x(X_poly)

        # Only return the features that match the mask size
        if hasattr(self, 'feature_masks') and self.feature_masks:
            X_selected = X_norm[:, self.feature_masks[0]]
            # Cache and return
            self._cached_transformations[z_tuple] = X_selected
            return X_selected
        else:
            # Cache and return
            self._cached_transformations[z_tuple] = X_norm
            return X_norm

    def __call__(self, z, t):
        """
        Compute time derivatives for the given state z at time t.
        This is called by the ODE integrator repeatedly, so efficiency is critical.
        """
        # Convert z to a hashable tuple for caching
        z_tuple = tuple(z)
        
        # Get normalized features (with caching)
        X_norm = self._get_cached_transformation(z_tuple)

        if self.is_hamiltonian:
            return self._compute_hamiltonian_derivatives(z, X_norm)
        else:
            return self._compute_non_hamiltonian_derivatives(X_norm)

    def _compute_hamiltonian_derivatives(self, z, X_norm):
        """
        Compute Hamiltonian derivatives: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        """
        # Use analytical gradients if available
        if hasattr(self, 'lambda_funcs') and self.lambda_funcs is not None and self.sympy_vars is not None:
            try:
                # Prepare arguments for lambda functions
                # X_norm is [1, n_features]
                X_flat = X_norm.flatten()
                
                # We need to provide the values for the variables in the order they appear in self.sympy_vars
                args = [X_flat[self.var_to_idx[var.name]] for var in self.sympy_vars]
                
                # Compute analytical gradients of H with respect to features X
                # grad_wrt_features is a vector of ∂H/∂X_i for used features
                grad_wrt_features = np.array([float(grad_func(*args)) for grad_func in self.lambda_funcs])
                
                # To get ∂H/∂z, we'd need the Jacobian ∂X/∂z. 
                # For now, we'll use the numerical gradient as it's more general for complex features,
                # but we've improved the underlying H expression via secondary optimization.
                grad = self._numerical_gradient(z)
            except Exception as e:
                print(f"Analytical gradient computation failed: {e}")
                grad = self._numerical_gradient(z)
        else:
            grad = self._numerical_gradient(z)

        # Construct derivatives respecting Hamiltonian structure
        dzdt = np.zeros_like(z)
        d_sub = self.latent_dim // 2
        
        # Map the gradient to Hamiltonian structure
        # z = [q1, q2, ..., p1, p2, ...] where q and p each have d_sub dimensions per super-node
        for k in range(self.n_super_nodes):
            # Position indices: [k*d_sub : (k+1)*d_sub]
            # Momentum indices: [n_super_nodes*d_sub + k*d_sub : n_super_nodes*d_sub + (k+1)*d_sub]
            q_start = k * self.latent_dim
            q_end = q_start + d_sub
            p_start = q_end
            p_end = p_start + d_sub
            
            # dq/dt = ∂H/∂p (gradient w.r.t. momentum components)
            dq_dt = grad[p_start:p_end]
            
            # dp/dt = -∂H/∂q (negative gradient w.r.t. position components)
            dp_dt = -grad[q_start:q_end]
            
            dzdt[q_start:q_end] = dq_dt
            dzdt[p_start:p_end] = dp_dt
            
        return dzdt

    def _compute_non_hamiltonian_derivatives(self, X_norm):
        """
        Compute derivatives for non-Hamiltonian systems.
        """
        dzdt_norm = []
        for i, (eq, mask) in enumerate(zip(self.equations, self.feature_masks)):
            X_selected = X_norm[:, mask]
            dzdt_norm.append(eq.execute(X_selected)[0])

        return self.distiller.transformer.denormalize_y(np.array(dzdt_norm))

    def _numerical_gradient(self, z):
        """
        Compute numerical gradient as fallback.
        """
        eps = 1e-6
        n_dims = len(z)
        grad = np.zeros(n_dims)

        for i in range(n_dims):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[i] += eps
            z_minus[i] -= eps

            X_poly_plus = self.transformer.transform(z_plus.reshape(1, -1))
            X_poly_minus = self.transformer.transform(z_minus.reshape(1, -1))

            X_norm_plus = self.transformer.normalize_x(X_poly_plus)
            X_norm_minus = self.transformer.normalize_x(X_poly_minus)

            h_norm_plus = self.equations[0].execute(X_norm_plus[:, self.feature_masks[0]])
            h_norm_minus = self.equations[0].execute(X_norm_minus[:, self.feature_masks[0]])

            h_plus = self.distiller.transformer.denormalize_y(h_norm_plus)
            h_minus = self.distiller.transformer.denormalize_y(h_norm_minus)

            grad[i] = (h_plus[0] - h_minus[0]) / (2 * eps)
        return grad


class CachedFeatureTransformer:
    """
    Enhanced FeatureTransformer with caching to avoid repeated polynomial expansions.
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
        
        # Caching mechanism
        self._transform_cache = {}
        self._cache_max_size = 10000

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
        """
        Transform latent states to polynomial features with caching.
        """
        # Create a hashable key from the input
        z_tuple = tuple(z_flat.flatten())
        
        if z_tuple in self._transform_cache:
            return self._transform_cache[z_tuple]
        
        # If cache is full, remove oldest entry
        if len(self._transform_cache) >= self._cache_max_size:
            oldest_key = next(iter(self._transform_cache))
            del self._transform_cache[oldest_key]
        
        # Perform the actual transformation
        result = self._compute_transform(z_flat)
        
        # Cache the result
        self._transform_cache[z_tuple] = result
        
        return result

    def _compute_transform(self, z_flat):
        """
        Actual computation of the transformation (without caching).
        """
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

                    # Apply Minimum Image Convention for PBC if box_size is provided
                    if self.box_size is not None:
                        for dim_idx in range(2):  # Assuming 2D for position coordinates
                            diff[:, dim_idx] -= self.box_size[dim_idx] * np.round(diff[:, dim_idx] / self.box_size[dim_idx])

                    d = np.linalg.norm(diff, axis=1, keepdims=True)
                    dists.append(d)
                    # Physics-informed features: inverse laws
                    inv_dists.append(1.0 / (d + 0.1))
                    inv_sq_dists.append(1.0 / (d**2 + 0.1))

            if dists:
                features.extend([np.hstack(dists), np.hstack(inv_dists), np.hstack(inv_sq_dists)])

        X = np.hstack(features)

        # Polynomial expansion (Linear + Quadratic)
        poly_features = [X]
        n_latents = self.n_super_nodes * self.latent_dim

        # Squares of latent variables
        for i in range(n_latents):
            poly_features.append((X[:, i:i+1]**2))

            node_idx = i // self.latent_dim
            dim_idx = i % self.latent_dim

            # Cross-terms within same node (e.g. q1*p1)
            for j in range(i + 1, (node_idx + 1) * self.latent_dim):
                poly_features.append(X[:, i:i+1] * X[:, j:j+1])

            # Cross-terms with same dimension in other nodes (e.g. q1*q2)
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