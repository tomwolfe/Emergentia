"""
Optimized Symbolic Dynamics that addresses the computational bottleneck by moving
polynomial expansion out of the ODE integration loop.
"""

import numpy as np
import torch
from scipy.integrate import odeint as scipy_odeint
from torchdiffeq import odeint as torch_odeint
import sympy as sp
from symbolic import SymbolicDistiller, gp_to_sympy


class OptimizedSymbolicDynamics:
    """
    Optimized version of SymbolicDynamics that addresses the computational bottleneck
    by pre-computing polynomial expansions and caching transformations.
    
    Now supports both scipy (CPU/LSODA) and torchdiffeq solvers for consistency.
    """
    
    def __init__(self, distiller, equations, feature_masks, is_hamiltonian, n_super_nodes, latent_dim, solver='scipy'):
        self.distiller = distiller
        self.equations = equations
        self.feature_masks = feature_masks
        self.is_hamiltonian = is_hamiltonian
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.solver = solver # 'scipy' or 'torch'

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

    def integrate(self, z0, t, method='dopri5', rtol=1e-4, atol=1e-6):
        """
        Integrate the dynamics using the selected solver.
        """
        if self.solver == 'scipy':
            return scipy_odeint(self, z0, t)
        else:
            # torchdiffeq expected t and z0 to be tensors
            z0_t = torch.tensor(z0, dtype=torch.float32)
            t_t = torch.tensor(t, dtype=torch.float32)
            
            # We need to wrap self into a torch.nn.Module or a callable that handles tensors
            class TorchODEFunc(torch.nn.Module):
                def __init__(self, parent):
                    super().__init__()
                    self.parent = parent
                def forward(self, t, z):
                    # Convert to numpy for the parent call (which handles caching/sympy)
                    z_np = z.detach().cpu().numpy().flatten()
                    dzdt = self.parent(z_np, t.item())
                    return torch.tensor(dzdt, dtype=torch.float32, device=z.device)
            
            func = TorchODEFunc(self)
            with torch.no_grad():
                sol = torch_odeint(func, z0_t, t_t, method=method, rtol=rtol, atol=atol)
            return sol.cpu().numpy()

    def _prepare_sympy_gradients(self):
        """Prepare SymPy gradients for the Hamiltonian"""
        # Convert the gplearn expression to SymPy using the robust converter
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
        return gp_to_sympy(str(gp_program))

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
        Compute Hamiltonian derivatives: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q - γp
        Using symbolic chain rule for efficiency and precision.
        """
        grad = None
        # Use analytical gradients if available
        if hasattr(self, 'lambda_funcs') and self.lambda_funcs is not None and self.sympy_vars is not None:
            try:
                # 1. dH_norm / dX_norm (Analytical from SymPy)
                X_flat = X_norm.flatten()
                args = [X_flat[self.var_to_idx[var.name]] for var in self.sympy_vars]
                dH_norm_dX_norm_sparse = np.array([float(grad_func(*args)) for grad_func in self.lambda_funcs])
                
                # Expand sparse gradient to full feature space size
                # feature_masks[0] is the list of indices of features used by the Hamiltonian
                full_feat_dim = self.transformer.x_poly_mean.shape[0] if hasattr(self.transformer, 'x_poly_mean') else X_norm.shape[1]
                if hasattr(self, 'feature_masks') and self.feature_masks:
                    dH_norm_dX_norm = np.zeros(len(self.feature_masks[0]))
                    # self.var_to_idx maps 'x5' to index 5 IN THE SELECTED FEATURES
                    for i, var in enumerate(self.sympy_vars):
                        idx = self.var_to_idx[var.name]
                        dH_norm_dX_norm[idx] = dH_norm_dX_norm_sparse[i]
                else:
                    dH_norm_dX_norm = dH_norm_dX_norm_sparse

                # 2. Scale by target std: dH / dX_norm = dH_norm / dX_norm * sigma_H
                sigma_H = self.distiller.transformer.target_std[0] if hasattr(self.distiller.transformer.target_std, '__len__') else self.distiller.transformer.target_std
                dH_dX_norm = dH_norm_dX_norm * sigma_H

                # 3. dX_norm / dX = 1 / sigma_X
                # We only need the sigma_X for the features used
                if hasattr(self, 'feature_masks') and self.feature_masks:
                    sigma_X = self.transformer.x_poly_std[self.feature_masks[0]]
                else:
                    sigma_X = self.transformer.x_poly_std
                
                dH_dX = dH_dX_norm / sigma_X

                # 4. dX / dz (Jacobian from transformer)
                dX_dz = self.transformer.transform_jacobian(z) # [n_features, n_latents]
                
                # 5. Final gradient: dH / dz = dH / dX * dX / dz
                grad = dH_dX @ dX_dz
            except Exception as e:
                # Fallback to numerical gradient if analytical fails
                grad = self._numerical_gradient(z)
        else:
            grad = self._numerical_gradient(z)

        # Construct derivatives respecting Hamiltonian structure
        dzdt = np.zeros_like(z)
        d_sub = self.latent_dim // 2
        
        # Check if the equation has dissipation coefficients
        dissipation_coeffs = None
        if hasattr(self.equations[0], 'dissipation_coeffs'):
            dissipation_coeffs = self.equations[0].dissipation_coeffs

        # Map the gradient to Hamiltonian structure
        # z = [q1, p1, q2, p2, ...] based on latent_dim blocks
        for k in range(self.n_super_nodes):
            q_start = k * self.latent_dim
            q_end = q_start + d_sub
            p_start = q_end
            p_end = (k + 1) * self.latent_dim
            
            # dq/dt = ∂H/∂p
            dq_dt = grad[p_start:p_end]
            
            # dp/dt = -∂H/∂q
            dp_dt = -grad[q_start:q_end]
            
            # Add dissipation if available: dp/dt = -∂H/∂q - γp
            if dissipation_coeffs is not None and k < len(dissipation_coeffs):
                gamma = dissipation_coeffs[k]
                p = z[p_start:p_end]
                dp_dt = dp_dt - gamma * p
            
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