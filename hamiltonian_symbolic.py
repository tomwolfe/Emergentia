"""
Enhanced Symbolic Distiller that enforces Hamiltonian structure during symbolic regression.
"""

import numpy as np
import sympy as sp
from symbolic import SymbolicDistiller, FeatureTransformer

class HamiltonianSymbolicDistiller(SymbolicDistiller):
    """
    Enhanced SymbolicDistiller that enforces Hamiltonian structure and accounts for dissipation.
    dq/dt = ∂H/∂p
    dp/dt = -∂H/∂q - γp
    """
    
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12,
                 enforce_hamiltonian_structure=True, estimate_dissipation=True,
                 perform_coordinate_alignment=True):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.enforce_hamiltonian_structure = enforce_hamiltonian_structure
        self.estimate_dissipation = estimate_dissipation
        self.perform_coordinate_alignment = perform_coordinate_alignment

    def _perform_coordinate_alignment(self, latent_states, n_super_nodes, latent_dim):
        """
        Perform coordinate alignment to rotate the latent z to maximize correlation with physical CoM.
        This ensures that X0, X1 in the discovered equations correspond to physical positions.

        Uses memory-efficient SVD to avoid computing large covariance matrices.
        """
        import numpy as np

        # Reshape latent states to [N, K, D] where N is number of samples, K is super nodes, D is latent dim
        n_samples = latent_states.shape[0]
        z_reshaped = latent_states.reshape(n_samples, n_super_nodes, latent_dim)

        # Flatten to [N*K, D] for PCA-like alignment
        z_flat = z_reshaped.reshape(-1, latent_dim)

        # Memory optimization: if data is huge, sample it for PCA computation
        if z_flat.shape[0] > 20000:
            indices = np.random.choice(z_flat.shape[0], 20000, replace=False)
            z_for_pca = z_flat[indices]
        else:
            z_for_pca = z_flat

        # Center the data
        z_mean = np.mean(z_for_pca, axis=0, keepdims=True)
        z_centered_pca = z_for_pca - z_mean

        # Use SVD instead of computing the full covariance matrix
        # For matrix A, A = U * S * V.T, where V.T contains the principal components
        # This avoids computing A.T @ A which would be huge
        try:
            # Use economy SVD which is more memory efficient
            # We only need Vt for the rotation, so we can use compute_uv=True but only take Vt
            _, _, Vt = np.linalg.svd(z_centered_pca, full_matrices=False)

            # Rotate the FULL dataset using the discovered principal components
            z_centered_full = z_flat - np.mean(z_flat, axis=0, keepdims=True)
            aligned_z_flat = z_centered_full @ Vt.T

            # Reshape back to original shape
            aligned_latent_states = aligned_z_flat.reshape(n_samples, n_super_nodes, latent_dim)

            # Reshape back to original format [N, K*D]
            aligned_latent_states = aligned_latent_states.reshape(n_samples, -1)

            print(f"  -> Performed coordinate alignment using principal component analysis")

            return aligned_latent_states
        except np.linalg.LinAlgError:
            # If SVD fails due to memory issues, fall back to a sampling approach
            print("  -> SVD failed, using sampled PCA approach...")
            return self._perform_coordinate_alignment_sampled(z_centered_pca, n_samples, n_super_nodes, latent_dim)

    def _perform_coordinate_alignment_sampled(self, z_centered, n_samples, n_super_nodes, latent_dim):
        """
        Fallback method that uses a sample of the data for PCA when full SVD fails.
        """
        import numpy as np

        # Sample a subset of the data for PCA
        sample_ratio = min(1.0, 10000 / z_centered.shape[0])  # Use max 10k samples
        if sample_ratio < 1.0:
            n_samples_to_use = int(z_centered.shape[0] * sample_ratio)
            indices = np.random.choice(z_centered.shape[0], n_samples_to_use, replace=False)
            z_sampled = z_centered[indices]
        else:
            z_sampled = z_centered

        # Perform SVD on the sampled data
        U, S, Vt = np.linalg.svd(z_sampled, full_matrices=False)

        # Use the rotation matrix from the sampled data to transform the full dataset
        aligned_z_flat = z_centered @ Vt.T

        # Reshape back to original shape
        n_total_samples = n_samples
        aligned_latent_states = aligned_z_flat.reshape(n_total_samples, n_super_nodes, latent_dim)

        # Reshape back to original format [N, K*D]
        aligned_latent_states = aligned_latent_states.reshape(n_total_samples, -1)

        return aligned_latent_states
        
    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None, model=None):
        """
        Distill symbolic equations, enforcing Hamiltonian structure and estimating dissipation.
        
        Args:
            latent_states: [N, D_total] latent states
            targets: [N, D_targets] targets (either scalar H or derivatives dz/dt)
            n_super_nodes: Number of super-nodes
            latent_dim: Latent dimension per super-node
            box_size: PBC box size
            model: Optional DiscoveryEngineModel to extract parameters from
        """
        if not self.enforce_hamiltonian_structure or latent_dim % 2 != 0:
            return super().distill(latent_states, targets, n_super_nodes, latent_dim, box_size)

        # Check if targets are scalar H (1 column) or derivatives (many columns)
        is_derivative_targets = targets.shape[1] > 1

        # NEW: Perform Coordinate Alignment to rotate latent z to maximize correlation with physical CoM
        if self.perform_coordinate_alignment:
            aligned_latent_states = self._perform_coordinate_alignment(latent_states, n_super_nodes, latent_dim)
        else:
            aligned_latent_states = latent_states

        print(f"  -> Initializing FeatureTransformer...")
        self.transformer = FeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)

        # If we have a model and it has H_net, we should try to get the scalar H as target
        # for better GP discovery of the energy function topology.
        h_targets = targets
        if is_derivative_targets and model is not None and hasattr(model.ode_func, 'H_net'):
            try:
                print(f"  -> Extracting Hamiltonian values from neural model...")
                import torch
                device = next(model.parameters()).device
                z_torch = torch.from_numpy(aligned_latent_states).float().to(device)
                with torch.no_grad():
                    h_val = model.ode_func.H_net(z_torch).cpu().numpy()

                # Ensure h_val has the same shape as latent_states first dimension
                if h_val.ndim > 1 and h_val.shape[1] == 1:
                    h_val = h_val.flatten()
                elif h_val.ndim > 1:
                    # If H_net returns multiple values per sample, take the first one
                    h_val = h_val[:, 0] if h_val.shape[1] > 0 else h_val.flatten()

                # Ensure the shape matches exactly with latent_states first dimension
                if h_val.shape[0] != aligned_latent_states.shape[0]:
                    print(f"    -> Shape mismatch: h_val {h_val.shape[0]}, expected {aligned_latent_states.shape[0]}")
                    h_targets = targets
                else:
                    h_targets = h_val
                    is_derivative_targets = False
            except Exception as e:
                print(f"    -> Failed to extract H from model: {e}")
                h_targets = targets

        print(f"  -> Fitting FeatureTransformer to {aligned_latent_states.shape[0]} samples...")
        self.transformer.fit(aligned_latent_states, h_targets)
        X_norm = self.transformer.normalize_x(self.transformer.transform(aligned_latent_states))
        Y_norm = self.transformer.normalize_y(h_targets)

        # Distill the Hamiltonian function H
        y_target = Y_norm[:, 0] if Y_norm.ndim > 1 else Y_norm
        
        # Ensure minimum search depth for Hamiltonian but respect lower values if quick mode is used
        original_populations = self.populations
        original_generations = self.generations
        self.populations = max(self.populations, 500)
        self.generations = max(self.generations, 10)
        
        print(f"  -> Starting symbolic regression for Hamiltonian (Pop: {self.populations}, Gen: {self.generations})...")
        h_prog, h_mask, h_conf = self._distill_single_target(0, X_norm, y_target, 1, latent_dim)
        
        # Restore original values
        self.populations = original_populations
        self.generations = original_generations
        
        if h_prog is None:
            return super().distill(latent_states, targets, n_super_nodes, latent_dim, box_size)

        # Validate Hamiltonian stability before accepting
        # Check for NaNs or Infs on the training buffer
        try:
            h_pred = h_prog.execute(X_norm[:, h_mask])
            if not np.all(np.isfinite(h_pred)):
                print("Warning: Discovered Hamiltonian produces NaNs/Infs. Reducing confidence.")
                h_conf *= 0.1
        except Exception as e:
            print(f"Warning: Failed to validate Hamiltonian stability: {e}")
            h_conf = 0.0
            
        # Estimate dissipation coefficients γ
        dissipation_coeffs = np.zeros(n_super_nodes)
        if self.estimate_dissipation:
            if model is not None and hasattr(model.ode_func, 'gamma'):
                # Extract directly from model
                import torch
                dissipation_coeffs = torch.exp(model.ode_func.gamma).detach().cpu().numpy().flatten()
            elif is_derivative_targets or targets.shape[1] > 1:
                # Estimate from residual of dp/dt = -dH/dq - γp
                # This requires computing numerical gradients of the discovered H
                dissipation_coeffs = self._estimate_gamma_from_residuals(
                    h_prog, h_mask, latent_states, targets, n_super_nodes, latent_dim
                )

        # Wrap it in a class that can compute derivatives
        # Try to compute analytical gradients for the Hamiltonian
        try:
            from symbolic import gp_to_sympy
            import sympy as sp
            
            # Convert discovered program to SymPy
            expr_str = str(h_prog)
            n_vars = X_norm.shape[1]  # Number of selected features
            sympy_expr = gp_to_sympy(expr_str)
            
            # Compute analytical gradients with respect to the input features X0, X1, ...
            # These X variables correspond to the masked features
            sympy_vars = [sp.Symbol(f'X{i}') for i in range(n_vars)]
            grad_funcs = []
            for var in sympy_vars:
                # sp.diff computes the analytical derivative
                grad_expr = sp.diff(sympy_expr, var)
                # lambdify for fast numerical evaluation
                grad_funcs.append(sp.lambdify(sympy_vars, grad_expr, 'numpy'))
                
            ham_eq = HamiltonianEquation(
                h_prog, h_mask, n_super_nodes, latent_dim, 
                dissipation_coeffs=dissipation_coeffs,
                sympy_expr=sympy_expr,
                grad_funcs=grad_funcs
            )
            print(f"  -> Successfully enabled analytical gradients for discovered Hamiltonian.")
        except Exception as e:
            print(f"  -> Fallback to numerical gradients: {e}")
            ham_eq = HamiltonianEquation(h_prog, h_mask, n_super_nodes, latent_dim, dissipation_coeffs)

        self.feature_masks = [h_mask]
        self.confidences = [h_conf]
        
        # Store the feature mask in the transformer for later use by TorchFeatureTransformer
        self.transformer.selected_feature_indices = h_mask
        
        return [ham_eq]

    def _estimate_gamma_from_residuals(self, h_prog, h_mask, states, derivs, n_super_nodes, latent_dim):
        """Estimate dissipation coefficients γ from residuals of the momentum equations."""
        gammas = np.zeros(n_super_nodes)
        d_sub = latent_dim // 2
        
        # We need numerical gradients of H for this estimation
        eps = 1e-4
        n_points = min(500, len(states))
        indices = np.random.choice(len(states), n_points, replace=False)
        
        for k in range(n_super_nodes):
            # Extract momentum p_k and its derivative dp_k/dt
            p_start = k * latent_dim + d_sub
            p_end = (k + 1) * latent_dim
            p = states[indices, p_start:p_end]
            dp_dt = derivs[indices, p_start:p_end]
            
            # Compute -dH/dq_k numerically
            q_start = k * latent_dim
            q_end = q_start + d_sub
            
            minus_dh_dq = np.zeros_like(p)
            for i in range(d_sub):
                idx = q_start + i
                s_plus = states[indices].copy()
                s_minus = states[indices].copy()
                s_plus[:, idx] += eps
                s_minus[:, idx] -= eps
                
                h_plus = h_prog.execute(self.transformer.normalize_x(self.transformer.transform(s_plus))[:, h_mask])
                h_minus = h_prog.execute(self.transformer.normalize_x(self.transformer.transform(s_minus))[:, h_mask])
                
                # dp/dt = -dH/dq -> minus_dh_dq = - (h_plus - h_minus) / (2*eps)
                minus_dh_dq[:, i] = -(h_plus - h_minus) / (2 * eps)
            
            # Residual: R = dp_dt - (-dH/dq) = -γp
            residual = dp_dt - minus_dh_dq
            
            # Fit γ via linear regression: residual = -γ * p
            # Flat arrays for regression
            res_flat = residual.flatten()
            p_flat = p.flatten()
            
            if np.any(np.abs(p_flat) > 1e-6):
                # γ = - residual / p (in least squares sense)
                gamma = - np.sum(res_flat * p_flat) / (np.sum(p_flat**2) + 1e-9)
                gammas[k] = max(0.0, gamma) # Dissipation should be non-negative
                
        return gammas

class HamiltonianEquation:
    """
    Special equation class that computes derivatives from a scalar Hamiltonian
    and incorporates dissipative terms.
    Supports both numerical and analytical gradients.
    """
    def __init__(self, h_prog, feature_mask, n_super_nodes, latent_dim, 
                 dissipation_coeffs=None, sympy_expr=None, grad_funcs=None):
        self.h_prog = h_prog
        self.feature_mask = feature_mask
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.dissipation_coeffs = dissipation_coeffs if dissipation_coeffs is not None else np.zeros(n_super_nodes)
        self.sympy_expr = sympy_expr
        self.grad_funcs = grad_funcs # List of lambdified functions for each input feature
        self.length_ = getattr(h_prog, 'length_', 1)

    def execute(self, X):
        return self.h_prog.execute(X)

    def compute_derivatives(self, z, transformer):
        """
        Compute dq/dt = ∂H/∂p and dp/dt = -∂H/∂q - γp
        Uses analytical gradients if available, otherwise falls back to finite differences.
        """
        n_total = len(z)
        d_sub = self.latent_dim // 2
        dzdt = np.zeros(n_total)
        
        # If we have analytical gradients, use them for much higher precision
        if self.grad_funcs is not None:
            # We must use the transformer to get the selected features
            X_p = transformer.transform(z.reshape(1, -1))
            X_n = transformer.normalize_x(X_p) # Already sliced to mask
            
            # Compute full Jacobian of the transformation dX/dz for ALL features
            # and then slice it to the selected features
            dX_dz = transformer.transform_jacobian(z) # [n_selected, n_latents]
            
            # dH/dX: gradients of H with respect to its input features (masked ones)
            dH_dX = np.zeros(X_n.shape[1])
            for i, grad_f in enumerate(self.grad_funcs):
                if grad_f is not None:
                    try:
                        # Pass all masked features to the lambdified gradient function
                        val = grad_f(*[X_n[0, j] for j in range(X_n.shape[1])])
                        dH_dX[i] = val
                    except:
                        pass
            
            # dH/dz = dH/dX @ dX_dz
            # Account for normalization: dH/dz = (dH/dX_norm / X_std_selected) @ dX_dz
            # transformer.x_poly_std is the FULL array now, so we must slice it
            x_std_selected = transformer.x_poly_std[self.feature_mask]
            dH_dX_norm = dH_dX / x_std_selected
            dH_dz = dH_dX_norm @ dX_dz
            
            for k in range(self.n_super_nodes):
                q_start = k * self.latent_dim
                q_end = q_start + d_sub
                p_start = q_end
                p_end = p_start + d_sub
                
                # dq/dt = ∂H/∂p
                dzdt[q_start:q_end] = dH_dz[p_start:p_end]
                
                # dp/dt = -∂H/∂q - γp
                gamma = self.dissipation_coeffs[k]
                dzdt[p_start:p_end] = -dH_dz[q_start:q_end] - gamma * z[p_start:p_end]
                
            return dzdt

        # Fallback to numerical gradients
        eps = 1e-4
        def get_H(state):
            X_p = transformer.transform(state.reshape(1, -1))
            X_n = transformer.normalize_x(X_p)
            return self.h_prog.execute(X_n[:, self.feature_mask])[0]

        for k in range(self.n_super_nodes):
            q_start = k * self.latent_dim
            q_end = q_start + d_sub
            p_start = q_end
            p_end = p_start + d_sub
            
            # dq/dt = ∂H/∂p
            for i in range(d_sub):
                idx = p_start + i
                z_plus = z.copy()
                z_minus = z.copy()
                z_plus[idx] += eps
                z_minus[idx] -= eps
                dzdt[q_start + i] = (get_H(z_plus) - get_H(z_minus)) / (2 * eps)
                
            # dp/dt = -∂H/∂q - γp
            gamma = self.dissipation_coeffs[k]
            for i in range(d_sub):
                idx = q_start + i
                z_plus = z.copy()
                z_minus = z.copy()
                z_plus[idx] += eps
                z_minus[idx] -= eps
                
                # Conservative part: -dH/dq
                dp_dt_cons = -(get_H(z_plus) - get_H(z_minus)) / (2 * eps)
                
                # Dissipative part: -γp
                dp_dt_diss = -gamma * z[p_start + i]
                
                dzdt[p_start + i] = dp_dt_cons + dp_dt_diss
            
        return dzdt

    def __str__(self):
        gamma_str = f", γ={self.dissipation_coeffs}" if np.any(self.dissipation_coeffs > 0) else ""
        return f"HamiltonianH({self.h_prog}{gamma_str})"
