"""
Enhanced Symbolic Distiller that enforces Hamiltonian structure during symbolic regression.
"""

import numpy as np
import sympy as sp
from symbolic import SymbolicDistiller
from balanced_features import BalancedFeatureTransformer as FeatureTransformer

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
        
    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None, model=None, quick=False, sim_type=None, enforce_separable=True):
        """
        Distill symbolic equations, enforcing Hamiltonian structure and estimating dissipation.
        
        Args:
            latent_states: [N, D_total] latent states
            targets: [N, D_targets] targets (either scalar H or derivatives dz/dt)
            n_super_nodes: Number of super-nodes
            latent_dim: Latent dimension per super-node
            box_size: PBC box size
            model: Optional DiscoveryEngineModel to extract parameters from
            quick: Whether to perform quick distillation (skip deep search)
            sim_type: Type of simulation ('spring', 'lj', etc.)
            enforce_separable: Whether to enforce H = V(q) + T(p)
        """
        if not self.enforce_hamiltonian_structure or latent_dim % 2 != 0:
            return super().distill(latent_states, targets, n_super_nodes, latent_dim, box_size, quick=quick, sim_type=sim_type)

        # Check if targets are scalar H (1 column) or derivatives (many columns)
        is_derivative_targets = targets.shape[1] > 1

        # NEW: Perform Coordinate Alignment to rotate latent z to maximize correlation with physical CoM
        if self.perform_coordinate_alignment:
            aligned_latent_states = self._perform_coordinate_alignment(latent_states, n_super_nodes, latent_dim)
        else:
            aligned_latent_states = latent_states

        print(f"  -> Initializing FeatureTransformer (Enforce Separable: {enforce_separable})...")
        # For non-separable, we include raw latents to allow H(q,p) coupling
        include_raw = not enforce_separable
        self.transformer = FeatureTransformer(n_super_nodes, latent_dim, box_size=box_size, include_raw_latents=include_raw, sim_type=sim_type)

        # If we have a model and it has a hamiltonian method, we should try to get the scalar H as target
        # for better GP discovery of the energy function topology.
        h_targets = targets
        if is_derivative_targets and model is not None and hasattr(model.ode_func, 'hamiltonian'):
            try:
                print(f"  -> Extracting Hamiltonian values from neural model...")
                import torch
                device = next(model.parameters()).device
                z_torch = torch.from_numpy(aligned_latent_states).float().to(device)
                with torch.no_grad():
                    h_val = model.ode_func.hamiltonian(z_torch).cpu().numpy()

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

        # Target for GP is either V(q) or full H(q, p)
        if enforce_separable:
            # STRUCTURAL FIX: Enforce H = sum(p^2/2) + V(q)
            # We calculate the kinetic energy term and subtract it from the target H to get V(q)
            d_sub = latent_dim // 2
            z_reshaped = aligned_latent_states.reshape(-1, n_super_nodes, latent_dim)
            p_vals = z_reshaped[:, :, d_sub:]
            ke_term = 0.5 * np.sum(p_vals**2, axis=(1, 2))
            gp_targets = h_targets.flatten() - ke_term
            target_name = "Potential V(q)"
        else:
            # Learn full H(q, p)
            gp_targets = h_targets.flatten()
            target_name = "Hamiltonian H(q, p)"

        print(f"  -> Fitting FeatureTransformer to {aligned_latent_states.shape[0]} samples...")
        self.transformer.fit(aligned_latent_states, gp_targets)
        
        if enforce_separable:
            # STRICT COORDINATE SEPARATION: Prune p-related features from the potential search
            # Potential V(q) MUST NOT see any momentum-related features or raw latents
            d_sub = latent_dim // 2
            q_mask = np.ones(len(self.transformer.feature_names), dtype=bool)
            for idx, name in enumerate(self.transformer.feature_names):
                if "p" in name: 
                    q_mask[idx] = False
                    continue
                import re
                z_indices = re.findall(r'z(\d+)', name)
                for z_idx_str in z_indices:
                    z_idx = int(z_idx_str)
                    if (z_idx % latent_dim) >= d_sub:
                        q_mask[idx] = False
                        break
        else:
            # For non-separable, we allow all features
            q_mask = np.ones(len(self.transformer.feature_names), dtype=bool)
        
        X_all = self.transformer.transform(aligned_latent_states)
        
        # Manually normalize all features before slicing
        X_norm_full = (X_all - self.transformer.x_poly_mean) / self.transformer.x_poly_std
        X_norm = X_norm_full[:, q_mask]
        
        # Temporarily update transformer feature names to reflect selected features for the GP search
        original_names = self.transformer.feature_names
        self.transformer.feature_names = [n for i, n in enumerate(original_names) if q_mask[i]]
        
        Y_norm = self.transformer.normalize_y(gp_targets)

        # Distill the function
        y_target = Y_norm[:, 0] if Y_norm.ndim > 1 else Y_norm
        
        # Ensure reasonable search depth but respect user settings
        print(f"  -> Starting symbolic regression for {target_name} (Pop: {self.populations}, Gen: {self.generations})...")
        h_prog, h_mask_relative, h_conf = self._distill_single_target(0, X_norm, y_target, 1, latent_dim, skip_deep_search=quick, sim_type=sim_type)
        
        # Map relative mask back to absolute indices
        q_indices = np.where(q_mask)[0]
        h_mask = q_indices[h_mask_relative]
        
        if h_prog is None:
            return super().distill(latent_states, targets, n_super_nodes, latent_dim, box_size)

        # Validate stability
        try:
            h_pred = h_prog.execute(X_norm[:, h_mask_relative])
            if not np.all(np.isfinite(h_pred)):
                print(f"Warning: Discovered {target_name} produces NaNs/Infs. Reducing confidence.")
                h_conf *= 0.1
        except Exception as e:
            print(f"Warning: Failed to validate {target_name} stability: {e}")
            h_conf = 0.0
            
        # Estimate dissipation coefficients γ
        dissipation_coeffs = np.zeros(n_super_nodes)
        if self.estimate_dissipation:
            if model is not None and hasattr(model.ode_func, 'gamma'):
                import torch
                dissipation_coeffs = torch.exp(model.ode_func.gamma).detach().cpu().numpy().flatten()

        # Wrap it in a class that can compute derivatives
        try:
            from symbolic_utils import gp_to_sympy
            import sympy as sp
            
            expr_str = str(h_prog)
            n_vars = X_norm.shape[1] # Number of selected features
            sympy_expr = gp_to_sympy(expr_str)
            
            sympy_vars = [sp.Symbol(f'X{i}') for i in range(n_vars)]
            grad_funcs = []
            for var in sympy_vars:
                grad_expr = sp.diff(sympy_expr, var)
                grad_funcs.append(sp.lambdify(sympy_vars, grad_expr, 'numpy'))
                
            ham_eq = HamiltonianEquation(
                h_prog, h_mask, n_super_nodes, latent_dim, 
                dissipation_coeffs=dissipation_coeffs,
                sympy_expr=sympy_expr,
                grad_funcs=grad_funcs,
                enforce_separable=enforce_separable
            )
            print(f"  -> Successfully enabled analytical gradients for discovered {target_name}.")
        except Exception as e:
            print(f"  -> Fallback to numerical gradients: {e}")
            ham_eq = HamiltonianEquation(h_prog, h_mask, n_super_nodes, latent_dim, dissipation_coeffs, enforce_separable=enforce_separable)

        self.feature_masks = [h_mask]
        self.confidences = [h_conf]
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
                 dissipation_coeffs=None, sympy_expr=None, grad_funcs=None, enforce_separable=False):
        self.h_prog = h_prog
        self.feature_mask = feature_mask
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.dissipation_coeffs = dissipation_coeffs if dissipation_coeffs is not None else np.zeros(n_super_nodes)
        self.sympy_expr = sympy_expr
        self.grad_funcs = grad_funcs # List of lambdified functions for each input feature
        self.length_ = getattr(h_prog, 'length_', 1)
        self.enforce_separable = enforce_separable

    def execute(self, X):
        # If X already has the correct number of features for h_prog, use it directly.
        # Otherwise, slice it using feature_mask (assuming X is the full feature vector).
        if hasattr(self.h_prog, 'feature_indices'):
            # Some programs handle their own indexing
            return self.h_prog.execute(X)
        
        if X.shape[1] == len(self.feature_mask):
            return self.h_prog.execute(X)
        return self.h_prog.execute(X[:, self.feature_mask])

    def compute_derivatives(self, z, transformer):
        """
        Compute dq/dt = ∂H/∂p and dp/dt = -∂H/∂q - γp
        Uses analytical gradients if available, otherwise falls back to finite differences.
        """
        n_total = len(z)
        d_sub = self.latent_dim // 2
        dzdt = np.zeros(n_total)
        
        # dH/dz = dV/dz + dT/dz
        # If enforce_separable=True, H = V(q) + sum(p^2/2)
        # -> dT/dq = 0, dT/dp = p
        
        # If we have analytical gradients, use them for much higher precision
        if self.grad_funcs is not None:
            # We must use the transformer to get the selected features
            X_p = transformer.transform(z.reshape(1, -1))
            X_n = transformer.normalize_x(X_p) # Already sliced to mask
            
            # Compute full Jacobian of the transformation dX/dz for ALL features
            # and then slice it to the selected features
            dX_dz = transformer.transform_jacobian(z) # [n_selected, n_latents]
            
            # dH/dX: gradients of H (or V) with respect to its input features (masked ones)
            dH_dX = np.zeros(X_n.shape[1])
            for i, grad_f in enumerate(self.grad_funcs):
                if grad_f is not None:
                    try:
                        # Pass all masked features to the lambdified gradient function
                        val = grad_f(*[X_n[0, j] for j in range(X_n.shape[1])])
                        dH_dX[i] = val
                    except:
                        pass
            
            # dV/dz = dV/dX @ dX_dz
            # Account for normalization: dV/dz = (dV/dX_norm / X_std_selected) * target_std @ dX_dz
            x_std_selected = transformer.x_poly_std[self.feature_mask]
            dV_dX_norm = dH_dX / x_std_selected
            
            # CRITICAL FIX: Multiply by target_std to get back to original scale
            dV_dz = (dV_dX_norm * transformer.target_std) @ dX_dz
            
            for k in range(self.n_super_nodes):
                q_start = k * self.latent_dim
                q_end = q_start + d_sub
                p_start = q_end
                p_end = p_start + d_sub
                
                # dq/dt = ∂H/∂p = ∂V/∂p + ∂T/∂p
                if self.enforce_separable:
                    # ∂V/∂p = 0 by construction of features, ∂T/∂p = p
                    dzdt[q_start:q_end] = z[p_start:p_end]
                else:
                    dzdt[q_start:q_end] = dV_dz[p_start:p_end]
                
                # dp/dt = -∂H/∂q - γp = -(∂V/∂q + ∂T/∂q) - γp
                # ∂T/∂q = 0
                gamma = self.dissipation_coeffs[k]
                dzdt[p_start:p_end] = -dV_dz[q_start:q_end] - gamma * z[p_start:p_end]
                
            return dzdt

        # Fallback to numerical gradients
        eps = 1e-4
        def get_V(state):
            X_p = transformer.transform(state.reshape(1, -1))
            X_n = transformer.normalize_x(X_p) # Already sliced to mask
            return self.h_prog.execute(X_n)[0]

        for k in range(self.n_super_nodes):
            q_start = k * self.latent_dim
            q_end = q_start + d_sub
            p_start = q_end
            p_end = p_start + d_sub
            
            # dq/dt = ∂H/∂p
            if self.enforce_separable:
                dzdt[q_start:q_end] = z[p_start:p_end]
            else:
                for i in range(d_sub):
                    idx = p_start + i
                    z_plus = z.copy()
                    z_minus = z.copy()
                    z_plus[idx] += eps
                    z_minus[idx] -= eps
                    dzdt[q_start + i] = (get_V(z_plus) - get_V(z_minus)) / (2 * eps)
                
            # dp/dt = -∂H/∂q - γp
            gamma = self.dissipation_coeffs[k]
            for i in range(d_sub):
                idx = q_start + i
                z_plus = z.copy()
                z_minus = z.copy()
                z_plus[idx] += eps
                z_minus[idx] -= eps
                
                # Conservative part: -∂V/∂q
                dp_dt_cons = -(get_V(z_plus) - get_V(z_minus)) / (2 * eps)
                
                # Dissipative part: -γp
                dp_dt_diss = -gamma * z[p_start + i]
                
                dzdt[p_start + i] = dp_dt_cons + dp_dt_diss
            
        return dzdt

    def __str__(self):
        gamma_str = f", γ={self.dissipation_coeffs}" if np.any(self.dissipation_coeffs > 0) else ""
        prefix = "SeparableHamiltonian" if self.enforce_separable else "HamiltonianH"
        return f"{prefix}(V={self.h_prog}{gamma_str}, T=p^2/2)"
