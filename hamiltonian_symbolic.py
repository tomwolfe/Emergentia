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
                 enforce_hamiltonian_structure=True, estimate_dissipation=True):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.enforce_hamiltonian_structure = enforce_hamiltonian_structure
        self.estimate_dissipation = estimate_dissipation
        
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
        
        self.transformer = FeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        
        # If we have a model and it has H_net, we should try to get the scalar H as target
        # for better GP discovery of the energy function topology.
        h_targets = targets
        if is_derivative_targets and model is not None and hasattr(model.ode_func, 'H_net'):
            try:
                import torch
                device = next(model.parameters()).device
                z_torch = torch.from_numpy(latent_states).float().to(device)
                with torch.no_grad():
                    h_val = model.ode_func.H_net(z_torch).cpu().numpy()
                h_targets = h_val
                is_derivative_targets = False
            except Exception as e:
                print(f"Failed to extract H from model: {e}")

        self.transformer.fit(latent_states, h_targets)
        X_norm = self.transformer.normalize_x(self.transformer.transform(latent_states))
        Y_norm = self.transformer.normalize_y(h_targets)

        # Distill the Hamiltonian function H
        # If we still have derivative targets here, it's a fallback (not ideal for H discovery)
        # Ensure Y_norm is 1D for _distill_single_target
        y_target = Y_norm[:, 0] if Y_norm.ndim > 1 else Y_norm
        h_prog, h_mask, h_conf = self._distill_single_target(0, X_norm, y_target, 1, latent_dim)
        
        if h_prog is None:
            return super().distill(latent_states, targets, n_super_nodes, latent_dim, box_size)
            
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
            n_vars = X_norm.shape[1]
            sympy_expr = gp_to_sympy(expr_str, n_features=n_vars)
            
            # Compute analytical gradients
            sympy_vars = [sp.Symbol(f'x{i}') for i in range(n_vars)]
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
            X_p = transformer.transform(z.reshape(1, -1))
            X_n = transformer.normalize_x(X_p)
            
            # Compute full Jacobian of the transformation dX/dz
            # X = f(z) -> dH/dz = dH/dX * dX/dz
            dX_dz = transformer.transform_jacobian(z) # [n_features, n_latents]
            
            # dH/dX: gradients of H with respect to its input features
            # We only care about features in the mask
            dH_dX = np.zeros(X_n.shape[1])
            # The grad_funcs are expected to be ordered the same as features
            for i, grad_f in enumerate(self.grad_funcs):
                if grad_f is not None:
                    # Execute lambdified gradient function
                    # Some grad_funcs might need all features as input
                    try:
                        val = grad_f(*[X_n[0, j] for j in range(X_n.shape[1])])
                        dH_dX[i] = val
                    except:
                        pass
            
            # dH/dz = dH/dX @ dX_dz
            # Account for normalization: dH/dz = (dH/dX_norm / X_std) @ dX_dz
            dH_dX_norm = dH_dX / transformer.x_poly_std
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
