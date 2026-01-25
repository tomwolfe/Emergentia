import torch
import torch.nn as nn
import sympy as sp
from enhanced_symbolic import TorchFeatureTransformer, SymPyToTorch
from symbolic_utils import gp_to_sympy

class SymbolicProxy(nn.Module):
    """
    A differentiable proxy for the discovered symbolic laws.
    Allows end-to-end gradient flow from symbolic equations back to the GNN.
    """
    def __init__(self, n_super_nodes, latent_dim, equations, transformer, hamiltonian=False):
        super().__init__()
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.is_hamiltonian = hamiltonian
        self.dissipation_coeffs = None

        # 1. Initialize differentiable feature transformer
        self.torch_transformer = TorchFeatureTransformer(transformer)

        # NEW: Learnable output gain to fix units mismatch automatically
        self.output_gain = nn.Parameter(torch.ones(1))

        # 2. Initialize differentiable symbolic modules
        self.sym_modules = nn.ModuleList()

        # Calculate n_inputs as the number of features AFTER feature selection
        try:
            dummy_input = torch.zeros(1, n_super_nodes * latent_dim, 
                                     dtype=self.torch_transformer.x_poly_mean.dtype, 
                                     device=self.torch_transformer.x_poly_mean.device)
            with torch.no_grad():
                dummy_output = self.torch_transformer(dummy_input)
                n_inputs = dummy_output.size(1)
        except Exception:
            if hasattr(self.torch_transformer, 'feature_mask') and self.torch_transformer.feature_mask.numel() > 0:
                n_inputs = self.torch_transformer.feature_mask.size(0)
            else:
                n_inputs = self.torch_transformer.x_poly_mean.size(0)

        for eq in equations:
            if eq is not None:
                if hasattr(eq, 'sympy_expr'):
                    sympy_expr = eq.sympy_expr
                else:
                    sympy_expr = gp_to_sympy(str(eq))

                if hasattr(eq, 'compute_derivatives') or "Hamiltonian" in str(eq):
                    self.is_hamiltonian = True
                    if hasattr(eq, 'dissipation_coeffs'):
                        self.register_buffer('dissipation', torch.from_numpy(eq.dissipation_coeffs).float())
                    if hasattr(eq, 'masses'):
                        self.register_buffer('masses', torch.from_numpy(eq.masses).float())
                    self.sym_modules.append(SymPyToTorch(sympy_expr, n_inputs))
                else:
                    self.sym_modules.append(SymPyToTorch(sympy_expr, n_inputs))
            else:
                self.sym_modules.append(None)

    def forward(self, t, z_flat=None):
        if z_flat is None:
            z_flat = t
            
        # Robustly get current device
        device = next(self.parameters()).device if list(self.parameters()) else \
                 next(self.buffers()).device if list(self.buffers()) else z_flat.device
        
        z_flat = z_flat.to(device).to(torch.float32)

        if self.is_hamiltonian:
            with torch.enable_grad():
                if not z_flat.requires_grad:
                    z_flat = z_flat.detach().requires_grad_(True)
                
                X_norm = self.torch_transformer(z_flat)
                H_norm = self.sym_modules[0](X_norm)
                V = self.torch_transformer.denormalize_y(H_norm)
                
                # Safety clamp for potential
                V = torch.clamp(V, -1e3, 1e3)
                
                # SEPARABLE HAMILTONIAN FIX:
                # If the symbolic part only represents the potential V(q),
                # we must add the kinetic term T(p) = sum(p^2 / 2m)
                # We assume m=1.0 for now unless masses are stored.
                d_sub = self.latent_dim // 2
                batch_size = z_flat.size(0)
                z_reshaped = z_flat.view(batch_size, self.n_super_nodes, self.latent_dim)
                p_vals = z_reshaped[:, :, d_sub:]
                
                # T = sum(p^2 / 2m)
                if hasattr(self, 'masses'):
                    m = self.masses.view(1, self.n_super_nodes, 1)
                    T = 0.5 * torch.sum(p_vals**2 / m, dim=(1, 2)).view(-1, 1)
                else:
                    T = 0.5 * torch.sum(p_vals**2, dim=(1, 2)).view(-1, 1)
                
                H = V + torch.clamp(T, max=1e3)
                
                if not H.requires_grad:
                    H = H + 0.0 * z_flat.sum()
                
                try:
                    dH_dz = torch.autograd.grad(H.sum(), z_flat, create_graph=True, retain_graph=True, allow_unused=True)[0]
                except Exception:
                    dH_dz = None
                
                if dH_dz is None:
                    dH_dz = torch.zeros_like(z_flat)
                else:
                    # CLAMP GRADIENTS for stability during stage 3
                    # Using a very tight clamp to prevent divergence
                    dH_dz = torch.clamp(dH_dz, -2.0, 2.0)
            
            d_sub = self.latent_dim // 2
            dz_dt = torch.zeros_like(z_flat)
            
            for k in range(self.n_super_nodes):
                q_start = k * self.latent_dim
                q_end = q_start + d_sub
                p_start = q_end
                p_end = p_start + d_sub
                
                dz_dt[:, q_start:q_end] = dH_dz[:, p_start:p_end]
                dz_dt[:, p_start:p_end] = -dH_dz[:, q_start:q_end]
                
                if hasattr(self, 'dissipation'):
                    gamma = self.dissipation[k]
                    dz_dt[:, p_start:p_end] -= gamma * z_flat[:, p_start:p_end]
            
            # Final output clipping for safety
            return torch.clamp(self.output_gain * dz_dt, -10.0, 10.0)

        X_norm = self.torch_transformer(z_flat)

        y_preds = []
        for sym_mod in self.sym_modules:
            if sym_mod is not None:
                if X_norm.size(1) != sym_mod.n_inputs:
                    if X_norm.size(1) > sym_mod.n_inputs:
                        X_input = X_norm[:, :sym_mod.n_inputs]
                    else:
                        padding = torch.zeros((X_norm.size(0), sym_mod.n_inputs - X_norm.size(1)),
                                              device=X_norm.device, dtype=X_norm.dtype)
                        X_input = torch.cat([X_norm, padding], dim=1)
                else:
                    X_input = X_norm

                y_preds.append(sym_mod(X_input).view(-1, 1))
            else:
                y_preds.append(torch.zeros((z_flat.size(0), 1), device=z_flat.device, dtype=torch.float32))

        Y_norm_pred = torch.stack(y_preds, dim=1).squeeze(-1)
        Y_pred = self.torch_transformer.denormalize_y(Y_norm_pred)

        # Final output clipping for safety
        return torch.clamp(self.output_gain * Y_pred.to(torch.float32), -10.0, 10.0)
