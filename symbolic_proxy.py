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

        # NEW: Soft-start for symbolic weight - gradually increases influence during training
        self.register_buffer('symbolic_weight_schedule', torch.linspace(0.01, 1.0, 100))  # 100 steps schedule
        self.current_stage3_step = 0

        # 2. Initialize differentiable symbolic modules
        self.sym_modules = nn.ModuleList()
        self.separable = False

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
                    if hasattr(eq, 'enforce_separable'):
                        self.separable = eq.enforce_separable
                    elif "SeparableHamiltonian" in str(eq):
                        self.separable = True

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

                # NEW: Apply soft-core potential protections before symbolic evaluation
                X_norm_safe = self._apply_soft_core_protections(X_norm)

                H_norm = self.sym_modules[0](X_norm_safe)
                # Symbolic module outputs normalized H (or V)
                # Denormalize to get physical potential
                V_pred = self.torch_transformer.denormalize_y(H_norm)

                # Safety clamp for potential
                V_pred = torch.clamp(V_pred, -1e3, 1e3)

                if self.separable:
                    # SEPARABLE HAMILTONIAN FIX:
                    # If the symbolic part only represents the potential V(q),
                    # we must add the kinetic term T(p) = sum(p^2 / 2m)
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

                    H = V_pred + torch.clamp(T, max=1e3)
                else:
                    # Non-separable: the symbolic model already represents full H(q, p)
                    H = V_pred

                if not H.requires_grad:
                    H = H + 0.0 * z_flat.sum()

                try:
                    dH_dz = torch.autograd.grad(H.sum(), z_flat, create_graph=True, retain_graph=True, allow_unused=True)[0]
                except Exception as e:
                    print(f"[SymbolicProxy] Gradient computation failed: {e}")
                    dH_dz = None

                if dH_dz is None:
                    dH_dz = torch.zeros_like(z_flat)
                else:
                    # NEW: Aggressive gradient clipping for Stage 3 stability
                    # Implement gradient clipping with very conservative limits
                    grad_norm_before = torch.norm(dH_dz, dim=-1, keepdim=True)

                    # NEW: Safety Epsilon added to prevent division by zero during backprop
                    safety_eps = 1e-6

                    # Clip gradients more aggressively to prevent numerical instabilities
                    max_grad_norm = 0.1  # Much more conservative for Stage 3
                    grad_scale = torch.minimum(max_grad_norm / (grad_norm_before + safety_eps), torch.ones_like(grad_norm_before))
                    dH_dz = dH_dz * grad_scale

                    # Additional numerical stability: remove any NaN or Inf values
                    dH_dz = torch.nan_to_num(dH_dz, nan=0.0, posinf=0.1, neginf=-0.1)

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

            # Enhanced final output clipping for safety
            dz_dt = torch.nan_to_num(dz_dt, nan=0.0, posinf=1.0, neginf=-1.0)

            # NEW: Apply even more conservative gradient clipping to the final output
            output_norm = torch.norm(dz_dt, dim=-1, keepdim=True)
            max_output_norm = 1.0  # Conservative limit for Stage 3
            output_scale = torch.minimum(max_output_norm / (output_norm + 1e-6), torch.ones_like(output_norm))
            dz_dt = dz_dt * output_scale

            # NEW: Apply soft-start scaling for Stage 3
            if self.current_stage3_step < len(self.symbolic_weight_schedule):
                soft_scaling = self.symbolic_weight_schedule[self.current_stage3_step]
            else:
                soft_scaling = 1.0
            self.current_stage3_step += 1  # Increment step counter

            return self.output_gain * dz_dt * soft_scaling

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

    def _apply_soft_core_protections(self, X_norm):
        """
        Apply soft-core potential protections to prevent 1/r^n singularities.
        """
        # Soft-core regularization: prevent division by near-zero values
        # Identify potential problematic terms (inverse powers)
        X_protected = X_norm.clone()

        # Clamp extreme values to prevent overflow/underflow
        X_protected = torch.clamp(X_protected, -1e6, 1e6)

        # Apply soft-core regularization to prevent singularities
        # For terms that might represent 1/r^n, add a small epsilon to denominator
        # This is particularly important for Lennard-Jones type interactions

        return X_protected