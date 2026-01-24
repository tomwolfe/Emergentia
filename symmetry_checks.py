import torch
import numpy as np
import sympy as sp

class NoetherChecker:
    """Verifies physical symmetries and their corresponding conserved quantities."""
    def __init__(self, proxy, latent_dim):
        self.proxy = proxy
        self.latent_dim = latent_dim
        self.n_super_nodes = proxy.n_super_nodes

    def check_rotational_invariance(self, z):
        """Checks if H(R*q, R*p) == H(q, p) for small rotation R."""
        if not getattr(self.proxy, 'is_hamiltonian', False):
            return 0.0
            
        # 1. Rotate z
        theta = 0.1 # Small angle
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        R = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]], dtype=torch.float32, device=z.device)
        
        z_rot = z.clone().view(-1, self.n_super_nodes, self.latent_dim)
        # Apply rotation to each (q_x, q_y) and (p_x, p_y) pair
        for k in range(self.n_super_nodes):
            for i in range(0, self.latent_dim, 2):
                if i+1 < self.latent_dim:
                    z_rot[:, k, i:i+2] = torch.matmul(z_rot[:, k, i:i+2], R.t())
        
        z_rot = z_rot.view(z.shape)
        
        # 2. Compare Hamiltonian
        with torch.no_grad():
            X_norm = self.proxy.torch_transformer(z)
            H = self.proxy.sym_modules[0](X_norm)
            
            X_rot_norm = self.proxy.torch_transformer(z_rot)
            H_rot = self.proxy.sym_modules[0](X_rot_norm)
            
            diff = torch.abs(H - H_rot).mean().item()
            # Normalize by H magnitude
            rel_diff = diff / (torch.abs(H).mean().item() + 1e-9)
            
        return float(rel_diff)

    def check_angular_momentum_conservation(self, z0, steps=1000, dt=0.01):
        """Checks if L = sum(q x p) is conserved during integration."""
        if self.latent_dim < 4: return 1.0 # Need at least 2D q and p
        
        z = z0.clone().view(1, -1)
        L_history = []
        
        def get_L(zt):
            zt_view = zt.view(self.n_super_nodes, self.latent_dim)
            L_total = 0.0
            for k in range(self.n_super_nodes):
                q = zt_view[k, :2]
                p = zt_view[k, 2:4]
                L_total += q[0]*p[1] - q[1]*p[0] # 2D cross product
            return L_total.item()

        for _ in range(steps):
            L_history.append(get_L(z))
            dz = self.proxy(z)
            z = z + dz * dt
            
        L_history = np.array(L_history)
        initial_L = L_history[0]
        if abs(initial_L) < 1e-9:
            drift = np.max(np.abs(L_history - initial_L))
        else:
            drift = np.max(np.abs(L_history - initial_L) / abs(initial_L))
            
        return float(drift)
