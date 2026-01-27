import torch
import numpy as np

class PhysicsSim:
    def __init__(self, n=4, mode='lj', seed=None, device='cpu'):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.device = device
        self.n = n
        self.mode = mode
        # LJ needs smaller dt for stability
        self.dt = 0.001 if mode == 'lj' else 0.01
        
        scale = 2.0 if mode == 'spring' else 3.5
        self.pos = torch.rand((n, 2), device=self.device, dtype=torch.float32) * scale
        self.vel = torch.randn((n, 2), device=self.device, dtype=torch.float32) * 0.1
        self.mask = (~torch.eye(n, device=self.device).bool()).unsqueeze(-1)

    def compute_forces(self, pos):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=1e-6)
        
        if self.mode == 'spring':
            # F = -k * (r - r0)
            f_mag = -10.0 * (dist - 1.0)
        else:
            # LJ: F = 48*(1/r^13) - 24*(1/r^7)
            d_inv = 1.0 / torch.clamp(dist, min=0.5, max=5.0)
            d_inv_13 = torch.clamp(torch.pow(d_inv, 13), max=1e10)
            d_inv_7 = torch.clamp(torch.pow(d_inv, 7), max=1e6)
            f_mag = 48.0 * d_inv_13 - 24.0 * d_inv_7
            
        f = f_mag * (diff / dist_clip) * self.mask
        return torch.sum(f, dim=1)

    def generate(self, steps=2000, noise_std=0.0):
        traj_p = torch.zeros((steps, self.n, 2), device=self.device, dtype=torch.float32)
        traj_f = torch.zeros((steps, self.n, 2), device=self.device, dtype=torch.float32)

        curr_pos = self.pos.clone()
        curr_vel = self.vel.clone()

        with torch.no_grad():
            for i in range(steps):
                f = self.compute_forces(curr_pos)
                traj_f[i] = f

                # Occasional random impulses to explore state space
                if i % 500 == 0:
                    impulse_factor = 0.5 if self.mode == 'lj' else 0.3
                    curr_vel += torch.randn_like(curr_vel) * impulse_factor

                curr_vel += f * self.dt
                curr_vel *= 0.99 # Damping for stability
                curr_pos += curr_vel * self.dt
                traj_p[i] = curr_pos

        if noise_std > 0:
            traj_p += torch.randn_like(traj_p) * noise_std

        return traj_p, traj_f
