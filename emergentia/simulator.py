import torch
import numpy as np
import time

class PhysicsSim:
    def __init__(self, n=4, mode='lj', seed=None, device=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.n = n
        self.mode = mode
        self.dt = 0.001 if mode == 'lj' else 0.01
        
        scale = 2.0 if mode == 'spring' else 3.5
        self.pos = torch.rand((n, 2), device=self.device, dtype=torch.float32) * scale
        self.vel = torch.randn((n, 2), device=self.device, dtype=torch.float32) * 0.1
        self.mask = (~torch.eye(n, device=self.device).bool()).unsqueeze(-1)

        # Force computation is the bottleneck
        self._compute_forces_compiled = torch.compile(self._compute_forces_raw) if hasattr(torch, 'compile') and self.device.type != 'mps' else self._compute_forces_raw

    def _compute_forces_raw(self, pos):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=1e-6)
        
        if self.mode == 'spring':
            f_mag = -10.0 * (dist - 1.0)
        else:
            d_inv = 1.0 / torch.clamp(dist, min=0.5, max=5.0)
            d_inv_13 = torch.pow(d_inv, 13)
            d_inv_7 = torch.pow(d_inv, 7)
            f_mag = 48.0 * d_inv_13 - 24.0 * d_inv_7
            
        f = f_mag * (diff / dist_clip) * self.mask
        return torch.sum(f, dim=1)

    def generate(self, steps=2000, noise_std=0.0):
        start = time.perf_counter()
        
        # Pre-allocate on device
        traj_p = torch.zeros((steps, self.n, 2), device=self.device)
        traj_f = torch.zeros((steps, self.n, 2), device=self.device)

        curr_pos = self.pos.clone()
        curr_vel = self.vel.clone()

        # Optimize: Move some logic outside the loop if possible
        dt = self.dt
        
        for i in range(steps):
            # Impulse check
            if i % 500 == 0:
                impulse_factor = 0.5 if self.mode == 'lj' else 0.3
                curr_vel += torch.randn_like(curr_vel) * impulse_factor

            f = self._compute_forces_compiled(curr_pos)
            traj_f[i] = f
            
            curr_vel += f * dt
            curr_vel *= 0.99 
            curr_pos += curr_vel * dt
            traj_p[i] = curr_pos

        if noise_std > 0:
            traj_p += torch.randn_like(traj_p) * noise_std
            
        sim_time = (time.perf_counter() - start) * 1000
        print(f"Simulation of {steps} steps took {sim_time:.2f}ms")

        return traj_p, traj_f