import torch
import numpy as np
import time
from abc import ABC, abstractmethod

class Potential(ABC):
    @abstractmethod
    def compute_force_magnitude(self, dist):
        pass

    @property
    @abstractmethod
    def default_scale(self):
        pass

    @property
    @abstractmethod
    def dt(self):
        pass

class HarmonicPotential(Potential):
    def __init__(self, k=10.0, r0=1.0):
        self.k = k
        self.r0 = r0

    def compute_force_magnitude(self, dist):
        return -self.k * (dist - self.r0)

    @property
    def default_scale(self):
        return 2.0

    @property
    def dt(self):
        return 0.01

class LennardJonesPotential(Potential):
    def __init__(self, epsilon=1.0, sigma=1.0):
        # 48 * eps * (sigma^12 / r^13) - 24 * eps * (sigma^6 / r^7)
        self.a = 48.0 * epsilon * (sigma**12)
        self.b = 24.0 * epsilon * (sigma**6)

    def compute_force_magnitude(self, dist):
        d_inv = 1.0 / torch.clamp(dist, min=0.5, max=5.0)
        return self.a * torch.pow(d_inv, 13) - self.b * torch.pow(d_inv, 7)

    @property
    def default_scale(self):
        return 3.5

    @property
    def dt(self):
        return 0.001

class MorsePotential(Potential):
    def __init__(self, De=1.0, a=1.0, re=1.0):
        self.De = De
        self.a = a
        self.re = re

    def compute_force_magnitude(self, dist):
        # F(r) = 2 * De * a * (exp(-a(r-re)) - exp(-2a(r-re)))
        # Note: Using the formula provided in the prompt
        diff = dist - self.re
        exp_neg_a = torch.exp(-self.a * diff)
        exp_neg_2a = torch.exp(-2.0 * self.a * diff)
        return 2.0 * self.De * self.a * (exp_neg_a - exp_neg_2a)

    @property
    def default_scale(self):
        return 3.0

    @property
    def dt(self):
        return 0.005

class PhysicsSim:
    def __init__(self, n=4, potential=None, seed=None, device=None):
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
        self.potential = potential if potential is not None else LennardJonesPotential()
        self.dt = self.potential.dt
        
        scale = self.potential.default_scale
        self.pos = torch.rand((n, 2), device=self.device, dtype=torch.float32) * scale
        self.vel = torch.randn((n, 2), device=self.device, dtype=torch.float32) * 0.1
        self.mask = (~torch.eye(n, device=self.device).bool()).unsqueeze(-1)

        # Force computation is the bottleneck
        self._compute_forces_compiled = torch.compile(self._compute_forces_raw) if hasattr(torch, 'compile') and self.device.type != 'mps' else self._compute_forces_raw

    def _compute_forces_raw(self, pos):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=1e-6)
        
        f_mag = self.potential.compute_force_magnitude(dist)
            
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
                impulse_factor = 0.5 if isinstance(self.potential, LennardJonesPotential) else 0.3
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
