import torch
import numpy as np
import time
from abc import ABC, abstractmethod

class Potential(ABC):
    @abstractmethod
    def compute_force_magnitude(self, dist):
        pass

    @abstractmethod
    def compute_potential(self, dist):
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

    def compute_potential(self, dist):
        return 0.5 * self.k * torch.pow(dist - self.r0, 2)

    @property
    def default_scale(self):
        return 2.0

    @property
    def dt(self):
        return 0.01

class LennardJonesPotential(Potential):
    def __init__(self, epsilon=1.0, sigma=1.0):
        self.eps = epsilon
        self.sig = sigma
        # 48 * eps * (sigma^12 / r^13) - 24 * eps * (sigma^6 / r^7)
        self.a = 48.0 * epsilon * (sigma**12)
        self.b = 24.0 * epsilon * (sigma**6)

    def compute_force_magnitude(self, dist):
        d_inv = 1.0 / torch.clamp(dist, min=0.5, max=5.0)
        return self.a * torch.pow(d_inv, 13) - self.b * torch.pow(d_inv, 7)

    def compute_potential(self, dist):
        # V(r) = 4 * eps * ((sigma/r)^12 - (sigma/r)^6)
        s_r = self.sig / torch.clamp(dist, min=0.5, max=5.0)
        return 4.0 * self.eps * (torch.pow(s_r, 12) - torch.pow(s_r, 6))

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
        diff = dist - self.re
        exp_neg_a = torch.exp(-self.a * diff)
        exp_neg_2a = torch.exp(-2.0 * self.a * diff)
        return 2.0 * self.De * self.a * (exp_neg_a - exp_neg_2a)

    def compute_potential(self, dist):
        # V(r) = De * (1 - exp(-a(r-re)))^2
        diff = dist - self.re
        val = 1.0 - torch.exp(-self.a * diff)
        return self.De * torch.pow(val, 2)

    @property
    def default_scale(self):
        return 3.0

    @property
    def dt(self):
        return 0.005

class GravityPotential(Potential):
    def __init__(self, G=1.0):
        self.G = G

    def compute_force_magnitude(self, dist):
        # F = -G / r^2
        return -self.G / torch.pow(torch.clamp(dist, min=0.1), 2)

    def compute_potential(self, dist):
        # V = -G / r
        return -self.G / torch.clamp(dist, min=0.1)

    @property
    def default_scale(self):
        return 5.0

    @property
    def dt(self):
        return 0.01

class CompositePotential(Potential):
    def __init__(self, potentials):
        self.potentials = potentials

    def compute_force_magnitude(self, dist):
        return sum(p.compute_force_magnitude(dist) for p in self.potentials)

    def compute_potential(self, dist):
        return sum(p.compute_potential(dist) for p in self.potentials)

    @property
    def default_scale(self):
        return max(p.default_scale for p in self.potentials)

    @property
    def dt(self):
        return min(p.dt for p in self.potentials)

class PhysicsSim:
    def __init__(self, n=4, dim=2, potential=None, seed=None, device=None):
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
        self.dim = dim
        self.potential = potential if potential is not None else LennardJonesPotential()
        self.dt = self.potential.dt
        
        scale = self.potential.default_scale
        self.pos = torch.rand((n, self.dim), device=self.device, dtype=torch.float32) * scale
        self.vel = torch.randn((n, self.dim), device=self.device, dtype=torch.float32) * 0.1
        self.mask = (~torch.eye(n, device=self.device).bool()).unsqueeze(-1)

        # Force computation is the bottleneck
        self._compute_forces_compiled = torch.compile(self._compute_forces_raw) if hasattr(torch, 'compile') and self.device.type != 'mps' else self._compute_forces_raw

    def get_hamiltonian(self, pos=None, vel=None):
        p = pos if pos is not None else self.pos
        v = vel if vel is not None else self.vel
        
        # Kinetic Energy: 0.5 * sum(v^2)
        ke = 0.5 * torch.sum(v**2)
        
        # Potential Energy: sum_{i<j} V(r_ij)
        diff = p.unsqueeze(1) - p.unsqueeze(0)
        dist = torch.norm(diff, dim=-1)
        
        n = self.n
        indices = torch.triu_indices(n, n, offset=1, device=self.device)
        dist_pairs = dist[indices[0], indices[1]]
        
        pe = torch.sum(self.potential.compute_potential(dist_pairs))
        return ke + pe

    def _compute_forces_raw(self, pos):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=1e-6)
        
        # Validation check for numerical instability
        if torch.any(dist + torch.eye(self.n, device=self.device).unsqueeze(-1) * 10.0 < 0.1):
            # We don't want to print every step in a compiled function, but this is a raw fallback
            pass

        f_mag = self.potential.compute_force_magnitude(dist)
            
        f = f_mag * (diff / dist_clip) * self.mask
        return torch.sum(f, dim=1)

    def generate(self, steps=2000, noise_std=0.0, impulses=True):
        start = time.perf_counter()
        
        # Pre-allocate on device
        traj_p = torch.zeros((steps, self.n, self.dim), device=self.device)
        traj_f = torch.zeros((steps, self.n, self.dim), device=self.device)

        curr_pos = self.pos.clone()
        curr_vel = self.vel.clone()

        dt = self.dt
        
        # Initial force
        f = self._compute_forces_compiled(curr_pos)
        
        for i in range(steps):
            # Impulse check
            if impulses and i % 500 == 0:
                impulse_factor = 0.5 if isinstance(self.potential, LennardJonesPotential) else 0.3
                curr_vel += torch.randn_like(curr_vel) * impulse_factor

            # Velocity Verlet: 
            # 1. v(t + dt/2) = v(t) + 0.5 * a(t) * dt
            curr_vel += 0.5 * f * dt
            
            # 2. x(t + dt) = x(t) + v(t + dt/2) * dt
            curr_pos += curr_vel * dt
            
            # 3. a(t + dt)
            f = self._compute_forces_compiled(curr_pos)
            traj_f[i] = f
            
            # 4. v(t + dt) = v(t + dt/2) + 0.5 * a(t + dt) * dt
            curr_vel += 0.5 * f * dt
            
            traj_p[i] = curr_pos
            
            # Periodic distance check (sampling instead of every step to avoid sync overhead)
            if i % 100 == 0:
                with torch.no_grad():
                    dist_check = torch.norm(curr_pos.unsqueeze(1) - curr_pos.unsqueeze(0), dim=-1)
                    dist_check = dist_check + torch.eye(self.n, device=self.device) * 10.0
                    if torch.any(dist_check < 0.1):
                        print(f"Warning: Step {i} | Minimum distance {torch.min(dist_check):.4f} < 0.1. Numerical instability likely.")

        self.pos = curr_pos.clone()
        self.vel = curr_vel.clone()

        if noise_std > 0:
            traj_p += torch.randn_like(traj_p) * noise_std
            
        sim_time = (time.perf_counter() - start) * 1000
        print(f"Simulation of {steps} steps took {sim_time:.2f}ms")

        return traj_p, traj_f
