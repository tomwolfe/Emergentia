import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from scipy.ndimage import gaussian_filter1d
from .registry import PhysicalBasisRegistry
from .preprocessing import AutoSmoother


class TrajectoryScaler:
    def __init__(self, mode="lj"):
        self.mode = mode
        self.p_scale = 1.0
        self.f_scale = 1.0

    def fit(self, p, f, potential=None):
        # Use potential's default_scale if available, else max position
        if potential is not None and hasattr(potential, 'default_scale'):
            self.p_scale = max(potential.default_scale, 1e-8)
        else:
            self.p_scale = max(torch.max(torch.abs(p)).item(), 1e-8)
        # Use 95th percentile of |f| for robust scaling (avoids extreme outliers)
        f_abs = torch.abs(f).flatten()
        f_95 = torch.quantile(f_abs.float(), 0.95).item()
        self.f_scale = max(f_95, 1e-8)

    def transform(self, p, f):
        return p / self.p_scale, f / self.f_scale

    def inverse_transform_f(self, f_scaled):
        return f_scaled * self.f_scale

    def fit_for_potential(self, p, f, potential):
        """Set scale using potential's default_scale for potential-based training."""
        self.p_scale = max(potential.default_scale, 1e-8)
        f_abs = torch.abs(f).flatten()
        f_95 = torch.quantile(f_abs.float(), 0.95).item()
        self.f_scale = max(f_95, 1e-8)
        self.v_scale = max(abs(torch.max(potential.compute_potential(torch.tensor([self.p_scale]))).item()), 1e-8)


class DiscoveryNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        # Basis-free: the network takes the raw pairwise distance r as its only
        # input and learns the potential V(r) directly. No feature dictionary.
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        # Numerical floor only (prevents 1/r singularities at coincident
        # particles). No upper bound: clamping to the training-data range would
        # flatten the learned force outside that range and corrupt distillation.
        self._dist_min = 1e-4

    def set_dist_range(self, r_min, r_max):
        """Retained for API compatibility. Only the lower floor is used."""
        self._dist_min = max(r_min * 0.1, 1e-6)

    def forward(self, pos_scaled):
        # pos_scaled: (batch, n_particles, dim)
        pos_scaled = pos_scaled.detach().requires_grad_(True)

        with torch.enable_grad():
            diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1)  # (batch, n, n, dim)
            dist = torch.norm(diff, dim=-1, keepdim=True)  # (batch, n, n, 1)
            dist_safe = torch.clamp(dist, min=self._dist_min)

            v_pair = self.net(dist_safe)  # (batch, n, n, 1)

            # Mask out self-interaction
            mask = (
                (~torch.eye(pos_scaled.shape[1], device=pos_scaled.device).bool())
                .unsqueeze(0)
                .unsqueeze(-1)
            )

            # Total potential energy (sum of pairs / 2)
            v_total = torch.sum(v_pair * mask) * 0.5

            # Force is negative gradient of potential energy
            forces = -torch.autograd.grad(
                v_total, pos_scaled, create_graph=True, retain_graph=True, allow_unused=True
            )[0]
        if forces is None:
            forces = torch.zeros_like(pos_scaled)
        return forces

    def predict_mag(self, r_scaled):
        # r_scaled: (num_points, 1)
        # Force magnitude F(r) = -dV/dr via autograd on the learned potential.
        r_safe = torch.clamp(r_scaled, min=self._dist_min)
        with torch.enable_grad():
            r_safe = r_safe.clone().requires_grad_(True)
            v = self.net(r_safe)  # potential values (num_points, 1)
            dv_dr = torch.autograd.grad(v.sum(), r_safe, create_graph=True)[0]
        return -dv_dr

    def forward_potential(self, r_scaled):
        # r_scaled: (num_points, 1)
        # Compute potential energy for scalar distances
        r_safe = torch.clamp(r_scaled, min=self._dist_min)
        return self.net(r_safe.view(-1, 1))


class EnsembleDiscoveryNet(nn.Module):
    """
    Ensemble of DiscoveryNet models that provides uncertainty estimates
    on force predictions by computing mean and std across members.
    """
    
    def __init__(self, n_members=3, hidden_size=128):
        super().__init__()
        self.n_members = n_members
        self.members = nn.ModuleList([
            DiscoveryNet(hidden_size=hidden_size)
            for _ in range(n_members)
        ])
    
    def forward(self, pos_scaled):
        # pos_scaled: (batch, n_particles, dim)
        # Return forces for each member in the ensemble
        member_forces = [member(pos_scaled) for member in self.members]
        # Stack: (n_members, batch, n_particles, dim)
        forces_stack = torch.stack(member_forces, dim=0)
        # Compute mean and std across members
        force_mean = torch.mean(forces_stack, dim=0)
        force_std = torch.std(forces_stack, dim=0)
        return force_mean, force_std
    
    def predict_mag(self, r_scaled):
        # r_scaled: (num_points, 1)
        # Return mean and std of force magnitudes across ensemble
        r_scaled = r_scaled.view(-1, 1)
        member_mags = [member.predict_mag(r_scaled) for member in self.members]
        # Stack: (n_members, num_points)
        mags_stack = torch.stack(member_mags, dim=0)
        mag_mean = torch.mean(mags_stack, dim=0)
        mag_std = torch.std(mags_stack, dim=0)
        return mag_mean, mag_std
