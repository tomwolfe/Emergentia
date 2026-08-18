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

    def fit(self, p, f):
        self.p_scale = max(torch.max(torch.abs(p)).item(), 1e-8)
        self.f_scale = max(torch.max(torch.abs(f)).item(), 1e-8)

    def transform(self, p, f):
        return p / self.p_scale, f / self.f_scale

    def inverse_transform_f(self, f_scaled):
        return f_scaled * self.f_scale

    def fit_for_potential(self, p, f, potential):
        """Set scale using potential's default_scale for potential-based training."""
        self.p_scale = max(potential.default_scale, 1e-8)
        self.f_scale = max(torch.max(torch.abs(f)).item(), 1e-8)
        self.v_scale = max(abs(torch.max(potential.compute_potential(torch.tensor([self.p_scale]))).item()), 1e-8)


class DiscoveryNet(nn.Module):
    def __init__(self, hidden_size=128, n_features=6):
        super().__init__()
        self.n_features = n_features
        # The network predicts the Potential V(r)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def _get_features(self, dist):
        dist_safe = torch.clamp(dist, min=0.1, max=50.0)
        return torch.cat(
            [
                dist_safe,                  # r
                1.0 / dist_safe,            # 1/r
                dist_safe ** 2,             # r^2
                1.0 / (dist_safe ** 2),     # 1/r^2
                torch.exp(-dist_safe),      # exp(-r)
                torch.log(dist_safe + 1.0), # log(r+1)
            ],
            dim=-1,
        )

    def forward(self, pos_scaled):
        # pos_scaled: (batch, n_particles, dim)
        pos_scaled = pos_scaled.detach().requires_grad_(True)

        with torch.enable_grad():
            diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1)  # (batch, n, n, dim)
            dist = torch.norm(diff, dim=-1, keepdim=True)  # (batch, n, n, 1)

            feat = self._get_features(dist)  # (batch, n, n, 6)
            v_pair = self.net(feat)  # (batch, n, n, 1)

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
        r_scaled = r_scaled.view(-1, 1)
        with torch.enable_grad():
            r_scaled = r_scaled.clone().requires_grad_(True)
            feat = self._get_features(r_scaled)
            v = self.net(feat)
            # Force magnitude F(r) = -dV/dr
            dv_dr = torch.autograd.grad(v.sum(), r_scaled, create_graph=True)[0]
        return -dv_dr
    
    def forward_potential(self, r_scaled):
        # r_scaled: (num_points, 1)
        # Compute potential energy for scalar distances
        feat = self._get_features(r_scaled)
        return self.net(feat)


class EnsembleDiscoveryNet(nn.Module):
    """
    Ensemble of DiscoveryNet models that provides uncertainty estimates
    on force predictions by computing mean and std across members.
    """
    
    def __init__(self, n_members=5, hidden_size=128, n_features=6):
        super().__init__()
        self.n_members = n_members
        self.n_features = n_features
        self.members = nn.ModuleList([
            DiscoveryNet(hidden_size=hidden_size, n_features=n_features)
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
