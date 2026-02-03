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


class LearnableAutoSmoother(nn.Module):
    """
    Learnable auto-smoother that co-evolves with the neural network.
    Uses learnable bandwidth parameter optimized via backpropagation.
    """

    def __init__(
        self,
        init_bandwidth: float = 1.0,
        min_bandwidth: float = 0.1,
        max_bandwidth: float = 5.0,
    ):
        """
        Initialize the learnable auto-smoother.

        Args:
            init_bandwidth: Initial smoothing bandwidth
            min_bandwidth: Minimum bandwidth
            max_bandwidth: Maximum bandwidth
        """
        super().__init__()
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        # Learnable bandwidth (log space for stability)
        self.log_bandwidth = nn.Parameter(
            torch.tensor(np.log(init_bandwidth), dtype=torch.float32)
        )

        self.current_bandwidth = init_bandwidth
        self.training_step = 0

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable smoothing to trajectory.

        Args:
            trajectory: Input trajectory (T, N, D) or (T, D) or (T,)

        Returns:
            Smoothed trajectory
        """
        # Get current bandwidth
        bandwidth = torch.clamp(
            torch.exp(self.log_bandwidth),
            min=self.min_bandwidth,
            max=self.max_bandwidth,
        ).item()

        self.current_bandwidth = bandwidth

        # Convert to numpy
        if isinstance(trajectory, torch.Tensor):
            trajectory_np = trajectory.detach().cpu().numpy()
            is_tensor = True
        else:
            trajectory_np = np.array(trajectory)
            is_tensor = False

        # Apply smoothing
        if trajectory_np.ndim == 1:
            smoothed = gaussian_filter1d(trajectory_np, sigma=bandwidth, axis=0)
        elif trajectory_np.ndim == 2:
            smoothed = gaussian_filter1d(trajectory_np, sigma=bandwidth, axis=0)
        elif trajectory_np.ndim == 3:
            smoothed = np.array(
                [
                    gaussian_filter1d(trajectory_np[t], sigma=bandwidth, axis=0)
                    for t in range(trajectory_np.shape[0])
                ]
            )
        else:
            if is_tensor:
                return trajectory
            else:
                return trajectory_np

        # Return as tensor
        if is_tensor:
            return torch.from_numpy(smoothed).to(trajectory.device)
        else:
            return smoothed

    def update_from_loss(self, prev_loss: float, current_loss: float):
        """
        Update bandwidth based on loss improvement.

        Args:
            prev_loss: Previous loss value
            current_loss: Current loss value
        """
        self.training_step += 1

        if self.training_step > 0 and (prev_loss - current_loss) > 0:
            # Loss improving - slightly reduce smoothing
            self.log_bandwidth.data = torch.clamp(
                self.log_bandwidth * 0.95,
                min=np.log(self.min_bandwidth),
                max=np.log(self.max_bandwidth),
            )
        elif self.training_step > 0 and (prev_loss - current_loss) < 0:
            # Loss worsening - increase smoothing
            self.log_bandwidth.data = torch.clamp(
                self.log_bandwidth * 1.05,
                min=np.log(self.min_bandwidth),
                max=np.log(self.max_bandwidth),
            )

    def get_bandwidth(self) -> float:
        """Get current bandwidth."""
        return torch.clamp(
            torch.exp(self.log_bandwidth),
            min=self.min_bandwidth,
            max=self.max_bandwidth,
        ).item()

    def get_info(self) -> Dict[str, Any]:
        """Get bandwidth information."""
        return {
            "bandwidth": float(self.get_bandwidth()),
            "log_bandwidth": float(self.log_bandwidth.item()),
            "training_step": self.training_step,
            "min_bandwidth": self.min_bandwidth,
            "max_bandwidth": self.max_bandwidth,
        }


class DiscoveryNet(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        # The network now predicts the Potential V(r)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def _get_features(self, dist):
        # Generic atomic-like features: r and 1/r
        dist_safe = torch.clamp(dist, min=0.1, max=50.0)
        return torch.cat([dist_safe, 1.0 / dist_safe], dim=-1)

    def forward(self, pos_scaled):
        # pos_scaled: (batch, n_particles, dim)
        if not pos_scaled.requires_grad:
            pos_scaled = pos_scaled.clone().requires_grad_(True)

        diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1)  # (batch, n, n, dim)
        dist = torch.norm(diff, dim=-1, keepdim=True)  # (batch, n, n, 1)

        feat = self._get_features(dist)  # (batch, n, n, 2)
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
        # Use allow_unused=True just in case, though it shouldn't be needed here
        forces = -torch.autograd.grad(
            v_total, pos_scaled, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        if forces is None:
            forces = torch.zeros_like(pos_scaled)
        return forces

    def predict_mag(self, r_scaled):
        # r_scaled: (num_points, 1)
        with torch.enable_grad():
            r_scaled = r_scaled.clone().requires_grad_(True)
            feat = self._get_features(r_scaled)
            v = self.net(feat)
            # Force magnitude F(r) = -dV/dr
            dv_dr = torch.autograd.grad(v.sum(), r_scaled, create_graph=True)[0]
        return -dv_dr
