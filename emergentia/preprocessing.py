import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Dict, Any


class AutoSmoother(nn.Module):
    """
    Learnable Gaussian smoothing parameter that co-evolves with the neural network
    to minimize HuberLoss on noisy trajectories.

    This trait automatically adjusts the smoothing bandwidth based on the noise
    level and optimizes alongside the network to find the best balance between
    noise reduction and signal preservation.
    """

    def __init__(
        self,
        init_bandwidth: float = 1.0,
        min_bandwidth: float = 0.1,
        max_bandwidth: float = 5.0,
    ):
        """
        Initialize the Auto-Smoother.

        Args:
            init_bandwidth: Initial smoothing bandwidth (sigma)
            min_bandwidth: Minimum allowed bandwidth
            max_bandwidth: Maximum allowed bandwidth
        """
        super().__init__()
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth

        # Learnable bandwidth parameter
        self.log_bandwidth = nn.Parameter(
            torch.tensor(np.log(init_bandwidth), dtype=torch.float32),
            requires_grad=True,
        )

        self.is_enabled = True
        self.noise_level = 0.0
        self.optimization_step = 0

    def forward(
        self,
        trajectory: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> torch.Tensor:
        """
        Apply learned Gaussian smoothing to a trajectory.

        Args:
            trajectory: Input trajectory tensor of shape (T, N, D) or (T, D)
            optimizer: Optional optimizer for bandwidth optimization

        Returns:
            Smoothed trajectory
        """
        if not self.is_enabled:
            return trajectory

        # Get current bandwidth from log parameter
        bandwidth = torch.clamp(
            torch.exp(self.log_bandwidth),
            min=self.min_bandwidth,
            max=self.max_bandwidth,
        ).item()

        # Convert to numpy if it's a tensor
        if isinstance(trajectory, torch.Tensor):
            trajectory_np = trajectory.detach().cpu().numpy()
            is_tensor = True
        else:
            trajectory_np = np.array(trajectory)
            is_tensor = False

        # Check dimensions
        if trajectory_np.ndim == 1:
            # Single trajectory (T,)
            smoothed = gaussian_filter1d(trajectory_np, sigma=bandwidth, axis=0)
        elif trajectory_np.ndim == 2:
            # Multiple trajectories (T, D)
            smoothed = gaussian_filter1d(trajectory_np, sigma=bandwidth, axis=0)
        elif trajectory_np.ndim == 3:
            # Batch of trajectories (T, N, D)
            smoothed = np.array(
                [
                    gaussian_filter1d(trajectory_np[t], sigma=bandwidth, axis=0)
                    for t in range(trajectory_np.shape[0])
                ]
            )
        else:
            # Unknown dimension - return original
            if is_tensor:
                return trajectory
            else:
                return trajectory_np

        # Return as tensor if input was tensor
        if is_tensor:
            return torch.from_numpy(smoothed).to(trajectory.device)
        else:
            return smoothed

    def optimize_bandwidth(self, trajectory: torch.Tensor, loss_history: list) -> float:
        """
        Optimize the bandwidth parameter to minimize loss on the trajectory.

        Args:
            trajectory: Noisy trajectory tensor
            loss_history: History of loss values for reference

        Returns:
            Current bandwidth value
        """
        if not self.is_enabled:
            return self.max_bandwidth

        self.optimization_step += 1

        # Convert trajectory to numpy
        trajectory_np = trajectory.detach().cpu().numpy()

        # Ensure we have at least 3 dimensions: (T, N, D)
        if trajectory_np.ndim == 1:
            # Single trajectory: reshape to (T, 1, 1) to work with batch processing
            trajectory_np = trajectory_np[:, np.newaxis, np.newaxis]
        elif trajectory_np.ndim == 2:
            # Multiple features: reshape to (T, 1, D) for single batch
            trajectory_np = trajectory_np[:, np.newaxis, :]

        T, N, D = trajectory_np.shape

        # Compute R^2 score for different bandwidths
        best_score = -1.0
        best_bandwidth = float(self.max_bandwidth)

        num_trials = min(10, int(self.max_bandwidth / 0.5))

        for trial in range(num_trials):
            bandwidth = (
                self.min_bandwidth
                + (self.max_bandwidth - self.min_bandwidth) * (trial + 1) / num_trials
            )

            # Apply smoothing
            smoothed = np.array(
                [
                    gaussian_filter1d(trajectory_np[t], sigma=bandwidth, axis=0)
                    for t in range(T)
                ]
            )

            # Compute R^2 score relative to original
            if T > 1:
                r2 = 1 - np.var(trajectory_np - smoothed) / np.var(trajectory_np)
            else:
                r2 = 0.0

            # Weighted combination: higher R^2 but not too close to 1.0 (avoid over-smoothing)
            score = r2 - 0.1 * np.abs(r2 - 0.9)

            if score > best_score:
                best_score = score
                best_bandwidth = bandwidth

        # Update log bandwidth towards optimal
        target_log_bw = np.log(best_bandwidth)
        current_log_bw = self.log_bandwidth.item()

        # Smooth update
        new_log_bw = current_log_bw * 0.7 + target_log_bw * 0.3
        self.log_bandwidth.data = torch.tensor(
            new_log_bw, dtype=torch.float32, device=self.log_bandwidth.device
        ).requires_grad_(True)

        return torch.clamp(
            torch.exp(self.log_bandwidth),
            min=self.min_bandwidth,
            max=self.max_bandwidth,
        ).item()

    def get_bandwidth(self) -> float:
        """Get the current bandwidth value."""
        return torch.clamp(
            torch.exp(self.log_bandwidth),
            min=self.min_bandwidth,
            max=self.max_bandwidth,
        ).item()

    def get_bandwidth_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the current bandwidth.

        Returns:
            Dictionary with bandwidth information
        """
        bandwidth = self.get_bandwidth()

        return {
            "bandwidth": bandwidth,
            "log_bandwidth": self.log_bandwidth.item(),
            "noise_level": self.noise_level,
            "optimization_step": self.optimization_step,
            "min_bandwidth": self.min_bandwidth,
            "max_bandwidth": self.max_bandwidth,
            "is_enabled": self.is_enabled,
        }

    def set_noise_level(self, noise_level: float):
        """
        Set the expected noise level to adjust smoothing behavior.

        Args:
            noise_level: Expected noise standard deviation
        """
        self.noise_level = noise_level

        # Adaptive bandwidth based on noise level
        if noise_level > 0.05:
            # High noise - use larger bandwidth
            target_bandwidth = max(noise_level * 10, 2.0)
            self.log_bandwidth.data = torch.clamp(
                torch.tensor(np.log(target_bandwidth)),
                min=np.log(self.min_bandwidth),
                max=np.log(self.max_bandwidth),
            ).requires_grad_(True)
        elif noise_level > 0.02:
            # Medium noise
            target_bandwidth = max(noise_level * 5, 1.0)
            self.log_bandwidth.data = torch.clamp(
                torch.tensor(np.log(target_bandwidth)),
                min=np.log(self.min_bandwidth),
                max=np.log(self.max_bandwidth),
            ).requires_grad_(True)
        else:
            # Low noise - minimal smoothing
            target_bandwidth = max(noise_level * 3, 0.5)
            self.log_bandwidth.data = torch.clamp(
                torch.tensor(np.log(target_bandwidth)),
                min=np.log(self.min_bandwidth),
                max=np.log(self.max_bandwidth),
            ).requires_grad_(True)

    def set_from_noise_std(self, noise_std: float):
        """
        Set smoothing behavior based on noise standard deviation.

        Args:
            noise_std: Noise standard deviation (e.g., 0.05 for 5% noise)
        """
        self.set_noise_level(noise_std)

    def disable(self):
        """Disable the auto-smoother."""
        self.is_enabled = False

    def enable(self):
        """Enable the auto-smoother."""
        self.is_enabled = True

    def update_with_loss(self, loss: float, prev_loss: float, trajectory: torch.Tensor):
        """
        Update smoothing based on loss improvement.

        Args:
            loss: Current loss value
            prev_loss: Previous loss value
            trajectory: Current trajectory (for optimization reference)
        """
        if self.optimization_step > 0 and (prev_loss - loss) < 0:
            # Loss is not improving - increase smoothing to reduce noise
            self.log_bandwidth.data = torch.clamp(
                self.log_bandwidth * 1.05,
                min=np.log(self.min_bandwidth),
                max=np.log(self.max_bandwidth),
            ).requires_grad_(True)
        elif self.optimization_step > 0 and (prev_loss - loss) > 0:
            # Loss is improving - reduce smoothing slightly
            self.log_bandwidth.data = torch.clamp(
                self.log_bandwidth * 0.95,
                min=np.log(self.min_bandwidth),
                max=np.log(self.max_bandwidth),
            ).requires_grad_(True)


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


class TrajectorySmoother:
    """
    Wrapper class for simpler API access to AutoSmoother.
    Provides a trait-like interface for smoother integration with existing code.
    """

    def __init__(self, enable_autosmoothing: bool = True):
        """
        Initialize the trajectory smoother.

        Args:
            enable_autosmoothing: Whether to enable learnable bandwidth
        """
        self.enable_autosmoothing = enable_autosmoothing
        self.smoother = AutoSmoother() if enable_autosmoothing else None

    def smooth(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Apply smoothing to a trajectory.

        Args:
            trajectory: Input trajectory tensor

        Returns:
            Smoothed trajectory
        """
        if self.smoother is not None:
            return self.smoother(trajectory)
        return trajectory

    def get_bandwidth(self) -> Optional[float]:
        """Get the current bandwidth value."""
        if self.smoother is not None:
            return self.smoother.get_bandwidth()
        return None

    def optimize(self, trajectory: torch.Tensor, loss_history: list = None) -> float:
        """
        Optimize the smoothing bandwidth.

        Args:
            trajectory: Input trajectory
            loss_history: Optional loss history

        Returns:
            Optimized bandwidth
        """
        if self.smoother is not None:
            return self.smoother.optimize_bandwidth(trajectory, loss_history or [])
        return 1.0

    def update_from_noise_std(self, noise_std: float):
        """
        Update smoothing from noise standard deviation.

        Args:
            noise_std: Noise standard deviation
        """
        if self.smoother is not None:
            self.smoother.set_from_noise_std(noise_std)

    def get_info(self) -> Dict[str, Any]:
        """Get smoother information."""
        if self.smoother is not None:
            return self.smoother.get_bandwidth_info()
        return {"bandwidth": None, "is_enabled": False}


class GaussianSmoother(nn.Module):
    """
    Fixed-parameter Gaussian smoother for comparison baseline.
    """

    def __init__(self, sigma: float = 1.0):
        """
        Initialize the fixed-parameter smoother.

        Args:
            sigma: Gaussian kernel sigma
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian smoothing with fixed sigma.

        Args:
            trajectory: Input trajectory

        Returns:
            Smoothed trajectory
        """
        if isinstance(trajectory, torch.Tensor):
            trajectory_np = trajectory.detach().cpu().numpy()
            is_tensor = True
        else:
            trajectory_np = np.array(trajectory)
            is_tensor = False

        if trajectory_np.ndim == 1:
            smoothed = gaussian_filter1d(trajectory_np, sigma=self.sigma, axis=0)
        elif trajectory_np.ndim == 2:
            smoothed = gaussian_filter1d(trajectory_np, sigma=self.sigma, axis=0)
        elif trajectory_np.ndim == 3:
            smoothed = np.array(
                [
                    gaussian_filter1d(trajectory_np[t], sigma=self.sigma, axis=0)
                    for t in range(trajectory_np.shape[0])
                ]
            )
        else:
            if is_tensor:
                return trajectory
            else:
                return trajectory_np

        if is_tensor:
            return torch.from_numpy(smoothed).to(trajectory.device)
        else:
            return smoothed
