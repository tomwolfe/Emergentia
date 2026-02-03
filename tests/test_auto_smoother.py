import pytest
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
from emergentia.preprocessing import (
    AutoSmoother,
    TrajectorySmoother,
    GaussianSmoother,
    LearnableAutoSmoother,
)


class TestAutoSmoother:
    def test_init_default(self):
        """Test AutoSmoother initialization with defaults."""
        smoother = AutoSmoother()
        assert smoother.is_enabled
        assert smoother.min_bandwidth <= 0.1
        assert smoother.max_bandwidth >= 5.0

    def test_init_custom_bandwidth(self):
        """Test AutoSmoother initialization with custom bandwidth."""
        smoother = AutoSmoother(
            init_bandwidth=2.0, min_bandwidth=0.5, max_bandwidth=4.0
        )
        assert smoother.min_bandwidth == 0.5
        assert smoother.max_bandwidth == 4.0

    def test_enable_disable(self):
        """Test enabling and disabling the auto-smoother."""
        smoother = AutoSmoother()
        assert smoother.is_enabled

        smoother.disable()
        assert not smoother.is_enabled

        smoother.enable()
        assert smoother.is_enabled

    def test_smooth_1d(self):
        """Test 1D trajectory smoothing."""
        smoother = AutoSmoother()

        # Create a noisy signal
        t = np.linspace(0, 10, 100)
        signal = np.sin(t) + 0.5 * np.random.randn(100)

        smoothed = smoother(torch.from_numpy(signal))

        assert smoothed.shape == signal.shape
        assert not torch.isnan(smoothed).any()

    def test_smooth_2d(self):
        """Test 2D trajectory smoothing."""
        smoother = AutoSmoother()

        # Create 2D noisy signal
        t = np.linspace(0, 10, 100)
        x = np.sin(t)
        y = np.cos(t) + 0.3 * np.random.randn(100)

        trajectory = np.column_stack([x, y])

        smoothed = smoother(torch.from_numpy(trajectory))

        assert smoothed.shape == trajectory.shape

    def test_smooth_3d(self):
        """Test 3D trajectory smoothing."""
        smoother = AutoSmoother()

        # Create 3D noisy signal
        t = np.linspace(0, 10, 100)
        x = np.sin(t)
        y = np.cos(t)
        z = np.sin(2 * t) + 0.2 * np.random.randn(100)

        trajectory = np.column_stack([x, y, z])

        smoothed = smoother(torch.from_numpy(trajectory))

        assert smoothed.shape == trajectory.shape

    def test_bandwidth_clamping(self):
        """Test that bandwidth is clamped to valid range."""
        smoother = AutoSmoother(
            init_bandwidth=10.0, min_bandwidth=1.0, max_bandwidth=5.0
        )

        smoothed = smoother(torch.from_numpy(np.random.randn(100)))

        bandwidth = smoother.get_bandwidth()
        assert 1.0 <= bandwidth <= 5.0

    def test_set_noise_level_high(self):
        """Test bandwidth adjustment for high noise."""
        smoother = AutoSmoother()
        smoother.set_noise_level(0.1)

        bandwidth = smoother.get_bandwidth()
        # Should be larger for high noise
        assert bandwidth >= 2.0

    def test_set_noise_level_low(self):
        """Test bandwidth adjustment for low noise."""
        smoother = AutoSmoother()
        smoother.set_noise_level(0.01)

        bandwidth = smoother.get_bandwidth()
        # Should be smaller for low noise
        assert bandwidth <= 1.0

    def test_set_from_noise_std(self):
        """Test setting bandwidth from noise standard deviation."""
        smoother = AutoSmoother()
        smoother.set_from_noise_std(0.05)

        bandwidth = smoother.get_bandwidth()
        assert 0.5 <= bandwidth <= 2.0

    def test_optimize_bandwidth(self):
        """Test bandwidth optimization."""
        smoother = AutoSmoother()

        # Create noisy signal with known noise level
        noise_std = 0.05
        signal = np.sin(np.linspace(0, 10, 200)) + noise_std * np.random.randn(200)

        bandwidth_before = smoother.get_bandwidth()
        optimized_bw = smoother.optimize_bandwidth(
            torch.from_numpy(signal), [1.0, 0.9, 0.8]
        )

        # Bandwidth should be optimized
        assert optimized_bw >= smoother.min_bandwidth
        assert optimized_bw <= smoother.max_bandwidth

    def test_get_bandwidth_info(self):
        """Test getting bandwidth information."""
        smoother = AutoSmoother(init_bandwidth=1.5)

        info = smoother.get_bandwidth_info()

        assert "bandwidth" in info
        assert "log_bandwidth" in info
        assert "optimization_step" in info
        assert "noise_level" in info
        assert info["bandwidth"] == 1.5

    def test_smooth_disabled(self):
        """Test that disabled smoother returns original trajectory."""
        smoother = AutoSmoother()
        smoother.disable()

        trajectory = torch.randn(100)
        result = smoother(trajectory)

        # Should return the original trajectory unchanged
        assert torch.allclose(result, trajectory)

    def test_noisy_trajectory_improves(self):
        """Test that smoothing improves signal quality."""
        smoother = AutoSmoother()

        # Create signal with moderate noise
        t = np.linspace(0, 10, 200)
        true_signal = np.sin(t)
        noisy_signal = true_signal + 0.05 * np.random.randn(200)

        smoothed = smoother(torch.from_numpy(noisy_signal))

        # Compute variance reduction
        noise_var = np.var(noisy_signal - true_signal)
        smoothed_var = np.var(smoothed.numpy() - true_signal)

        # Smoothing should reduce noise variance
        assert smoothed_var < noise_var


class TestTrajectorySmoother:
    def test_init_with_autosmoothing(self):
        """Test TrajectorySmoother initialization with auto-smoothing enabled."""
        smoother = TrajectorySmoother(enable_autosmoothing=True)
        assert smoother.smoother is not None
        assert smoother.enable_autosmoothing

    def test_init_without_autosmoothing(self):
        """Test TrajectorySmoother initialization without auto-smoothing."""
        smoother = TrajectorySmoother(enable_autosmoothing=False)
        assert smoother.smoother is None
        assert not smoother.enable_autosmoothing

    def test_smooth_disabled(self):
        """Test that disabled smoother returns original."""
        smoother = TrajectorySmoother(enable_autosmoothing=False)

        trajectory = torch.randn(100)
        result = smoother.smooth(trajectory)

        assert torch.allclose(result, trajectory)

    def test_smooth_enabled(self):
        """Test that enabled smoother applies smoothing."""
        smoother = TrajectorySmoother(enable_autosmoothing=True)

        trajectory = torch.randn(100)
        result = smoother.smooth(trajectory)

        # Should be smoothed (different from original)
        assert not torch.allclose(result, trajectory)

    def test_smoothed_trajectory_has_bandwidth(self):
        """Test that smooth trajectory has meaningful bandwidth."""
        smoother = TrajectorySmoother(enable_autosmoothing=True)

        smoother.optimize(torch.randn(100))
        bandwidth = smoother.get_bandwidth()

        assert bandwidth is not None
        assert bandwidth > 0

    def test_smoothed_trajectory_info(self):
        """Test getting info from enabled smoother."""
        smoother = TrajectorySmoother(enable_autosmoothing=True)

        info = smoother.get_info()

        assert "bandwidth" in info
        assert "is_enabled" in info
        assert info["is_enabled"]


class TestGaussianSmoother:
    def test_init_default_sigma(self):
        """Test GaussianSmoother initialization with default sigma."""
        smoother = GaussianSmoother()
        assert smoother.sigma == 1.0

    def test_init_custom_sigma(self):
        """Test GaussianSmoother initialization with custom sigma."""
        smoother = GaussianSmoother(sigma=2.0)
        assert smoother.sigma == 2.0

    def test_smooth_1d_fixed_sigma(self):
        """Test 1D smoothing with fixed sigma."""
        smoother = GaussianSmoother(sigma=1.5)

        signal = np.sin(np.linspace(0, 10, 100)) + 0.5 * np.random.randn(100)
        smoothed = smoother(torch.from_numpy(signal))

        assert smoothed.shape == signal.shape

    def test_smoothed_compared_to_autosmoother(self):
        """Compare GaussianSmoother with AutoSmoother."""
        gaussian = GaussianSmoother(sigma=1.0)
        auto = AutoSmoother(init_bandwidth=1.0)

        signal = np.sin(np.linspace(0, 10, 100)) + 0.3 * np.random.randn(100)

        result_gaussian = gaussian(torch.from_numpy(signal))
        result_auto = auto(torch.from_numpy(signal))

        # Both should produce similar smoothing results
        # (not identical, but conceptually similar)
        assert result_gaussian.shape == result_auto.shape


class TestLearnableAutoSmoother:
    def test_init_learnable(self):
        """Test LearnableAutoSmoother initialization."""
        smoother = LearnableAutoSmoother()
        assert hasattr(smoother, "log_bandwidth")
        assert smoother.log_bandwidth.requires_grad

    def test_smooth_learnable(self):
        """Test that LearnableAutoSmoother applies smoothing."""
        smoother = LearnableAutoSmoother()

        signal = np.sin(np.linspace(0, 10, 100)) + 0.3 * np.random.randn(100)
        result = smoother(torch.from_numpy(signal))

        assert result.shape == signal.shape

    def test_bandwidth_optimization(self):
        """Test bandwidth optimization with LearnableAutoSmoother."""
        smoother = LearnableAutoSmoother(init_bandwidth=1.0)

        signal = np.sin(np.linspace(0, 10, 200)) + 0.05 * np.random.randn(200)

        # Optimize bandwidth
        smoother.update_from_loss(1.0, 0.5)

        bandwidth = smoother.get_bandwidth()

        # Bandwidth should be optimized
        assert smoother.training_step > 0
        assert smoother.min_bandwidth <= bandwidth <= smoother.max_bandwidth

    def test_bandwidth_update_on_loss(self):
        """Test that bandwidth updates based on loss."""
        smoother = LearnableAutoSmoother()

        bandwidth_before = smoother.get_bandwidth()

        # Update with loss improvement
        smoother.update_from_loss(1.0, 0.8)

        bandwidth_after = smoother.get_bandwidth()

        # Should have updated
        assert smoother.training_step > 0
        assert bandwidth_after <= bandwidth_before  # Reduced smoothing

    def test_bandwidth_increase_on_worsening_loss(self):
        """Test that bandwidth increases when loss worsens."""
        smoother = LearnableAutoSmoother()

        bandwidth_before = smoother.get_bandwidth()

        # Update with loss worsening
        smoother.update_from_loss(0.5, 0.8)

        bandwidth_after = smoother.get_bandwidth()

        # Should have updated and increased
        assert smoother.training_step > 0
        assert bandwidth_after >= bandwidth_before  # Increased smoothing

    def test_get_learnable_bandwidth_info(self):
        """Test getting info from LearnableAutoSmoother."""
        smoother = LearnableAutoSmoother(init_bandwidth=1.5)

        info = smoother.get_info()

        assert "bandwidth" in info
        assert "log_bandwidth" in info
        assert "training_step" in info
        assert info["bandwidth"] == 1.5


class TestR2ScoreSimulation:
    def test_r2_score_calculation(self):
        """Test R2 score calculation in bandwidth optimization."""
        smoother = AutoSmoother()

        # Create signal with known noise
        true_signal = np.sin(np.linspace(0, 10, 200))
        noise_std = 0.05
        noisy_signal = true_signal + noise_std * np.random.randn(200)

        # Optimize bandwidth
        bandwidth = smoother.optimize_bandwidth(
            torch.from_numpy(noisy_signal), [1.0, 0.9, 0.8, 0.7, 0.6]
        )

        # Should get a reasonable bandwidth
        assert smoother.min_bandwidth <= bandwidth <= smoother.max_bandwidth
        assert bandwidth > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
