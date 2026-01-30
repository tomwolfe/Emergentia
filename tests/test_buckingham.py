import torch
import pytest
import math
from emergentia.simulator import BuckinghamPotential
from emergentia import PhysicsSim, DiscoveryPipeline


def test_buckingham_force_magnitude_at_r1():
    """Test that compute_force_magnitude at r=1.0 matches the manual calculation:
    10.0 * 1.0 * exp(-1.0) - 6.0 * 1.0 * (1.0)^(-7) ≈ 3.6788 - 6.0 = -2.3212
    """
    potential = BuckinghamPotential(A=10.0, B=1.0, C=1.0)
    dist = torch.tensor([1.0])
    force = potential.compute_force_magnitude(dist)
    
    expected = 10.0 * 1.0 * math.exp(-1.0) - 6.0 * 1.0 * (1.0)**(-7)
    assert abs(force.item() - expected) < 1e-4, f"Expected {expected}, got {force.item()}"


def test_buckingham_potential_calculation():
    """Test that compute_potential calculates correctly"""
    potential = BuckinghamPotential(A=10.0, B=1.0, C=1.0)
    dist = torch.tensor([1.0])
    pot_val = potential.compute_potential(dist)
    
    expected = 10.0 * math.exp(-1.0) - 1.0 * (1.0)**(-6)
    assert abs(pot_val.item() - expected) < 1e-4, f"Expected {expected}, got {pot_val.item()}"


def test_buckingham_force_magnitude_clamping():
    """Test that distances are properly clamped to prevent singularities"""
    potential = BuckinghamPotential(A=10.0, B=1.0, C=1.0)
    # Test with a very small distance that should be clamped to 0.5
    dist = torch.tensor([0.1])
    force = potential.compute_force_magnitude(dist)
    
    # Calculate expected force with clamped distance of 0.5
    expected = 10.0 * 1.0 * math.exp(-1.0 * 0.5) - 6.0 * 1.0 * (0.5)**(-7)
    assert abs(force.item() - expected) < 1e-4, f"Expected {expected}, got {force.item()}"


def test_buckingham_integration_with_discovery_pipeline():
    """Integration test that runs a mini DiscoveryPipeline on a Buckingham trajectory"""
    potential = BuckinghamPotential(A=5.0, B=1.5, C=2.0)  # Different values for variety
    
    # Create a simple simulation with Buckingham potential
    sim = PhysicsSim(n=3, dim=2, potential=potential, seed=42)
    
    # Generate a short trajectory
    traj_pos, traj_forces = sim.generate(steps=100, noise_std=0.0)
    
    # Create a DiscoveryPipeline with Buckingham-specific basis set
    pipeline = DiscoveryPipeline(
        mode='buckingham', 
        potential=potential, 
        device=sim.device, 
        seed=42, 
        basis_set=['1', '1/r^7', 'exp(-r)']
    )
    
    # Run a short discovery process (500 epochs as specified)
    result = pipeline.run(sim, nn_epochs=500, noise_std=0.0)
    
    # Check that the result contains expected fields
    assert 'formula' in result
    assert 'mse' in result
    assert 'r2' in result
    assert 'success' in result
    
    # The refined expression should not be None if the pipeline ran successfully
    # Note: We're not asserting success=True since 500 epochs might not be enough for full convergence
    # But the pipeline should at least run without errors
    assert result['formula'] is not None


if __name__ == "__main__":
    pytest.main([__file__])