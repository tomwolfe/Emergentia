import torch
import pytest
import math
from emergentia.simulator import YukawaPotential
from emergentia import PhysicsSim, DiscoveryPipeline

def test_yukawa_force_magnitude_at_r1():
    """
    Test compute_force_magnitude at r=1.0.
    F(r) = A * exp(-B * r) * (B/r + 1/r^2)
    For A=1.0, B=1.0, r=1.0:
    F(1.0) = 1.0 * exp(-1.0) * (1.0/1.0 + 1.0/1.0^2) = exp(-1.0) * 2.0 ≈ 0.73575888
    """
    potential = YukawaPotential(A=1.0, B=1.0)
    dist = torch.tensor([1.0])
    force = potential.compute_force_magnitude(dist)
    
    expected = 1.0 * math.exp(-1.0) * (1.0/1.0 + 1.0/1.0**2)
    assert abs(force.item() - expected) < 1e-6, f"Expected {expected}, got {force.item()}"

def test_yukawa_potential_at_r1():
    """
    Test compute_potential at r=1.0.
    V(r) = A * exp(-B * r) / r
    For A=1.0, B=1.0, r=1.0:
    V(1.0) = 1.0 * exp(-1.0) / 1.0 = exp(-1.0) ≈ 0.36787944
    """
    potential = YukawaPotential(A=1.0, B=1.0)
    dist = torch.tensor([1.0])
    pot_val = potential.compute_potential(dist)
    
    expected = 1.0 * math.exp(-1.0) / 1.0
    assert abs(pot_val.item() - expected) < 1e-6, f"Expected {expected}, got {pot_val.item()}"

def test_yukawa_integration():
    """Integration test that runs a short DiscoveryPipeline on a Yukawa trajectory"""
    potential = YukawaPotential(A=1.0, B=1.0)
    
    # Create a simple simulation with Yukawa potential
    sim = PhysicsSim(n=3, dim=2, potential=potential, seed=42)
    
    # Generate a short trajectory
    traj_pos, traj_forces = sim.generate(steps=100, noise_std=0.0)
    
    # Create a DiscoveryPipeline with Yukawa-specific basis set
    pipeline = DiscoveryPipeline(
        mode='yukawa', 
        potential=potential, 
        device=sim.device, 
        seed=42, 
        basis_set=['1/r', '1/r^2', 'exp(-r)/r']
    )
    
    # Run a short discovery process (100 epochs as specified in mission)
    result = pipeline.run(sim, nn_epochs=100, noise_std=0.0)
    
    # Check that the result contains expected fields
    assert 'formula' in result
    assert 'mse' in result
    assert 'r2' in result
    assert 'success' in result
    assert result['formula'] is not None

if __name__ == "__main__":
    pytest.main([__file__])
