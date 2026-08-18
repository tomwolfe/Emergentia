import torch
import pytest
from emergentia.models import TrajectoryScaler

def test_scaling_reversibility():
    scaler = TrajectoryScaler()
    p = torch.randn(100, 4, 2) * 5.0
    f = torch.randn(100, 4, 2) * 10.0
    
    scaler.fit(p, f)
    p_s, f_s = scaler.transform(p, f)
    
    # Check if scales are within [0, 1] approximately
    assert torch.max(torch.abs(p_s)) <= 1.0001
    assert torch.max(torch.abs(f_s)) <= 1.0001
    
    f_inv = scaler.inverse_transform_f(f_s)
    assert torch.allclose(f, f_inv, atol=1e-5)

def test_scaling_zero_motion():
    scaler = TrajectoryScaler()
    p = torch.zeros(10, 2, 2)
    f = torch.zeros(10, 2, 2)
    
    scaler.fit(p, f)
    p_s, f_s = scaler.transform(p, f)
    
    assert torch.all(p_s == 0)
    assert torch.all(f_s == 0)
    assert scaler.p_scale == 1e-8
    assert scaler.f_scale == 1e-8

def test_extreme_scales():
    scaler = TrajectoryScaler()
    p = torch.randn(10, 2, 2) * 1e-10
    f = torch.randn(10, 2, 2) * 1e10
    
    scaler.fit(p, f)
    p_s, f_s = scaler.transform(p, f)
    
    assert torch.max(torch.abs(p_s)) <= 1.0001
    assert torch.max(torch.abs(f_s)) <= 1.0001
    
    f_inv = scaler.inverse_transform_f(f_s)
    assert torch.allclose(f, f_inv)


def test_10_particles():
    """Test 10-particle LJ simulation with neighbor list."""
    from emergentia.simulator import PhysicsSim, LennardJonesPotential
    
    # Test with neighbor list
    sim = PhysicsSim(n=10, dim=2, potential=LennardJonesPotential(), seed=42, use_neighbor_list=True, cutoff=3.0)
    traj_p, traj_f = sim.generate(steps=500, noise_std=0.0)
    
    # Check energy conservation
    H_before = sim.get_hamiltonian()
    
    # Test without neighbor list for comparison
    sim2 = PhysicsSim(n=10, dim=2, potential=LennardJonesPotential(), seed=42, use_neighbor_list=False)
    traj_p2, traj_f2 = sim2.generate(steps=500, noise_std=0.0)
    H_before2 = sim2.get_hamiltonian()
    
    # Energies should be similar
    energy_diff = abs(H_before.item() - H_before2.item()) / abs(H_before2.item())
    assert energy_diff < 0.01, f'Energies differ by {energy_diff*100:.1f}%'
    
    # Verify trajectory shapes
    assert traj_p.shape == (500, 10, 2)
    assert traj_f.shape == (500, 10, 2)
    
    print(f'10-particle test passed! Energy: {H_before.item():.2f}')
