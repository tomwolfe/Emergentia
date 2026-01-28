import torch
import numpy as np
import sympy as sp
from emergentia.simulator import PhysicsSim, LennardJonesPotential
from emergentia.engine import DiscoveryPipeline

def test_hamiltonian_conservation():
    # Test 3D LJ system energy conservation
    n = 4
    dim = 3
    potential = LennardJonesPotential()
    sim = PhysicsSim(n=n, dim=dim, potential=potential, seed=42, device=torch.device("cpu"))
    
    steps = 500 # Short run to check conservation
    h_start = sim.get_hamiltonian().item()
    
    # To test pure conservation, we need to avoid the random impulses in generate()
    p_traj, f_traj = sim.generate(steps=400, noise_std=0.0, impulses=False)
    
    h_end = sim.get_hamiltonian().item()
    drift = abs(h_end - h_start) / abs(h_start)
    
    print(f"Energy drift over {400} steps with Velocity Verlet: {drift:.2e}")
    # Velocity Verlet should have MUCH better conservation than Euler
    # Euler with dt=0.001 usually has drift ~1e-2 to 1e-3 over 400 steps.
    # VV should be 1e-4 or better in float32.
    assert drift < 1e-4 

def test_symbolic_conservative():
    r = sp.Symbol('r')
    # Example discovered expression
    expr = 48.0 / r**13 - 24.0 / r**7
    
    # Check if it has an antiderivative (potential)
    potential = -sp.integrate(expr, r)
    assert potential is not None
    
    # Verify gradient
    force_back = -sp.diff(potential, r)
    assert sp.simplify(force_back - expr) == 0

def test_3d_discovery_flow():
    # Integration test for 3D discovery
    n = 3
    dim = 3
    potential = LennardJonesPotential()
    sim = PhysicsSim(n=n, dim=dim, potential=potential, seed=42, device=torch.device("cpu"))
    
    pipeline = DiscoveryPipeline(mode='lj', potential=potential, device='cpu', basis_set=['1/r^7', '1/r^13'])
    
    # Short run for testing
    p_traj, f_traj = sim.generate(steps=500)
    pipeline.train_nn(p_traj, f_traj, epochs=200)
    expr = pipeline.distill_symbolic(population_size=200, generations=5)
    
    assert expr is not None
    print(f"Discovered 3D formula: {expr}")
    
    # Refine
    refined = pipeline.refine_constants(expr, p_traj, f_traj)
    print(f"Refined 3D formula: {refined}")
    assert refined is not None
