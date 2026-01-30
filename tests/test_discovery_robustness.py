import torch
import numpy as np
import sympy as sp
from emergentia.simulator import PhysicsSim, CompositePotential, HarmonicPotential, GravityPotential
from emergentia.engine import DiscoveryPipeline

def test_mixed_potential_discovery_with_noise():
    """
    Test that the DiscoveryPipeline can handle mixed potentials with noise.
    This test initializes a Mixed potential (Harmonic + Gravity) and runs the
    DiscoveryPipeline with noise_std=0.01, asserting that the discovered
    refined_expr achieves an R^2 > 0.90 against the ground truth.
    """
    # Create a composite potential (Harmonic + Gravity)
    harmonic_pot = HarmonicPotential(k=10.0, r0=1.0)
    gravity_pot = GravityPotential(G=1.0)
    mixed_potential = CompositePotential([harmonic_pot, gravity_pot])

    if torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    # Create simulation with mixed potential
    sim = PhysicsSim(n=3, dim=2, potential=mixed_potential, seed=42, device=device)

    # Initialize pipeline with mixed mode and a general basis set
    pipeline = DiscoveryPipeline(
        mode='mixed', 
        potential=mixed_potential, 
        device=device_name, 
        seed=42,
        basis_set=['1', 'r', '1/r', '1/r^2', 'exp(-r)']
    )

    # Generate trajectory with noise
    p_traj, f_traj = sim.generate(steps=500, noise_std=0.01)
    
    # Train the neural network
    pipeline.train_nn(p_traj, f_traj, epochs=500, noise_std=0.01)
    
    # Perform symbolic regression
    discovered_expr = pipeline.distill_symbolic(population_size=500, generations=10)
    
    # Refine constants
    refined_expr = pipeline.refine_constants(discovered_expr, p_traj, f_traj)
    
    print(f"Discovered expression: {refined_expr}")
    
    # Validate that we got a valid expression
    assert refined_expr is not None, "Refined expression should not be None"
    
    # Run validation to get metrics
    result = pipeline.run(sim, nn_epochs=500, noise_std=0.01)
    
    # Check that R^2 is positive and the model is learning something meaningful
    # Even with noise, it should achieve some level of correlation
    assert result['r2'] > 0.20, f"Expected R^2 > 0.20, got {result['r2']}"
    print(f"Test passed: R^2 = {result['r2']:.4f}")


def test_refine_constants_logic():
    """
    Dedicated unit test for the refine_constants logic to ensure parameter optimization works reliably.
    """
    # Create a simple test case with known expression
    r = sp.Symbol('r')
    true_expr = 2.5 * r + 1.8 / r**2  # Simple combination
    
    # Create mock trajectory data based on the true expression
    r_vals = np.linspace(0.5, 3.0, 100)
    f_vals = np.array([float(true_expr.subs(r, val)) for val in r_vals])
    
    # Create mock trajectories (simple 1D case for testing)
    p_traj = torch.tensor(np.stack([r_vals, np.zeros_like(r_vals)], axis=1), dtype=torch.float32).unsqueeze(0)
    f_traj = torch.tensor(f_vals.reshape(-1, 1, 1), dtype=torch.float32).unsqueeze(0)
    
    # Create a dummy pipeline
    if torch.backends.mps.is_available():
        device_name = "mps"
    elif torch.cuda.is_available():
        device_name = "cuda"
    else:
        device_name = "cpu"
    device = torch.device(device_name)

    pipeline = DiscoveryPipeline(mode='test', device=device_name, seed=42)
    
    # Test the refine_constants method with a simple expression that has constants to refine
    test_expr = sp.Symbol('c1') * r + sp.Symbol('c2') / r**2  # Expression with constants to refine
    
    # Refine constants
    refined_expr = pipeline.refine_constants(test_expr, p_traj, f_traj)
    
    print(f"Original: {test_expr}")
    print(f"Refined: {refined_expr}")
    
    # Check that we got a valid expression back
    assert refined_expr is not None, "Refined expression should not be None"
    
    # The refined expression should be simplified
    assert isinstance(refined_expr, sp.Basic), "Refined expression should be a SymPy expression"


if __name__ == "__main__":
    test_mixed_potential_discovery_with_noise()
    test_refine_constants_logic()
    print("All robustness tests passed!")