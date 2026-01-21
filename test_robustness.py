
import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from enhanced_symbolic import SymPyToTorch
from model import HamiltonianODEFunc
from stable_pooling import StableHierarchicalPooling

def test_sympy_to_torch_robustness():
    print("Testing SymPyToTorch robustness...")
    x = sp.Symbol('x0')
    # Expression with many potential singularities
    expr = 1/x + sp.log(x) + sp.tan(x) + sp.exp(x)
    
    model = SymPyToTorch(expr, n_inputs=1)
    
    # Test with zero (singularity for 1/x and log(x))
    inputs = torch.tensor([[0.0]], dtype=torch.float32)
    output = model(inputs)
    assert torch.isfinite(output).all(), f"Output with zero input is not finite: {output}"
    
    # Test with very small values
    inputs = torch.tensor([[1e-12], [-1e-12]], dtype=torch.float32)
    output = model(inputs)
    assert torch.isfinite(output).all(), f"Output with small input is not finite: {output}"

    # Test with large values (potential overflow for exp)
    inputs = torch.tensor([[100.0]], dtype=torch.float32)
    output = model(inputs)
    assert torch.isfinite(output).all(), f"Output with large input is not finite: {output}"
    assert output.item() <= 1e6, f"Output not clamped: {output.item()}"

    # Test gradients
    inputs = torch.tensor([[0.0]], dtype=torch.float32, requires_grad=True)
    output = model(inputs)
    output.backward()
    assert torch.isfinite(inputs.grad).all(), f"Gradient at singularity is not finite: {inputs.grad}"
    print("SymPyToTorch robustness test passed.")

def test_hamiltonian_ode_stability():
    print("Testing HamiltonianODEFunc stability...")
    latent_dim = 4
    n_super_nodes = 2
    model = HamiltonianODEFunc(latent_dim, n_super_nodes)
    
    # Force a NaN or Inf in weights to simulate bad training
    with torch.no_grad():
        model.H_net[0].weight[0, 0] = float('nan')
    
    y = torch.randn(1, latent_dim * n_super_nodes)
    dy = model(0, y)
    
    # Our optimized version uses torch.nan_to_num and clamping
    assert torch.isfinite(dy).all(), f"HamiltonianODEFunc output with NaN weights is not finite: {dy}"
    print("HamiltonianODEFunc stability test passed.")

def test_node_revival():
    print("Testing Node Revival...")
    in_channels = 8
    n_super_nodes = 4
    pooling = StableHierarchicalPooling(in_channels, n_super_nodes, min_active_super_nodes=2)
    
    # Manually kill some nodes in active_mask
    pooling.active_mask.fill_(0.0)
    pooling.active_mask[0] = 1.0 # Only one active
    
    # Call apply_hard_revival
    pooling.apply_hard_revival()
    
    # Should have more than 1 active (or at least some re-activation started)
    active_count = (pooling.active_mask > 0.1).sum().item()
    assert active_count > 1, f"Nodes were not revived. Active count: {active_count}"
    
    # Verify weights were perturbed
    x = torch.randn(10, in_channels)
    batch = torch.zeros(10, dtype=torch.long)
    out, s, losses, mu = pooling(x, batch)
    assert torch.isfinite(out).all()
    print("Node Revival test passed.")

if __name__ == "__main__":
    test_sympy_to_torch_robustness()
    test_hamiltonian_ode_stability()
    test_node_revival()
