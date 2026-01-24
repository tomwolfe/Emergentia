import torch
import numpy as np
import sympy as sp
import pytest
from balanced_features import BalancedFeatureTransformer
from enhanced_symbolic import SymPyToTorch

def test_sympy_to_torch_lj():
    """Verify SymPyToTorch handles LJ-style power laws correctly."""
    x0 = sp.Symbol('x0')
    # LJ potential like: 4*( (1/x0)^12 - (1/x0)^6 )
    expr = 4 * ( (1/x0)**12 - (1/x0)**6 )
    
    n_inputs = 1
    model = SymPyToTorch(expr, n_inputs)
    
    # Test values
    test_vals = torch.tensor([[1.0], [1.122], [2.0]]) # 1.122 is approx 2^(1/6)
    outputs = model(test_vals)
    
    # Expected at 1.0: 4*(1 - 1) = 0
    assert torch.abs(outputs[0]) < 1e-5
    
    # Expected at 1.122: approx -1.0 (minimum)
    assert outputs[1] < 0
    assert torch.abs(outputs[1] - (-1.0)) < 1e-2

def test_balanced_feature_transformer_jacobian():
    """
    Verify BalancedFeatureTransformer jacobians match numerical gradients.
    """
    n_super_nodes = 2
    latent_dim = 4
    sim_type = 'lj'
    
    transformer = BalancedFeatureTransformer(
        n_super_nodes=n_super_nodes,
        latent_dim=latent_dim,
        sim_type=sim_type,
        basis_functions='physics_informed'
    )
    
    # Mock fit
    z_mock = np.random.randn(10, n_super_nodes * latent_dim)
    targets_mock = np.random.randn(10, 1)
    transformer.fit(z_mock, targets_mock)
    
    # Test point
    z_test = np.random.randn(n_super_nodes * latent_dim)
    
    # Analytical Jacobian
    jac_analytical = transformer.transform_jacobian(z_test)
    
    # Numerical Jacobian via finite differences
    eps = 1e-6
    n_latents = n_super_nodes * latent_dim
    
    # Get one transform to see how many features it actually returns
    X_sample = transformer.transform(z_test.reshape(1, -1))
    # During transform, it might be using selected features if they were fitted
    
    jac_numerical_full = np.zeros((X_sample.shape[1], n_latents))
    
    for i in range(n_latents):
        z_plus = z_test.copy()
        z_plus[i] += eps
        z_minus = z_test.copy()
        z_minus[i] -= eps
        
        X_plus = transformer.transform(z_plus.reshape(1, -1))
        X_minus = transformer.transform(z_minus.reshape(1, -1))
        
        jac_numerical_full[:, i] = (X_plus[0] - X_minus[0]) / (2 * eps)

    # Check match for distance-based features
    # Analytical Jacobian should match the selected features
    n_features = jac_analytical.shape[0]
    
    for f_idx in range(min(5, n_features)):
        ana = jac_analytical[f_idx]
        num = jac_numerical_full[f_idx] # Assuming they align in order
        
        # Check if they are non-zero
        norm_num = np.linalg.norm(num)
        norm_ana = np.linalg.norm(ana)
        if norm_num > 1e-9 and norm_ana > 1e-9:
            correlation = np.dot(ana, num) / (norm_ana * norm_num + 1e-9)
            print(f"Feature {f_idx} correlation: {correlation}")
            # Relaxed slightly for finite difference approximations of complex terms
            assert correlation > 0.95, f"Jacobian mismatch for feature {f_idx}: correlation={correlation}"

def test_inverse_square_law():
    """Verify inverse square law handling."""
    x0 = sp.Symbol('x0')
    expr = 1 / x0**2
    model = SymPyToTorch(expr, 1)
    
    test_vals = torch.tensor([[0.5], [1.0], [2.0]])
    outputs = model(test_vals)
    
    expected = torch.tensor([4.0, 1.0, 0.25])
    assert torch.allclose(outputs.flatten(), expected, atol=1e-5)

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
