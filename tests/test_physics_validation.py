import torch
import numpy as np
import pytest
from model import EquivariantGNNLayer, HamiltonianODEFunc
from balanced_features import BalancedFeatureTransformer

def test_equivariant_gnn_rotational_invariance():
    """
    Verify that EquivariantGNNLayer is rotationally invariant for scalar features.
    """
    torch.manual_seed(42)
    in_channels = 4
    out_channels = 8
    hidden_dim = 16
    layer = EquivariantGNNLayer(in_channels, out_channels, hidden_dim)
    
    n_nodes = 5
    x = torch.randn(n_nodes, in_channels)
    pos = torch.randn(n_nodes, 2)
    vel = torch.randn(n_nodes, 2)
    
    # Fully connected graph
    edge_index = torch.tensor([[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j]).t()
    
    # Original output
    out1 = layer(x, pos, vel, edge_index)
    
    # Rotate by 90 degrees
    theta = torch.tensor(np.pi / 2)
    rot_mat = torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta), torch.cos(theta)]
    ])
    
    pos_rot = pos @ rot_mat.t()
    vel_rot = vel @ rot_mat.t()
    
    # Rotated output
    out2 = layer(x, pos_rot, vel_rot, edge_index)
    
    # Check if outputs are the same (within tolerance)
    assert torch.allclose(out1, out2, atol=1e-5), f"Rotational invariance failed. Max diff: {torch.max(torch.abs(out1 - out2))}"

def test_hamiltonian_symplectic_condition():
    """
    Verify that HamiltonianODEFunc satisfies the symplectic condition.
    M^T J + J M = 0 where M is the Jacobian of the dynamics.
    """
    latent_dim = 4 # (q1, q2, p1, p2)
    n_super_nodes = 1
    hidden_dim = 32
    # Non-separable to be more general
    model = HamiltonianODEFunc(latent_dim, n_super_nodes, hidden_dim, dissipative=False, separable=False)
    
    y = torch.randn(1, latent_dim * n_super_nodes, requires_grad=True)
    t = torch.tensor(0.0)
    
    # Dynamics f(y) = dy/dt
    f = model(t, y)
    
    # Compute Jacobian M = df/dy
    M = torch.zeros(latent_dim, latent_dim)
    for i in range(latent_dim):
        grads = torch.autograd.grad(f[0, i], y, create_graph=True)[0]
        M[i] = grads[0]
    
    # Symplectic matrix J
    J = torch.zeros(latent_dim, latent_dim)
    half = latent_dim // 2
    J[:half, half:] = torch.eye(half)
    J[half:, :half] = -torch.eye(half)
    
    # Symplectic condition: M^T J + J M = 0
    res = M.t() @ J + J @ M
    
    assert torch.allclose(res, torch.zeros_like(res), atol=1e-4), f"Symplectic condition failed. Max value: {torch.max(torch.abs(res))}"

def test_pbc_distance_calculation():
    """
    Verify that BalancedFeatureTransformer correctly handles periodic boundary conditions.
    """
    n_super_nodes = 2
    latent_dim = 4
    box_size = (10.0, 10.0)
    
    transformer = BalancedFeatureTransformer(
        n_super_nodes=n_super_nodes,
        latent_dim=latent_dim,
        box_size=box_size,
        include_dists=True
    )
    
    # Particles at (1, 1) and (9, 9)
    # Direct distance: sqrt(8^2 + 8^2) = sqrt(128) approx 11.3
    # PBC distance: sqrt((-2)^2 + (-2)^2) = sqrt(8) approx 2.82
    
    z_nodes = np.zeros((1, n_super_nodes, latent_dim))
    z_nodes[0, 0, :2] = [1.0, 1.0]
    z_nodes[0, 1, :2] = [9.0, 9.0]
    
    dist_features = transformer._compute_distance_features(z_nodes)
    # dist_features[0] is sum_d
    
    expected_dist = np.sqrt(2.0**2 + 2.0**2)
    actual_dist = dist_features[0][0, 0]
    
    assert np.allclose(actual_dist, expected_dist, atol=1e-5), f"PBC distance failed. Expected {expected_dist}, got {actual_dist}"

if __name__ == "__main__":
    pytest.main([__file__])
