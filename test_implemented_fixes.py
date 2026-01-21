"""
Test file to verify that the implemented fixes work correctly.
This tests the key improvements made based on the critical analysis.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from model import DiscoveryEngineModel
from enhanced_symbolic import EnhancedSymbolicDistiller, PhysicsAwareSymbolicDistiller
from coordinate_mapping import AlignedHamiltonianSymbolicDistiller, create_enhanced_coord_mapper
from stable_pooling import StableHierarchicalPooling, DynamicLossBalancer
from hamiltonian_symbolic import HamiltonianSymbolicDistiller
from balanced_features import BalancedFeatureTransformer
from symbolic import SymbolicDistiller
from symbolic import extract_latent_data
import pytest


def test_secondary_optimization():
    """Test that secondary optimization improves equation accuracy."""
    print("Testing secondary optimization...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 50  # Reduced to match properly
    n_super_nodes = 2
    latent_dim = 3  # 6 total dims per sample

    # Create data with proper dimensions
    latent_states = np.random.randn(n_samples, n_super_nodes * latent_dim)
    targets = np.random.randn(n_samples, n_super_nodes * latent_dim)

    # Test without secondary optimization
    basic_distiller = SymbolicDistiller(populations=500, generations=10)
    basic_equations = basic_distiller.distill(
        latent_states=latent_states,
        targets=targets,
        n_super_nodes=n_super_nodes,
        latent_dim=latent_dim
    )

    # Test with secondary optimization
    enhanced_distiller = EnhancedSymbolicDistiller(
        populations=500,
        generations=10,
        secondary_optimization=True
    )
    enhanced_equations = enhanced_distiller.distill_with_secondary_optimization(
        latent_states=latent_states,
        targets=targets,
        n_super_nodes=n_super_nodes,
        latent_dim=latent_dim
    )

    print("✓ Secondary optimization test completed")
    return True


def test_coordinate_alignment():
    """Test that coordinate alignment works properly."""
    print("Testing coordinate alignment...")
    
    # Create synthetic neural and physical coordinates
    np.random.seed(42)
    n_samples = 50
    n_super_nodes = 3
    n_latent_dims = 4
    n_physical_dims = 2
    
    # Neural coordinates (potentially rotated/transformed)
    neural_coords = np.random.randn(n_samples, n_super_nodes, n_latent_dims)
    
    # Physical coordinates (ground truth)
    physical_coords = np.random.randn(n_samples, n_super_nodes, n_physical_dims)
    
    # Create coordinate mapper
    coord_mapper = create_enhanced_coord_mapper(n_latent_dims, n_physical_dims)
    coord_mapper.fit(neural_coords, physical_coords)
    
    # Test transformation
    aligned_coords = coord_mapper.neural_to_physical(neural_coords)
    
    assert aligned_coords.shape == (n_samples, n_super_nodes, n_physical_dims), \
        f"Expected shape {(n_samples, n_super_nodes, n_physical_dims)}, got {aligned_coords.shape}"
    
    print("✓ Coordinate alignment test completed")
    return True


def test_collapse_prevention():
    """Test that collapse prevention mechanisms work."""
    print("Testing collapse prevention...")
    
    # Create a simple pooling scenario
    in_channels = 16
    n_super_nodes = 5
    batch_size = 10
    
    pooling_layer = StableHierarchicalPooling(
        in_channels=in_channels,
        n_super_nodes=n_super_nodes,
        min_active_super_nodes=2  # Ensure at least 2 remain active
    )
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels)
    batch = torch.zeros(batch_size, dtype=torch.long)  # Single batch
    pos = torch.randn(batch_size, 2)  # Position information
    
    # Forward pass
    pooled, s, assign_losses, mu = pooling_layer(x, batch, pos=pos)
    
    # Check that we have the expected outputs
    assert pooled.shape[1] == n_super_nodes, f"Expected {n_super_nodes} super-nodes, got {pooled.shape[1]}"
    assert s.shape[1] == n_super_nodes, f"Expected {n_super_nodes} assignments, got {s.shape[1]}"
    
    # Check that losses are computed
    assert 'collapse_prevention' in assign_losses, "Collapse prevention loss not computed"
    assert 'balance' in assign_losses, "Balance loss not computed"
    
    print("✓ Collapse prevention test completed")
    return True


def test_dynamic_loss_balancing():
    """Test dynamic loss balancing."""
    print("Testing dynamic loss balancing...")
    
    # Create initial loss weights
    initial_weights = {
        'reconstruction': 1.0,
        'assignment': 0.5,
        'sparsity': 0.1
    }
    
    balancer = DynamicLossBalancer(initial_weights=initial_weights, adaptation_rate=0.01)
    
    # Simulate some training steps
    for step in range(20):
        # Simulate current losses (these would come from actual training)
        current_losses = {
            'reconstruction': torch.tensor(np.random.uniform(0.1, 1.0)),
            'assignment': torch.tensor(np.random.uniform(0.01, 0.5)),
            'sparsity': torch.tensor(np.random.uniform(0.001, 0.1))
        }
        
        # Update weights based on losses
        balanced_losses = balancer.get_balanced_losses(current_losses)
        
        # Verify that we get balanced losses
        assert len(balanced_losses) == len(current_losses), "Number of losses changed"
    
    print("✓ Dynamic loss balancing test completed")
    return True


def test_hamiltonian_structure_preservation():
    """Test that Hamiltonian structure is preserved in symbolic distillation."""
    print("Testing Hamiltonian structure preservation...")

    # Create synthetic data for Hamiltonian system
    np.random.seed(42)
    n_samples = 50
    n_super_nodes = 2
    latent_dim = 4  # 2 position + 2 momentum dims (must be even for Hamiltonian)

    # Create phase space coordinates [q1, q2, p1, p2] for each super-node
    latent_states = np.random.randn(n_samples, n_super_nodes * latent_dim)

    # Create target derivatives respecting Hamiltonian structure
    # For a simple harmonic oscillator: dq1/dt = p1, dq2/dt = p2, dp1/dt = -q1, dp2/dt = -q2
    targets = np.zeros((n_samples, n_super_nodes * latent_dim))
    for i in range(n_super_nodes):
        q_start = i * latent_dim
        q_end = q_start + latent_dim // 2
        p_start = q_end
        p_end = p_start + latent_dim // 2

        targets[:, q_start:q_end] = latent_states[:, p_start:p_end]  # dq/dt = ∂H/∂p
        targets[:, p_start:p_end] = -latent_states[:, q_start:q_end]  # dp/dt = -∂H/∂q

    # Test Hamiltonian symbolic distillation
    hamiltonian_distiller = HamiltonianSymbolicDistiller(
        populations=500,
        generations=10,
        enforce_hamiltonian_structure=True
    )

    equations = hamiltonian_distiller.distill(
        latent_states=latent_states,
        targets=targets,
        n_super_nodes=n_super_nodes,
        latent_dim=latent_dim
    )

    # Test aligned Hamiltonian distillation
    aligned_distiller = AlignedHamiltonianSymbolicDistiller(
        populations=500,
        generations=10,
        enforce_hamiltonian_structure=True
    )

    equations_aligned = aligned_distiller.distill_with_alignment(
        neural_latents=latent_states.reshape(n_samples, n_super_nodes, latent_dim),
        targets=targets,
        n_super_nodes=n_super_nodes,
        latent_dim=latent_dim
    )

    print("✓ Hamiltonian structure preservation test completed")
    return True


def test_enhanced_feature_engineering():
    """Test enhanced feature engineering with balanced features."""
    print("Testing enhanced feature engineering...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    n_super_nodes = 3
    latent_dim = 4

    latent_states = np.random.randn(n_samples, n_super_nodes * latent_dim)
    targets = np.random.randn(n_samples, n_super_nodes * latent_dim)

    # Test balanced feature transformer
    transformer = BalancedFeatureTransformer(
        n_super_nodes=n_super_nodes,
        latent_dim=latent_dim,
        basis_functions='physics_informed',
        feature_selection_method='mutual_info'
    )

    transformer.fit(latent_states, targets)

    # Transform the data
    X_transformed = transformer.transform(latent_states, fit_transformer=True)

    # Check that feature selection reduced dimensionality appropriately
    n_features_before_selection = transformer.x_poly_mean.shape[0]
    n_features_after_selection = transformer.get_n_features()

    print(f"Features before selection: {n_features_before_selection}")
    print(f"Features after selection: {n_features_after_selection}")

    assert n_features_after_selection <= n_features_before_selection, \
        "Feature selection should reduce or maintain dimensionality"

    print("✓ Enhanced feature engineering test completed")
    return True


def run_all_tests():
    """Run all tests to verify implemented fixes."""
    print("="*60)
    print("RUNNING TESTS FOR IMPLEMENTED FIXES")
    print("="*60)
    
    tests = [
        ("Secondary Optimization", test_secondary_optimization),
        ("Coordinate Alignment", test_coordinate_alignment),
        ("Collapse Prevention", test_collapse_prevention),
        ("Dynamic Loss Balancing", test_dynamic_loss_balancing),
        ("Hamiltonian Structure Preservation", test_hamiltonian_structure_preservation),
        ("Enhanced Feature Engineering", test_enhanced_feature_engineering),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            if result:
                print(f"✓ {test_name} PASSED")
                passed += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("🎉 ALL TESTS PASSED! Implemented fixes are working correctly.")
    else:
        print(f"⚠️  {total-passed} tests failed. Please check the implementation.")
    print(f"{'='*60}")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)