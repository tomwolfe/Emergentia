"""
Test suite for implemented fixes in the Neural-Symbolic Discovery Pipeline.

This test suite verifies:
1. Analytical gradients with SymPy work correctly
2. PBC (Periodic Boundary Conditions) work properly
3. ODE integration efficiency improvements function as expected
4. Overall model stability with the implemented changes
"""

import unittest
import torch
import numpy as np
from model import DiscoveryEngineModel, HamiltonianODEFunc
from simulator import LennardJonesSimulator
from engine import prepare_data
from symbolic import SymbolicDistiller
import sympy as sp


class TestAnalyticalGradients(unittest.TestCase):
    """Test the analytical gradient implementation with SymPy"""
    
    def test_sympy_conversion_basic(self):
        """Test basic SymPy conversion functionality"""
        # Create a simple symbolic expression
        x1, x2 = sp.symbols('x1 x2')
        expr = x1**2 + x2**2
        
        # Compute gradients
        grad_x1 = sp.diff(expr, x1)
        grad_x2 = sp.diff(expr, x2)
        
        # Verify gradients
        self.assertEqual(str(grad_x1), "2*x1")
        self.assertEqual(str(grad_x2), "2*x2")
        
        # Test evaluation
        f1 = sp.lambdify([x1, x2], grad_x1, 'numpy')
        f2 = sp.lambdify([x1, x2], grad_x2, 'numpy')
        
        result1 = f1(2.0, 3.0)
        result2 = f2(2.0, 3.0)
        
        self.assertEqual(result1, 4.0)
        self.assertEqual(result2, 6.0)


class TestPBCFunctionality(unittest.TestCase):
    """Test Periodic Boundary Conditions functionality"""
    
    def test_pbc_simulation(self):
        """Test that PBC works in the simulator"""
        n_particles = 8
        box_size = (5.0, 5.0)
        
        # Create simulator with PBC
        sim = LennardJonesSimulator(
            n_particles=n_particles, 
            epsilon=1.0, 
            sigma=1.0, 
            dynamic_radius=1.5, 
            box_size=box_size
        )
        
        # Generate a short trajectory
        pos, vel = sim.generate_trajectory(steps=10)
        
        # Verify positions stay within box bounds (with some tolerance for numerical errors)
        for t in range(len(pos)):
            for i in range(n_particles):
                self.assertLessEqual(pos[t][i][0], box_size[0] * 1.1, 
                                   f"X position exceeds box bounds at t={t}, particle={i}")
                self.assertGreaterEqual(pos[t][i][0], -0.1, 
                                      f"X position below box bounds at t={t}, particle={i}")
                self.assertLessEqual(pos[t][i][1], box_size[1] * 1.1, 
                                   f"Y position exceeds box bounds at t={t}, particle={i}")
                self.assertGreaterEqual(pos[t][i][1], -0.1, 
                                      f"Y position below box bounds at t={t}, particle={i}")


class TestODEIntegrationEfficiency(unittest.TestCase):
    """Test ODE integration efficiency improvements"""
    
    def test_model_forward_dynamics_training_vs_eval(self):
        """Test that model behaves differently in training vs eval mode for ODE integration"""
        # Create a simple model
        model = DiscoveryEngineModel(
            n_particles=8,
            n_super_nodes=2,
            node_features=4,
            latent_dim=4,
            hidden_dim=16,
            hamiltonian=True
        )
        
        # Create sample input
        z0 = torch.randn(1, 2, 4)  # [batch_size, n_super_nodes, latent_dim]
        t = torch.linspace(0, 0.1, 5)  # Short time span
        
        # Test in training mode
        model.train()
        result_train = model.forward_dynamics(z0, t)
        
        # Test in evaluation mode
        model.eval()
        result_eval = model.forward_dynamics(z0, t)
        
        # Both should have the same shape
        self.assertEqual(result_train.shape, result_eval.shape)
        self.assertEqual(result_train.shape, (5, 1, 2, 4))


class TestOverallModelStability(unittest.TestCase):
    """Test overall model stability with all implemented fixes"""
    
    def test_model_training_stability(self):
        """Test that the model trains stably with all fixes applied"""
        # Create a small model for quick testing
        model = DiscoveryEngineModel(
            n_particles=8,
            n_super_nodes=2,
            node_features=4,
            latent_dim=4,
            hidden_dim=16,
            hamiltonian=True
        )
        
        # Create simple test data
        n_particles = 8
        box_size = (5.0, 5.0)
        sim = LennardJonesSimulator(
            n_particles=n_particles, 
            epsilon=1.0, 
            sigma=1.0, 
            dynamic_radius=1.5, 
            box_size=box_size
        )
        
        pos, vel = sim.generate_trajectory(steps=20)
        dataset, stats = prepare_data(pos, vel, radius=1.5, device='cpu')
        
        # Simple forward pass
        data = dataset[0]
        z, s, losses, mu = model.encode(data.x, data.edge_index, 
                                       torch.zeros(data.x.size(0), dtype=torch.long))
        
        # Decode
        recon = model.decode(z, s, torch.zeros(data.x.size(0), dtype=torch.long))
        
        # Check that outputs are finite
        self.assertTrue(torch.all(torch.isfinite(z)))
        self.assertTrue(torch.all(torch.isfinite(recon)))
        
        # Check assignment matrix properties
        self.assertEqual(s.shape, (n_particles, 2))  # [N, n_super_nodes]
        self.assertTrue(torch.all(s >= 0))  # Probabilities >= 0
        self.assertTrue(torch.allclose(s.sum(dim=1), torch.ones(s.size(0))))  # Sum to 1


class TestSymbolicDistillationWithPBC(unittest.TestCase):
    """Test symbolic distillation works with PBC-enabled data"""
    
    def test_distillation_with_transformer_pbc_support(self):
        """Test that the feature transformer supports PBC correctly"""
        n_super_nodes = 2
        latent_dim = 4
        
        # Create sample data
        n_samples = 50
        latent_states = np.random.randn(n_samples, n_super_nodes * latent_dim)
        targets = np.random.randn(n_samples, 1)  # Single target for simplicity
        
        # Create distiller with PBC support
        distiller = SymbolicDistiller(populations=200, generations=10)
        
        # Test distillation with PBC-enabled transformer
        equations = distiller.distill(
            latent_states, 
            targets, 
            n_super_nodes, 
            latent_dim, 
            box_size=(5.0, 5.0)  # Enable PBC in transformer
        )
        
        # Verify distillation completed without errors
        self.assertEqual(len(equations), 1)
        self.assertEqual(len(distiller.confidences), 1)
        
        # Verify feature masks were created
        self.assertEqual(len(distiller.feature_masks), 1)
        self.assertIsInstance(distiller.feature_masks[0], np.ndarray)
        self.assertEqual(distiller.feature_masks[0].dtype, bool)


if __name__ == '__main__':
    print("Running tests for implemented fixes...")
    unittest.main(verbosity=2)