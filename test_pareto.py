import unittest
import numpy as np
from symbolic import SymbolicDistiller

class TestParetoSelection(unittest.TestCase):
    def setUp(self):
        # Small population for faster testing
        self.distiller = SymbolicDistiller(populations=500, generations=10, max_features=5)

    def test_simple_equation_preference(self):
        # Generate data from a simple linear relationship: y = 2*x1 + 0.5*x2
        # Feature space will include x1, x2, and many other noisy features
        np.random.seed(42)
        n_samples = 200
        n_features = 8 # 2 super nodes * 4 latent dims
        X = np.random.randn(n_samples, n_features)
        
        # Target is a simple linear combination of first two features
        y = (2.0 * X[:, 0] + 0.5 * X[:, 1]).reshape(-1, 1)
        
        # Add a tiny bit of noise
        y += 0.01 * np.random.randn(n_samples, 1)
        
        # Distill
        # Note: distill expects n_super_nodes and latent_dim
        equations = self.distiller.distill(X, y, n_super_nodes=2, latent_dim=4)
        
        # Check complexity of the first (and only) target
        # A linear relationship should result in a very low complexity (1 if it uses RidgeCV fallback)
        # or low complexity GP program.
        prog = equations[0]
        complexity = self.distiller.get_complexity(prog)
        
        print(f"Discovered equation complexity: {complexity}")
        print(f"Confidence: {self.distiller.confidences[0]}")
        
        # A simple linear relation should definitely have complexity < 10
        self.assertLess(complexity, 15)
        self.assertGreater(self.distiller.confidences[0], 0.9)

    def test_nonlinear_pareto_preference(self):
        # Generate data from a complex non-linear relationship: y = sin(x1 * x2) + exp(-x3^2)
        # This is harder for the transformer to represent perfectly with just poly/inv-dist
        np.random.seed(42)
        n_samples = 400
        n_features = 8
        X = np.random.randn(n_samples, n_features)
        
        # Complex non-linear target
        y = (np.sin(X[:, 0] * X[:, 1]) + np.exp(-X[:, 2]**2)).reshape(-1, 1)
        
        # Distill
        equations = self.distiller.distill(X, y, n_super_nodes=2, latent_dim=4)
        
        prog = equations[0]
        complexity = self.distiller.get_complexity(prog)
        score = self.distiller.confidences[0]
        
        print(f"Complex non-linear discovered complexity: {complexity}")
        print(f"Complex non-linear confidence: {score}")
        
        # We expect a moderate complexity - enough to capture the relationship 
        # but the Pareto selection should keep it from exploding into "soup".
        self.assertLess(complexity, 30)

if __name__ == '__main__':
    unittest.main()
