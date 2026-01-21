import unittest
import numpy as np
from symbolic import FeatureTransformer

class TestFeatureTransformer(unittest.TestCase):
    def setUp(self):
        self.n_super_nodes = 2
        self.latent_dim = 4
        self.transformer = FeatureTransformer(self.n_super_nodes, self.latent_dim)
        
        # Mock data: 10 samples
        self.latent_states = np.random.randn(10, self.n_super_nodes * self.latent_dim)
        self.targets = np.random.randn(10, self.n_super_nodes * self.latent_dim)
        self.transformer.fit(self.latent_states, self.targets)

    def test_transform_shape(self):
        z = np.random.randn(5, self.n_super_nodes * self.latent_dim)
        X_poly = self.transformer.transform(z)
        
        # Expected features:
        # raw latents: 2 * 4 = 8
        # distance features: 6 (for 2 super nodes)
        # Total base: 8 + 6 = 14
        # Squares of latents: 8
        # Cross-terms within same node: 2 * (4 choose 2) = 2 * 6 = 12
        # Cross-node same-dimension terms: (2 choose 2) * 4 = 1 * 4 = 4
        # Total: 14 + 8 + 12 + 4 = 38
        self.assertEqual(X_poly.shape, (5, 38))

    def test_inverse_distance_physics(self):
        # Create two nodes at (0,0) and (1,0)
        # Latent dim = 4, so first 2 are positions
        z = np.zeros((1, 8))
        z[0, 0:2] = [0, 0] # Node 0 pos
        z[0, 4:6] = [1, 0] # Node 1 pos
        
        X_poly = self.transformer.transform(z)
        
        # Find index of distance feature (after raw latents)
        # raw: 0-7, dist: 8, inv_dist: 9, inv_sq_dist: 10, exp_dist: 11, screened: 12, log: 13
        dist = X_poly[0, 8]
        inv_dist = X_poly[0, 9]
        inv_sq_dist = X_poly[0, 10]
        
        self.assertAlmostEqual(dist, 1.0, places=5)
        self.assertAlmostEqual(inv_dist, 1.0 / (1.0 + 0.1), places=5)
        self.assertAlmostEqual(inv_sq_dist, 1.0 / (1.0**2 + 0.1), places=5)

    def test_normalization_consistency(self):
        z = np.random.randn(5, 8)
        X_poly = self.transformer.transform(z)
        X_norm = self.transformer.normalize_x(X_poly)
        
        # Check that normalized features have mean roughly 0 and std roughly 1 if it was fit on this
        self.transformer.fit(z, np.random.randn(5, 8))
        X_poly_fit = self.transformer.transform(z)
        X_norm_fit = self.transformer.normalize_x(X_poly_fit)
        
        # Only check non-constant features
        stds = np.std(X_norm_fit, axis=0)
        for i, std in enumerate(stds):
            if std > 1e-6:
                self.assertAlmostEqual(std, 1.0, places=1)

if __name__ == '__main__':
    unittest.main()
