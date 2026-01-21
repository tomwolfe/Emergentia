#!/usr/bin/env python
"""
Test script to verify the fixes for the symbolic dynamics evaluation issue.
"""

import numpy as np
from improved_symbolic import ImprovedSymbolicDynamics

# Create mock objects to simulate the issue
class MockEquation:
    def execute(self, X):
        # Simulate the issue where execute returns a single value instead of an array
        # This is what was causing the "tuple index out of range" error
        result = np.sum(X, axis=1)  # This should return an array with shape (n_samples,)
        return result

class MockTransformer:
    def transform(self, z):
        return z
    
    def normalize_x(self, X):
        return X

# Create mock distiller
class MockDistiller:
    def __init__(self):
        self.transformer = MockTransformer()

# Test the symbolic dynamics function
def test_symbolic_dynamics():
    print("Testing symbolic dynamics with fixed implementation...")
    
    # Create mock objects
    distiller = MockDistiller()
    equations = [MockEquation()]
    feature_masks = [None]
    
    # Create the dynamics function
    dyn_fn = ImprovedSymbolicDynamics(distiller, equations, feature_masks, is_hamiltonian=True, n_super_nodes=2, latent_dim=8)
    
    # Test with a sample input
    z0 = np.random.randn(16)  # 16-dimensional state vector
    print(f"Input shape: {z0.shape}")
    
    try:
        result = dyn_fn(None, z0)
        print(f"Output shape: {result.shape}")
        print(f"Output: {result}")
        print("SUCCESS: No errors occurred!")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_symbolic_dynamics()
    if success:
        print("\nThe fix appears to be working correctly!")
    else:
        print("\nThe fix did not resolve the issue.")