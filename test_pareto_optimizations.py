
import torch
import numpy as np
from stable_pooling import StableHierarchicalPooling, SparsityScheduler
from balanced_features import BalancedFeatureTransformer, RecursiveFeatureSelector
from optimized_symbolic import OptimizedSymbolicDynamics
import sympy as sp

def test_sparsity_scheduler():
    print("Testing SparsityScheduler...")
    scheduler = SparsityScheduler(initial_weight=0.0, target_weight=1.0, warmup_steps=10, max_steps=100)
    
    # Warmup
    for _ in range(5):
        weight = scheduler.step()
        assert weight == 0.0
        
    # In progress
    scheduler.current_step = 55 # 50% through after warmup
    weight = scheduler.get_weight()
    assert 0.0 < weight < 1.0
    
    # Finished
    scheduler.current_step = 100
    weight = scheduler.get_weight()
    assert np.isclose(weight, 1.0, atol=1e-2)
    print("SparsityScheduler test passed!")

def test_recursive_feature_selector():
    print("Testing RecursiveFeatureSelector...")
    # Generate dummy data
    np.random.seed(42)
    n_samples = 200
    n_features = 50
    X = np.random.randn(n_samples, n_features)
    # Target depends on features 5 and 10
    y = 2.5 * X[:, 5] - 1.2 * X[:, 10] + 0.1 * np.random.randn(n_samples)
    
    # Add some constant features
    X[:, 0] = 1.0
    X[:, 1] = 0.0
    
    selector = RecursiveFeatureSelector(max_features=5)
    selector.fit(X, y)
    
    # Ensure constant features are NOT selected
    assert 0 not in selector.selected_indices
    assert 1 not in selector.selected_indices
    
    # Ensure informative features ARE selected
    assert 5 in selector.selected_indices
    assert 10 in selector.selected_indices
    
    print(f"Selected indices: {selector.selected_indices}")
    print("RecursiveFeatureSelector test passed!")

def test_sympy_robustness():
    print("Testing SymPy robustness in OptimizedSymbolicDynamics...")
    # Mock objects for initialization
    class MockDistiller:
        def __init__(self):
            self.transformer = None
            
    distiller = MockDistiller()
    
    # Test complex string conversion
    from gplearn.genetic import _Program
    
    # We'll just test the _convert_to_sympy method directly if we can, 
    # or mock a program object
    class MockProgram:
        def __init__(self, expr_str):
            self.expr_str = expr_str
        def __str__(self):
            return self.expr_str
            
    dynamics = OptimizedSymbolicDynamics.__new__(OptimizedSymbolicDynamics)
    
    # Test prefix notation and custom functions
    prog = MockProgram("add(mul(X0, X1), inv(add(X2, 1.0)))")
    expr = dynamics._convert_to_sympy(prog)
    print(f"Converted expr: {expr}")
    assert isinstance(expr, sp.Expr)
    assert 'x0' in str(expr)
    assert 'x1' in str(expr)
    assert 'x2' in str(expr)

    # Test robustness with negative values in sqrt/log
    prog2 = MockProgram("log(sub(X0, 10.0))")
    expr2 = dynamics._convert_to_sympy(prog2)
    print(f"Converted expr (log): {expr2}")
    # Should be simplified to something involving Abs or just be valid
    
    print("SymPy robustness test passed!")

if __name__ == "__main__":
    test_sparsity_scheduler()
    test_recursive_feature_selector()
    test_sympy_robustness()
