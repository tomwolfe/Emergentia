import numpy as np
import sympy as sp
from hamiltonian_symbolic import HamiltonianEquation, HamiltonianSymbolicDistiller
from symbolic import FeatureTransformer, gp_to_sympy

def test_analytical_gradients():
    # Setup a simple Hamiltonian: H = 0.5 * p^2 + 0.5 * q^2
    # In features: x0=q, x1=p, x2=q^2, x3=p^2
    # X_norm = (X - 0) / 1
    
    class MockProg:
        def __init__(self, expr_str):
            self.expr_str = expr_str
            self.length_ = 5
        def execute(self, X):
            # Very simple implementation for testing
            q = X[:, 0]
            p = X[:, 1]
            return 0.5 * p**2 + 0.5 * q**2
        def __str__(self):
            return self.expr_str

    h_prog = MockProg("add(mul(0.5, mul(X1, X1)), mul(0.5, mul(X0, X0)))")
    feature_mask = [True, True]
    n_super_nodes = 1
    latent_dim = 2 # q, p
    
    # Create SymPy expression and gradients
    sympy_expr = gp_to_sympy(str(h_prog), n_features=2)
    sympy_vars = [sp.Symbol(f'x{i}') for i in range(2)]
    grad_funcs = [sp.lambdify(sympy_vars, sp.diff(sympy_expr, v), 'numpy') for v in sympy_vars]
    
    ham_eq = HamiltonianEquation(
        h_prog, feature_mask, n_super_nodes, latent_dim,
        sympy_expr=sympy_expr, grad_funcs=grad_funcs
    )
    
    class MockTransformer:
        def __init__(self):
            self.x_poly_std = np.ones(2)
        def transform(self, z):
            return z
        def normalize_x(self, X):
            return X
        def transform_jacobian(self, z):
            return np.eye(2)
            
    transformer = MockTransformer()
    
    # Test point: q=1, p=2
    z = np.array([1.0, 2.0])
    # dq/dt = dH/dp = p = 2
    # dp/dt = -dH/dq = -q = -1
    
    dzdt = ham_eq.compute_derivatives(z, transformer)
    print(f"Analytical dzdt: {dzdt}")
    
    assert np.allclose(dzdt, [2.0, -1.0])
    
    # Test without analytical gradients (fallback to numerical)
    ham_eq_num = HamiltonianEquation(h_prog, feature_mask, n_super_nodes, latent_dim)
    dzdt_num = ham_eq_num.compute_derivatives(z, transformer)
    print(f"Numerical dzdt: {dzdt_num}")
    assert np.allclose(dzdt_num, [2.0, -1.0], atol=1e-3)

def test_constant_simplification():
    from symbolic import SymbolicDistiller
    distiller = SymbolicDistiller()
    
    # Mock data for y = 2.1 * x^2
    X = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2.0 * X[:, 0]**2 # Ground truth is 2.0
    
    # We want to see if it simplifies a "noisy" 2.1 to 2.0
    # or a "noisy" 1.9 to 2.0
    
    class MockProg:
        def __init__(self, c):
            self.c = c
            self.length_ = 3
        def execute(self, X):
            return self.c * X[:, 0]**2
        def __str__(self):
            return f"mul({self.c}, mul(X0, X0))"

    candidate = {'prog': MockProg(2.08), 'score': 0.99, 'complexity': 3}
    
    # We need a real X for _refine_constants
    # Mock FeatureTransformer to avoid errors
    distiller.transformer = type('obj', (object,), {'x_poly_std': np.array([1.0]), 'transform': lambda x: x, 'normalize_x': lambda x: x})
    
    refined = distiller._refine_constants(candidate, X, y)
    
    print(f"Original const: 2.08")
    print(f"Refined program: {refined['prog']}")
    print(f"Refined score: {refined['score']}")
    
    # Check if the constant was updated to something close to 2.0
    refined_str = str(refined['prog'])
    import re
    new_consts = re.findall(r'\d+\.\d+', refined_str)
    print(f"New consts found: {new_consts}")
    
    if new_consts:
        val = float(new_consts[0])
        print(f"Refined value: {val}")
        assert abs(val - 2.0) < 1e-2
    else:
        # Check if it's an integer 2
        new_consts_int = re.findall(r'\b\d+\b', refined_str)
        if '2' in new_consts_int or '2.0' in refined_str:
            print("Successfully simplified to 2 or 2.0")
        else:
            print(f"Failed to find expected constant in {refined_str}")
            assert False

if __name__ == "__main__":
    test_analytical_gradients()
    test_constant_simplification()
    print("All verification tests passed!")
