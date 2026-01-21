
import torch
import numpy as np
import sympy as sp
from symbolic import gp_to_sympy
from enhanced_symbolic import SymPyToTorch, TorchFeatureTransformer
from engine import SymbolicProxy

def test_gp_to_sympy():
    print("Testing gp_to_sympy...")
    exprs = [
        "add(X0, X1)",
        "mul(X0, div(1, X2))",
        "sin(X0)",
        "X10",
        "add(X0, mul(2.5, X1))"
    ]
    for e in exprs:
        se = gp_to_sympy(e)
        print(f"  '{e}' -> {se}")
        assert isinstance(se, sp.Expr)
    print("gp_to_sympy tests passed!")

def test_vectorized_dists():
    print("Testing vectorized distances...")
    from symbolic import FeatureTransformer
    ft = FeatureTransformer(n_super_nodes=4, latent_dim=4, include_dists=True)
    z = np.random.randn(10, 16) # 10 samples, 4 nodes * 4 dims
    ft.x_poly_mean = np.zeros(1) # dummy
    ft.x_poly_std = np.ones(1) # dummy
    
    X = ft.transform(z)
    print(f"  Feature shape: {X.shape}")
    # K=4, K*(K-1)/2 = 6 pairs. 6 dists, 6 inv_dists, 6 inv_sq_dists = 18 dist features.
    # Raw latents = 16.
    # Total linear features = 16 + 18 = 34.
    # Quadratic features: 16 squares, plus cross terms... it's a lot.
    assert X.shape[1] > 34
    print("Vectorized distances tests passed!")

def test_sympy_to_torch():
    print("Testing SymPyToTorch stability...")
    x0, x1 = sp.symbols('x0 x1')
    expr = 1.0 / (x0**2 + 0.1) + sp.sqrt(sp.Abs(x1))
    mod = SymPyToTorch(expr, n_inputs=2)
    
    # Test with zero input (potential singularity)
    x_in = torch.zeros(5, 2)
    out = mod(x_in)
    print(f"  Output at zero: {out}")
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    
    # Test gradient
    x_in.requires_grad_(True)
    out = mod(x_in)
    out.sum().backward()
    print(f"  Gradient at zero: {x_in.grad}")
    assert not torch.isnan(x_in.grad).any()
    print("SymPyToTorch stability tests passed!")

if __name__ == "__main__":
    test_gp_to_sympy()
    test_vectorized_dists()
    test_sympy_to_torch()
