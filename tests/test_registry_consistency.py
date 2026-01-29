import torch
import numpy as np
import sympy as sp
import pytest
from emergentia.registry import PhysicalBasisRegistry

def test_registry_consistency():
    basis_names = PhysicalBasisRegistry.list_basis()
    r_values = np.linspace(0.5, 5.0, 100).astype(np.float32)
    r_torch = torch.from_numpy(r_values)
    r_sym = sp.Symbol('r')
    
    tolerance = 1e-5
    
    for name in basis_names:
        print(f"Testing basis: {name}")
        
        # NumPy output
        np_func = PhysicalBasisRegistry.get(name, backend='numpy')
        np_out = np_func(r_values)
        
        # Torch output
        torch_func = PhysicalBasisRegistry.get(name, backend='torch')
        torch_out = torch_func(r_torch).numpy()
        
        # SymPy output
        sp_func = PhysicalBasisRegistry.get(name, backend='sympy')
        sp_expr = sp_func(r_sym)
        sp_lambdas = sp.lambdify(r_sym, sp_expr, 'numpy')
        sp_out = sp_lambdas(r_values)
        
        # If sympy returns a scalar (like for '1'), broadcast it
        if np.isscalar(sp_out):
            sp_out = np.full_like(np_out, sp_out)
        
        # Verify NumPy vs Torch
        np.testing.assert_allclose(np_out, torch_out, atol=tolerance, rtol=tolerance, 
                                   err_msg=f"NumPy vs Torch inconsistency for basis '{name}'")
        
        # Verify NumPy vs SymPy
        np.testing.assert_allclose(np_out, sp_out, atol=tolerance, rtol=tolerance, 
                                   err_msg=f"NumPy vs SymPy inconsistency for basis '{name}'")

if __name__ == "__main__":
    test_registry_consistency()
