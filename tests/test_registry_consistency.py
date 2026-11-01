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

def test_registry_unknown_basis_raises():
    with pytest.raises(ValueError):
        PhysicalBasisRegistry.get("no_such_basis", backend="torch")


def test_registry_get_registry_complete():
    reg = PhysicalBasisRegistry.get_registry(backend="torch")
    assert set(reg.keys()) == set(PhysicalBasisRegistry.list_basis())
    assert len(reg) > 0
    assert all(callable(v) for v in reg.values())


def test_registry_unknown_backend_raises():
    with pytest.raises(ValueError):
        PhysicalBasisRegistry.get("r", backend="no_such_backend")


def test_registry_exact_values_spot_check():
    import torch
    import numpy as np
    import sympy as sp
    r_sym = sp.Symbol("r")
    # Exact spot values kill coefficient/operator mutants in _BASIS lambdas.
    cases = {
        "r": 2.0,
        "1/r": 0.5,
        "1/r^2": 0.25,
        "r^2": 4.0,
        "exp(-r)": float(np.exp(-2.0)),
        "1": 1.0,
    }
    for name, expected in cases.items():
        for backend in ("torch", "numpy", "sympy"):
            fn = PhysicalBasisRegistry.get(name, backend=backend)
            if backend == "torch":
                out = float(fn(torch.tensor(2.0)).item())
            elif backend == "numpy":
                out = float(fn(np.array(2.0)).item())
            else:
                out = float(sp.lambdify(r_sym, fn(r_sym), "numpy")(2.0))
            assert abs(out - expected) < 1e-6, f"{name}/{backend}: {out} != {expected}"


def test_registry_basis_set_stable():
    expected = {
        "1", "r", "1/r", "1/r^2", "1/r^6", "1/r^7", "1/r^12",
        "1/r^13", "exp(-r)", "exp(-r)/r", "r^2", "log(r+1)", "exp(-r)/r^2",
    }
    assert set(PhysicalBasisRegistry.list_basis()) == expected


def test_neighbor_list_path_runs():
    import torch
    from emergentia.simulator import PhysicsSim, HarmonicPotential
    sim = PhysicsSim(n=3, dim=2, potential=HarmonicPotential(), seed=0,
                     device=torch.device("cpu"), use_neighbor_list=True)
    p, f = sim.generate(steps=5)
    assert p.shape == (5, 3, 2) and torch.isfinite(f).all()
