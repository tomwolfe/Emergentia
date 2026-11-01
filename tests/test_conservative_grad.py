"""QED hardening: ConservativeForceField.forward matches analytical -grad V."""
import torch
import sympy as sp
from emergentia.simulator import HarmonicPotential, LennardJonesPotential, MorsePotential, GravityPotential, BuckinghamPotential, YukawaPotential
from emergentia.physics_priors import PhysicsPriorProvider

def test_force_is_neg_dVdr():
    r = sp.Symbol('r', positive=True)
    cases = {
        'harmonic': (sp.Rational(1,2)*10*(r-1)**2, -10*(r-1)),
        'gravity': (-1/r, -1/r**2),
        'morse': (1*(1-sp.exp(-1*(r-1)))**2, -2*(sp.exp(-(r-1))-sp.exp(-2*(r-1)))),
        'buckingham': (10*sp.exp(-r)-1/r**6, 10*sp.exp(-r)-6/r**7),
        'yukawa': (sp.exp(-r)/r, sp.exp(-r)*(1/r+1/r**2)),
    }
    import numpy as np
    fns = {n: (sp.lambdify(r, sp.diff(V, r) + F, 'numpy')) for n, (V, F) in cases.items()}
    xs = np.linspace(0.5, 5.0, 50)
    for name, fn in fns.items():
        assert np.allclose(fn(xs), 0, atol=1e-8), f"{name}: F != -dV/dr"

def test_conservative_field_matches_grad():
    from emergentia.physics_constraints import ConservativeForceField
    import torch.nn as nn
    class V(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
            self._dist_min = 1e-4
        def forward(self, x): return x
    net = V()
    field = ConservativeForceField(net)
    pos = torch.randn(2, 3, 2, requires_grad=False)
    f = field(pos)
    assert f.shape == pos.shape


def test_spring_functional_form_rejects_exp():
    from emergentia.utils import check_functional_form
    r = sp.Symbol('r')
    assert check_functional_form(r, "spring") is True
    assert check_functional_form(sp.exp(-r), "spring") is False


def test_verify_equivalence_rejects_bad_expr():
    from emergentia.utils import verify_equivalence
    ok, m = verify_equivalence(sp.sympify("0"), "spring", potential=HarmonicPotential())
    assert not ok
    assert m.get("r2", 0.0) < 0.95


def test_harmonic_force_sign():
    pot = HarmonicPotential(k=10.0, r0=1.0)
    f = pot.compute_force_magnitude(torch.tensor([[2.0]]))
    assert float(f.view(-1)[0]) < 0, "F(2) must be -k*(2-1)=-10"


def test_conservative_field_sign_is_neg_grad():
    from emergentia.physics_constraints import ConservativeForceField
    import torch.nn as nn
    class IdentityV(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Identity()
            self._dist_min = 1e-4
        def forward(self, x):
            return x
    field = ConservativeForceField(IdentityV())
    pos = torch.tensor([[[0.0], [1.0]]])
    f = field(pos)
    assert float(f[0, 0, 0].detach()) > 0, "attractive V=r must pull pos0 +x"
    assert float(f[0, 1, 0].detach()) < 0, "attractive V=r must pull pos1 -x"
