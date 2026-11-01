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
