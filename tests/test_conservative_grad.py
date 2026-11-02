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
    assert pos.requires_grad is False
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


def test_torch_force_matches_neg_dVdr_all_potentials():
    """Torch identity: compute_force_magnitude == -dV/dr via autograd."""
    pots = [
        HarmonicPotential(), LennardJonesPotential(), MorsePotential(),
        GravityPotential(), BuckinghamPotential(), YukawaPotential(),
    ]
    for pot in pots:
        rg = torch.tensor([0.8, 1.0, 1.5, 2.0], requires_grad=True)
        Vg = pot.compute_potential(rg)
        Vg.sum().backward()
        expected = -rg.grad
        got = pot.compute_force_magnitude(rg.detach())
        assert torch.allclose(got, expected, atol=1e-4), (
            f"{type(pot).__name__}: F != -dV/dr: {got.tolist()} vs {expected.tolist()}"
        )


def test_potential_scales_and_dt_exact():
    # Exact scale/dt values kill constant mutants in simulator.py.
    from emergentia.simulator import (
        LennardJonesPotential, MorsePotential, GravityPotential,
        BuckinghamPotential, YukawaPotential, CompositePotential,
    )
    assert HarmonicPotential().default_scale == 2.0
    assert HarmonicPotential().dt == 0.01
    assert LennardJonesPotential().default_scale == 3.5
    assert LennardJonesPotential().dt == 0.001
    assert MorsePotential().default_scale == 3.0
    assert MorsePotential().dt == 0.005
    assert GravityPotential().default_scale == 5.0
    assert GravityPotential().dt == 0.01
    assert BuckinghamPotential().default_scale == 3.0
    assert BuckinghamPotential().dt == 0.001
    assert YukawaPotential().default_scale == 4.0
    assert YukawaPotential().dt == 0.005
    comp = CompositePotential([HarmonicPotential(), GravityPotential()])
    assert comp.default_scale == 5.0
    assert comp.dt == 0.01


def test_force_spot_values_kill_coefficient_mutants():
    from emergentia.simulator import (
        LennardJonesPotential, MorsePotential, GravityPotential,
        BuckinghamPotential, YukawaPotential, CompositePotential,
    )
    # Harmonic: F = -k*(r-r0), k=10, r0=1
    h = HarmonicPotential(k=10.0, r0=1.0)
    assert float(h.compute_force_magnitude(torch.tensor([[1.0]])).view(-1)[0]) == 0.0
    assert float(h.compute_force_magnitude(torch.tensor([[2.0]])).view(-1)[0]) == -10.0
    # Gravity: F = -G/r^2
    g = GravityPotential(G=1.0)
    assert abs(float(g.compute_force_magnitude(torch.tensor([2.0])).view(-1)[0]) + 0.25) < 1e-6
    # LJ at r=1 (unclamped interior): 48 - 24 = 24
    lj = LennardJonesPotential(epsilon=1.0, sigma=1.0)
    assert abs(float(lj.compute_force_magnitude(torch.tensor([1.0])).view(-1)[0]) - 24.0) < 1e-4
    # Morse at r=re: F = 0
    m = MorsePotential(De=1.0, a=1.0, re=1.0)
    assert abs(float(m.compute_force_magnitude(torch.tensor([1.0])).view(-1)[0])) < 1e-6
    # Buckingham / Yukawa spot values vs closed form (kills A*B / 6*C mutants)
    import math
    b = BuckinghamPotential(A=10.0, B=1.0, C=1.0)
    r = 1.0
    expect_b = 10.0 * math.exp(-r) - 6.0 * r**-7
    assert abs(float(b.compute_force_magnitude(torch.tensor([r])).view(-1)[0]) - expect_b) < 1e-4
    y = YukawaPotential(A=1.0, B=1.0)
    expect_y = math.exp(-r) * (1.0 / r + 1.0 / r**2)
    assert abs(float(y.compute_force_magnitude(torch.tensor([r])).view(-1)[0]) - expect_y) < 1e-4
    # Composite sums components
    comp = CompositePotential([HarmonicPotential(k=10.0, r0=1.0), GravityPotential(G=1.0)])
    rr = torch.tensor([2.0])
    expect_c = float(HarmonicPotential(k=10.0, r0=1.0).compute_force_magnitude(rr).view(-1)[0]) + float(
        GravityPotential(G=1.0).compute_force_magnitude(rr).view(-1)[0])
    assert abs(float(comp.compute_force_magnitude(rr).view(-1)[0]) - expect_c) < 1e-6


def test_composite_potential_sums_potential():
    from emergentia.simulator import CompositePotential
    comp = CompositePotential([HarmonicPotential(k=10.0, r0=1.0), GravityPotential(G=1.0)])
    d = torch.tensor([2.0])
    got = comp.compute_potential(d)
    assert got is not None
    h = HarmonicPotential(k=10.0, r0=1.0).compute_potential(d)
    g = GravityPotential(G=1.0).compute_potential(d)
    assert torch.allclose(got, h + g, atol=1e-6)


def test_initial_positions_use_multiplicative_scale():
    from emergentia.simulator import PhysicsSim
    sim = PhysicsSim(n=2, dim=2, potential=HarmonicPotential(k=10.0, r0=1.0), seed=0, device=torch.device("cpu"), use_neighbor_list=True, cutoff=10.0)
    assert sim.use_neighbor_list is True
    assert float(sim.pos.max()) > 1.0
    assert float(sim.pos.max()) <= 2.0


def _make_nl_sim(cutoff, seed=0):
    from emergentia.simulator import PhysicsSim
    return PhysicsSim(n=2, dim=2, potential=HarmonicPotential(k=10.0, r0=1.0), seed=seed, device=torch.device("cpu"), use_neighbor_list=True, cutoff=cutoff)


def test_neighbor_list_matches_raw_when_cutoff_covers_all():
    sim = _make_nl_sim(cutoff=10.0)
    pos = torch.tensor([[0.0, 0.0], [1.2, 0.3]])
    f_nl = sim._compute_forces_neighbor_list(pos)
    f_raw = sim._compute_forces_raw(pos)
    assert torch.allclose(f_nl, f_raw, atol=1e-6)


def test_neighbor_list_boundary_inclusive_kills_gte_mutant():
    sim = _make_nl_sim(cutoff=2.0)
    pos = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    f_nl = sim._compute_forces_neighbor_list(pos)
    f_raw = sim._compute_forces_raw(pos)
    assert torch.allclose(f_nl, f_raw, atol=1e-6)
    assert f_nl.abs().max().item() > 1.0


def test_neighbor_list_excludes_beyond_cutoff():
    sim = _make_nl_sim(cutoff=2.0)
    pos = torch.tensor([[0.0, 0.0], [2.5, 0.0]])
    f_nl = sim._compute_forces_neighbor_list(pos)
    assert torch.allclose(f_nl, torch.zeros_like(f_nl), atol=1e-6)
    f_raw = sim._compute_forces_raw(pos)
    assert f_raw.abs().max().item() > 1.0


def _identity_field():
    from emergentia.physics_constraints import ConservativeForceField
    import torch.nn as nn
    class IdentityV(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Identity()
            self._dist_min = 1e-4
        def forward(self, x):
            return x
    return ConservativeForceField(IdentityV())


def test_conservative_field_force_magnitude_pins_pair_half():
    # V(r) = r identity net: E = 0.5 * sum_{i!=j} |xi-xj| = 1.0 for unit pair.
    # Kills total_energy scale mutants (0.5 -> 0.6 gives |F| = 1.2).
    field = _identity_field()
    pos = torch.tensor([[[0.0], [1.0]]])
    f = field(pos)
    assert torch.allclose(f, torch.tensor([[[1.0], [-1.0]]]), atol=1e-6), f


def test_get_potential_energy_pins_pair_half():
    # Same identity setup: scalar energy must be exactly 1.0 (kills *0.5 mutants).
    field = _identity_field()
    pos = torch.tensor([[[0.0], [1.0]]])
    e = field.get_potential_energy(pos)
    assert abs(float(e.detach())) - 1.0 < 1e-6, e


def test_predict_mag_is_neg_dvdr_identity():
    # V(r) = r -> F = -dV/dr = -1 everywhere (kills negation-drop mutants).
    field = _identity_field()
    r = torch.tensor([[0.5], [2.0]])
    got = field.predict_mag(r)
    assert torch.allclose(got, torch.tensor([[-1.0], [-1.0]]), atol=1e-6), got
