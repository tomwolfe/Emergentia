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


# ---------------------------------------------------------------------------
# Mutation-hardening for emergentia-09 formal-export gate (test-only,
# behavioral: each test checks numerics/grads/errors, not source text).
# ---------------------------------------------------------------------------

def _make_odefunc(n_particles=2, dim=2):
    from emergentia.differentiable_solver import ODEFunc
    from emergentia.models import DiscoveryNet
    torch.manual_seed(0)
    model = DiscoveryNet(hidden_size=8)
    return ODEFunc(model, mass=1.0, dim=dim, n_particles=n_particles)


def test_odefunc_valid_forward_shape_and_finite():
    # Kills 28:15, 32:19, 33:15: valid (batch, 2*n*dim) must succeed.
    # n=3, dim=2 separates 2*n*dim=12 from (2*n)/dim=3.
    from emergentia.differentiable_solver import ODEFunc
    from emergentia.models import DiscoveryNet
    torch.manual_seed(0)
    model = DiscoveryNet(hidden_size=8)
    ode = ODEFunc(model, mass=1.0, dim=2, n_particles=3)
    state = torch.tensor([[0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = ode(torch.tensor(0.0), state)
    assert out.shape == (2, 12), out.shape
    assert torch.isfinite(out).all().item(), out


def test_odefunc_rejects_bad_shapes():
    ode = _make_odefunc(n_particles=3, dim=2)
    t = torch.tensor(0.0)
    # Build a valid state then corrupt it; mutant (== -> !=) accepts bad.
    import pytest
    with pytest.raises(AssertionError):
        ode(t, torch.randn(2, 11))
    with pytest.raises(AssertionError):
        ode(t, torch.randn(2, 12, 1))


def test_odefunc_requires_grad_true_branch():
    # Kills 59:55 (requires_grad_(True)->False): forward must not raise
    # and output must stay finite; mutant raises RuntimeError in grad.
    ode = _make_odefunc(n_particles=2, dim=2)
    state = torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = ode(torch.tensor(0.0), state)
    assert torch.isfinite(out).all().item(), out


def test_odefunc_grad_to_params_kills_retain_mutant():
    # Kills 64:65 (retain_graph True->False): the first outer grad works
    # either way, but only the original survives a SECOND backward through
    # the same graph. Mutant raises "backward through the graph a second
    # time" on the repeat.
    ode = _make_odefunc(n_particles=2, dim=2)
    state = torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    out = ode(torch.tensor(0.0), state)
    assert torch.isfinite(out).all().item(), out
    params = list(ode.model.parameters())
    grads = torch.autograd.grad(out.sum(), params, allow_unused=True, retain_graph=True)
    assert any(g is not None for g in grads), "grad must reach model params"
    grads2 = torch.autograd.grad(out.sum(), params, allow_unused=True)
    assert any(g is not None for g in grads2), "second backward must work"


def test_odefunc_accel_scales_with_inverse_mass():
    # Kills 81:18 arithmetic variants: with mass=2.0, `/ mass` halves the
    # force while `* mass` doubles it (indistinguishable only at mass=1.0,
    # so this test pins mass=2.0 explicitly).
    # Fixed, well-separated geometry: no coincident pair, dist = 2.0,
    # so no singular path can produce NaNs.
    state = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]).view(1, 8)
    t = torch.tensor(0.0)
    torch.manual_seed(0)
    from emergentia.models import DiscoveryNet
    from emergentia.differentiable_solver import ODEFunc
    ode1 = ODEFunc(DiscoveryNet(hidden_size=8), mass=1.0, dim=2, n_particles=2)
    torch.manual_seed(0)
    ode2 = ODEFunc(DiscoveryNet(hidden_size=8), mass=2.0, dim=2, n_particles=2)
    a1 = ode1(t, state)[:, 4:]
    a2 = ode2(t, state)[:, 4:]
    assert torch.isfinite(a2).all().item(), a2
    assert torch.allclose(a2, a1 / 2.0, atol=1e-6), (a1, a2)


def test_odefunc_finite_forces_kill_div_mutant():
    # Kills 81:18 (f_mag * dir_vec -> / gives inf/nan).
    ode = _make_odefunc(n_particles=2, dim=2)
    state = torch.tensor([[0.0, 0.0, 2.0, 0.0, 0.5, -0.5, 0.25, 0.75]])
    out = ode(torch.tensor(0.0), state)
    assert torch.isfinite(out).all().item(), out
    assert out.abs().max().item() < 1e6, out


def test_simulator_dim1_neighbor_list_matches_raw():
    # Kills simulator.py:336:15 (dim==2 -> !=2): dim=1 takes the flat
    # branch; mutant takes the 2D branch and raises IndexError.
    from emergentia.simulator import HarmonicPotential, PhysicsSim
    sim = PhysicsSim(n=2, dim=1, potential=HarmonicPotential(k=10.0, r0=1.0),
                     seed=0, device=torch.device("cpu"),
                     use_neighbor_list=True, cutoff=10.0)
    # Pin the flag itself: kills tests/test_conservative_grad.py:338:39
    # [flip_bool] (True -> False is otherwise unobservable since the test
    # calls _compute_forces_neighbor_list directly).
    assert sim.use_neighbor_list is True
    assert getattr(sim._compute_forces_compiled, "__func__", None) is PhysicsSim._compute_forces_neighbor_list
    pos = torch.tensor([[0.0], [1.2]])
    f_nl = sim._compute_forces_neighbor_list(pos)
    f_raw = sim._compute_forces_raw(pos)
    assert torch.allclose(f_nl, f_raw, atol=1e-6), (f_nl, f_raw)
    # Dispatched path must follow the flag.
    f_dispatch = sim._compute_forces_compiled(pos)
    assert torch.allclose(f_dispatch, f_raw, atol=1e-6), (f_dispatch, f_raw)


def test_get_potential_energy_requires_grad():
    # Kills physics_constraints.py:62:11 (not -> identity): non-grad input
    # must still yield an energy that requires grad for -grad(V).
    field = _identity_field()
    pos = torch.tensor([[[0.0], [1.0]]])
    assert pos.requires_grad is False
    e = field.get_potential_energy(pos)
    assert abs(float(e.detach()) - 1.0) < 1e-6, e
    assert e.requires_grad is True, "energy must require grad"
    assert e.grad_fn is not None


def test_pairwise_mask_multiplies_not_divides():
    # Kills physics_constraints.py:35:17 (* mask -> / mask): with
    # _dist_min=0 the diagonal V is exactly 0, so * masks it to 0
    # (finite, exact forces) while / divides 0/0 on the False diagonal
    # (nan -> nan forces). With the default 1e-4 floor the mutant still
    # yields finite forces (diagonal grad is zero), so the zero floor is
    # required to make the mutant observable.
    from emergentia.physics_constraints import ConservativeForceField
    import torch.nn as nn

    class ZeroMinNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Identity()
            self._dist_min = 0.0

    field = ConservativeForceField(ZeroMinNet())
    pos = torch.tensor([[[0.0], [1.0]]])
    f = field(pos)
    assert torch.isfinite(f).all().item(), f
    assert torch.allclose(f, torch.tensor([[[1.0], [-1.0]]]), atol=1e-6), f
    e = field.get_potential_energy(pos)
    assert torch.isfinite(e).item() and abs(float(e.detach()) - 1.0) < 1e-6, e


def test_refine_constants_recovers_spring_kills_objective_mutant():
    # Kills engine.py:605:26 (* -> / gives nan MSE, refine stalls at 9).
    import sympy as sp
    import numpy as np
    from emergentia.engine import DiscoveryPipeline
    from emergentia.simulator import HarmonicPotential, PhysicsSim
    r = sp.Symbol("r")
    torch.manual_seed(0)
    np.random.seed(0)
    pot = HarmonicPotential(k=10.0, r0=1.0)
    sim = PhysicsSim(n=2, dim=2, potential=pot, seed=0,
                     device=torch.device("cpu"))
    p_traj = torch.randn(10, 2, 2) * 2.0
    f_traj = torch.stack([sim._compute_forces_raw(p) for p in p_traj])
    pipe = DiscoveryPipeline(mode="spring", potential=pot, device="cpu",
                             seed=0, enable_unit_checker=False,
                             enable_physics_priors=False,
                             enable_auto_smoother=False)
    ref = pipe.refine_constants(sp.Float(9.0) * (r - 1), p_traj, f_traj)
    fn = sp.lambdify(r, ref, "numpy")
    assert abs(float(fn(2.0)) - 10.0) < 0.5, ref


def test_engine_validate_trajectory_sub_vs_add():
    # Kills engine.py:865:32 (- -> +): true law diff-error < 2.0,
    # sum-error is several x larger.
    import sympy as sp
    from emergentia.engine import DiscoveryPipeline
    from emergentia.simulator import HarmonicPotential, PhysicsSim
    sim = PhysicsSim(n=2, dim=2, potential=HarmonicPotential(), seed=1,
                     device=torch.device("cpu"))
    pipe = DiscoveryPipeline(mode="spring", potential=sim.potential, seed=1,
                             device="cpu", enable_unit_checker=False,
                             enable_physics_priors=False,
                             enable_auto_smoother=False)
    r = sp.Symbol("r")
    res = pipe.validate_trajectory(-10 * (r - 1), sim, steps=5)
    assert res["trajectory_mape"] < 2.0, res


def test_engine_run_returns_dict():
    # Kills engine.py:825:8 (return result_dict -> None) via fast mocks.
    from emergentia.engine import DiscoveryPipeline
    from emergentia.simulator import HarmonicPotential, PhysicsSim
    import sympy as sp
    sim = PhysicsSim(n=2, dim=2, potential=HarmonicPotential(), seed=0,
                     device=torch.device("cpu"))
    pipe = DiscoveryPipeline(mode="spring", potential=sim.potential, seed=0,
                             device="cpu", enable_unit_checker=False,
                             enable_physics_priors=False,
                             enable_auto_smoother=False)
    pipe.train_nn = lambda *a, **k: 0.01
    pipe.distill_symbolic = lambda *a, **k: sp.Symbol("r")
    pipe.refine_constants = lambda expr, p, f: expr
    pipe.validate_trajectory = lambda expr, s, steps=200: {"trajectory_mape": 0.0}
    sim.generate = lambda steps=2000, noise_std=0.0, impulses=True: (
        torch.randn(100, 2, 2), torch.randn(100, 2, 2))
    result = pipe.run(sim, nn_epochs=1, noise_std=0.0)
    assert isinstance(result, dict), type(result)
    assert "mode" in result and "formula" in result, result.keys()


def test_differentiable_train_early_return_pins_max_idx():
    # Kills engine.py:930:22 (- 1 -> + 1): T=11, epochs=1 gives n_steps=10,
    # orig max_idx=0 early-returns, mutant max_idx=2 proceeds to ODE error.
    import io
    import contextlib
    from emergentia.engine import DifferentiableDiscoveryPipeline
    from emergentia.simulator import HarmonicPotential
    pipe = DifferentiableDiscoveryPipeline(
        mode="spring", potential=HarmonicPotential(), device="cpu", seed=0,
        enable_unit_checker=False, enable_physics_priors=False,
        enable_auto_smoother=False)
    p_traj = torch.randn(11, 2, 2)
    f_traj = torch.randn(11, 2, 2)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = pipe.train_nn(p_traj, f_traj, epochs=1)
    assert ret == 0.0
    assert "Not enough timesteps" in buf.getvalue(), buf.getvalue()


# ---------------------------------------------------------------------------
# Preprocessing teeth: the deliverable requires kill_rate 1.0 over
# emergentia/preprocessing.py, which no gate suite covered (0.0000 measured).
# ---------------------------------------------------------------------------

def _sine_fixture(T=150, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    y = np.sin(np.linspace(0, 6, T))[:, None, None]
    y = y + 0.05 * rng.randn(T, 1, 1)
    return torch.tensor(y, dtype=torch.float32)


def test_smoother_forward_grad_path_returns_tensor():
    # Kills preprocessing.py:134:12 (return _differentiable_smooth -> None):
    # requires_grad input must take the differentiable path and return a
    # finite tensor carrying a grad_fn (the mutant returns None).
    from emergentia.preprocessing import AutoSmoother
    s = AutoSmoother()
    out = s(torch.randn(32, 2, 1, requires_grad=True))
    assert isinstance(out, torch.Tensor), type(out)
    assert torch.isfinite(out).all().item(), out
    assert out.grad_fn is not None, "differentiable path must keep grad"


def test_smoother_forward_nongrad_matches_scipy():
    # Guards the SciPy branch: non-grad input returns a finite tensor of the
    # same shape (and the smoother is enabled by default).
    from emergentia.preprocessing import AutoSmoother
    s = AutoSmoother()
    t = torch.randn(32, 2, 1)
    out = s(t)
    assert isinstance(out, torch.Tensor) and out.shape == t.shape
    assert torch.isfinite(out).all().item()


def test_optimize_bandwidth_pins_exact_schedule():
    # Kills preprocessing.py:210:18 (trial schedule), 249:50, 249:70,
    # 276:54 (LOO edge-index arithmetic): any change to the tried bandwidth
    # set or the leave-one-out predictions moves best_bandwidth and hence
    # the smoothed update. Values calibrated on the committed implementation
    # (pure-numpy LOO loop, bit-stable across runs).
    from emergentia.preprocessing import AutoSmoother
    s = AutoSmoother()
    assert abs(s.get_bandwidth() - 1.0) < 1e-9
    traj = _sine_fixture()
    r1 = s.optimize_bandwidth(traj, [])
    assert abs(r1 - 1.3242257833480835) < 1e-6, r1
    r2 = s.optimize_bandwidth(traj, [r1])
    assert abs(r2 - 1.6118903160095215) < 1e-6, r2
    assert 0.1 <= r2 <= 5.0


def test_optimize_bandwidth_disabled_returns_max():
    # Pins the is_enabled early return (fail-closed coverage of the branch).
    from emergentia.preprocessing import AutoSmoother
    s = AutoSmoother()
    s.disable()
    assert s.optimize_bandwidth(_sine_fixture(T=32), []) == s.max_bandwidth


def _spike_fixtures(T=150, seed=0):
    import numpy as np
    rng = np.random.RandomState(seed)
    base = np.sin(np.linspace(0, 6, T))[:, None, None]
    base = base + 0.05 * rng.randn(T, 1, 1)
    spike = np.zeros((T, 1, 1))
    spike[75, 0, 0] = 2.0
    d1 = torch.tensor(base + spike, dtype=torch.float32)
    d2 = torch.tensor(np.concatenate([base, base + spike], axis=2),
                      dtype=torch.float32)
    return d1, d2


def test_optimize_bandwidth_mid_spike_selects_wide_kernel():
    # Kills preprocessing.py:249:70 (`(T-1-pad)` -> `(T-1+pad)` in the D==1
    # edge branch): with a mid-trajectory spike the wide kernel (5.0) wins
    # LOO-CV; the mutant shifts edge predictions by 2*pad, the ranking flips
    # to 2.55, and the smoothed update moves from 1.6206... to 1.3242....
    from emergentia.preprocessing import AutoSmoother
    d1, _ = _spike_fixtures()
    got = AutoSmoother().optimize_bandwidth(d1, [])
    assert abs(got - 1.6206566095352173) < 1e-6, got


def test_optimize_bandwidth_multifeature_mid_spike():
    # Kills preprocessing.py:276:54 (`+ (...)` -> `- (...)` in the D>1 edge
    # branch): the D==1 fixture never executes that branch, so a two-feature
    # fixture with the spike in one feature is required. Same flip shape.
    from emergentia.preprocessing import AutoSmoother
    _, d2 = _spike_fixtures()
    got = AutoSmoother().optimize_bandwidth(d2, [])
    assert abs(got - 1.6206566095352173) < 1e-6, got


def test_gaussian_smoother_4d_returns_input_tensor():
    # Kills preprocessing.py:625:24 (is_tensor True->False): a >3-D tensor
    # input must come back as the same tensor; the mutant returns numpy.
    from emergentia.preprocessing import GaussianSmoother
    g = GaussianSmoother()
    x = torch.randn(2, 2, 2, 2)
    out = g(x)
    assert isinstance(out, torch.Tensor), type(out)
    assert out is x


def test_select_generalizing_max_term_multiplies_not_divides():
    # Kills engine.py:544:45 (* -> /): the dynamic-range term must grow with
    # max|y| (x), not shrink (/). gp has smaller amplitude and smaller max, so
    # the correct penalty prefers gp; the / mutant prefers the template.
    import sympy as sp
    from emergentia.engine import DiscoveryPipeline
    r = sp.Symbol("r")
    gp = sp.Float(0.001) + sp.Float(1e-12) * r
    tm = sp.Float(0.002) + sp.Float(1e-12) * r
    assert DiscoveryPipeline._select_generalizing(r, gp, tm).equals(gp)


def test_select_generalizing_unevaluable_gp_loses_to_template():
    # Kills engine.py:546:16 (except -> return None): an un-evaluable GP
    # candidate must take infinite penalty, not None (which raises TypeError
    # on comparison instead of selecting the valid template).
    import sympy as sp
    from emergentia.engine import DiscoveryPipeline
    r = sp.Symbol("r")
    bad_gp = sp.Function("undefined_fn")(r)
    tm = sp.Float(0.002) + sp.Float(1e-12) * r
    assert DiscoveryPipeline._select_generalizing(r, bad_gp, tm).equals(tm)


def test_validate_conservativeness_true_and_imaginary():
    # Kills engine.py:728:19 (return True -> False): r integrates to a
    # non-constant, imaginary-free potential, so it must validate True;
    # I*r carries an imaginary potential and must validate False.
    import sympy as sp
    from emergentia.engine import DiscoveryPipeline
    from emergentia.simulator import HarmonicPotential, PhysicsSim
    sim = PhysicsSim(n=2, dim=2, potential=HarmonicPotential(), seed=1,
                     device=torch.device("cpu"))
    pipe = DiscoveryPipeline(mode="spring", potential=sim.potential, seed=1,
                             device="cpu", enable_unit_checker=False,
                             enable_physics_priors=False,
                             enable_auto_smoother=False)
    r = sp.Symbol("r")
    assert pipe.validate_conservativeness(r) is True
    assert pipe.validate_conservativeness(sp.I * r) is False
