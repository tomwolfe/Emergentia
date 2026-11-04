"""Shape, gradient-flow, and mass-scaling contract for the differentiable path."""

import torch

from emergentia.differentiable_solver import DifferentiableSimulator, ODEFunc
from emergentia.models import DiscoveryNet


def _make_sim(n=3, dim=2, mass=1.0):
    model = DiscoveryNet()
    return DifferentiableSimulator(model, dim=dim, n_particles=n, mass=mass), model, n, dim


def test_state_shape_and_forward():
    sim, _, n, dim = _make_sim()
    batch = 4
    state_dim = 2 * n * dim
    x0 = torch.randn(batch, state_dim)
    t = torch.linspace(0.0, 0.01, 3)
    out = sim(x0, t)
    assert out.shape == (3, batch, state_dim)


def test_backward_grad_flows():
    sim, model, n, dim = _make_sim()
    batch = 2
    state_dim = 2 * n * dim
    x0 = torch.randn(batch, state_dim)
    t = torch.linspace(0.0, 0.005, 2)
    out = sim(x0, t)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients flowed into the model"


def test_mass_scaling_halves_accel():
    _, model, n, dim = _make_sim()
    f1 = ODEFunc(model, mass=1.0, dim=dim, n_particles=n)
    f2 = ODEFunc(model, mass=2.0, dim=dim, n_particles=n)
    torch.manual_seed(0)
    state = torch.randn(2, 2 * n * dim)
    t = torch.tensor(0.0)
    d1 = f1(t, state)
    d2 = f2(t, state)
    pos_dim = n * dim
    # Velocities identical; accelerations halve with doubled mass.
    assert torch.allclose(d1[:, :pos_dim], d2[:, :pos_dim])
    assert torch.allclose(d1[:, pos_dim:] / 2.0, d2[:, pos_dim:], atol=1e-5)
