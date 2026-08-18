import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    """
    ODE function for differentiable simulation.

    Given state ``s = (x, v)`` of shape (batch, 2 * n_particles * dim), this
    computes the time-derivative ``ds/dt = (v, F/m)`` where ``F`` is the
    force predicted by the supplied ``DiscoveryNet`` (or compatible) model.

    Shape contract:
      - t: scalar tensor (current time)
      - state: (batch, state_dim) where state_dim = 2 * n_particles * dim
      - returns: (batch, state_dim)
    """

    def __init__(self, model, mass=1.0, dim=2, n_particles=3):
        super().__init__()
        self.model = model
        self.mass = mass
        self.dim = dim
        self.n_particles = n_particles

    def forward(self, t, state):
        assert state.dim() == 2, (
            f"Expected state of shape (batch, state_dim), got {tuple(state.shape)}"
        )
        state_dim = state.shape[1]
        expected = 2 * self.n_particles * self.dim
        assert state_dim == expected, (
            f"Expected state_dim={expected}, got {state_dim}"
        )

        batch_size = state.shape[0]
        pos_dim = self.n_particles * self.dim

        pos = state[:, :pos_dim]
        vel = state[:, pos_dim:]
        pos_reshaped = pos.view(batch_size, self.n_particles, self.dim)

        # Pairwise geometry (detached so gradients only flow through |F| network)
        diff = pos_reshaped.unsqueeze(2) - pos_reshaped.unsqueeze(1)  # (B, n, n, D)
        dist = torch.norm(diff, dim=-1, keepdim=True)  # (B, n, n, 1)
        dist_safe = torch.clamp(dist, min=0.1, max=50.0)

        # Direction vectors — no gradient needed through geometry
        dist_detached = dist.detach()
        dir_vec = diff.detach() / torch.clamp(dist_detached, min=1e-6)

        # Compute force magnitude F(r) = -dV/dr via first-order autograd on 1-D distances
        # This avoids the nested (second-order) autograd.grad inside DiscoveryNet.forward
        # that destabilises gradients through the ODE solver.
        dist_flat = dist_safe.detach().view(-1, 1)  # (B*n*n, 1)
        model_dtype = next(self.model.parameters()).dtype
        dist_model = dist_flat.to(model_dtype)
        dist_model = dist_model.clone().requires_grad_(True)

        v = self.model.net(dist_model)  # potential values (B*n*n, 1)

        dv_dr = torch.autograd.grad(
            v.sum(), dist_model, create_graph=True, retain_graph=True
        )[0]
        f_mag = -dv_dr  # (|B*n*n|, 1)
        # Cast back to state dtype so the ODE solver gets consistent dtypes
        f_mag = f_mag.to(dist_flat.dtype)

        f_mag = torch.nan_to_num(f_mag, nan=0.0, posinf=1e3, neginf=-1e3)
        f_mag = f_mag.view(batch_size, self.n_particles, self.n_particles, 1)

        # Mask self-interaction
        mask = (
            (~torch.eye(self.n_particles, device=pos_reshaped.device).bool())
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        # Force on particle i = sum_j F_mag[i,j] * r_hat[i,j]
        forces = (f_mag * dir_vec * mask).sum(dim=2)  # (B, n, D)

        accel = forces / self.mass
        return torch.cat([vel, accel.view(batch_size, -1)], dim=-1)


class DifferentiableSimulator(nn.Module):
    def __init__(self, model, mass=1.0, method='rk4', dim=2, n_particles=3):
        super().__init__()
        self.odefunc = ODEFunc(model, mass, dim, n_particles)
        self.method = method

    def forward(self, x0, t, method=None, atol=None, rtol=None):
        kwargs = {}
        if method is not None:
            kwargs["method"] = method
        else:
            kwargs["method"] = self.method
        if atol is not None:
            kwargs["atol"] = atol
        if rtol is not None:
            kwargs["rtol"] = rtol
        return odeint(self.odefunc, x0, t, **kwargs)
