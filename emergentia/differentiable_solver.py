import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, model, mass=1.0):
        super().__init__()
        self.model = model
        self.mass = mass

    def forward(self, t, state):
        # state: (batch, 2 * n_particles * dim)
        # We need to reshape state to (batch, n_particles, dim) for pos and vel
        # Assuming 2D for now as per current simulator, but let's be general if possible.
        # Actually, let's assume the state passed to odeint is (batch, 2, n_particles, dim)
        # or flat. Let's use flat for simplicity with odeint.
        
        # Let's say state is (batch, 2 * N * D)
        batch_size = state.shape[0]
        n_dim = state.shape[1] // 2
        
        pos = state[:, :n_dim]
        vel = state[:, n_dim:]
        
        # Reshape for DiscoveryNet: (batch, n_particles, dim)
        # We need to know n_particles and dim.
        # Let's assume dim=2 and n_particles = n_dim // 2
        dim = 2
        n_particles = n_dim // dim
        
        pos_reshaped = pos.view(batch_size, n_particles, dim)
        
        # Compute forces
        forces = self.model(pos_reshaped) # (batch, n_particles, dim)
        accel = forces / self.mass
        
        # Return [vel, accel] flat
        return torch.cat([vel, accel.view(batch_size, -1)], dim=-1)

class DifferentiableSimulator(nn.Module):
    def __init__(self, model, mass=1.0, method='rk4'):
        super().__init__()
        self.odefunc = ODEFunc(model, mass)
        self.method = method

    def forward(self, x0, t):
        # x0: initial state (batch, 2 * N * D)
        # t: time points to evaluate (T,)
        return odeint(self.odefunc, x0, t, method=self.method)
