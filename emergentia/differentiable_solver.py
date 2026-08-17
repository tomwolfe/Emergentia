import torch
import torch.nn as nn
from torchdiffeq import odeint

class ODEFunc(nn.Module):
    def __init__(self, model, mass=1.0, dim=2, n_particles=3):
        super().__init__()
        self.model = model
        self.mass = mass
        self.dim = dim
        self.n_particles = n_particles

    def forward(self, t, state):
        batch_size = state.shape[0]
        n_dim = state.shape[1] // 2
        
        pos = state[:, :n_dim]
        vel = state[:, n_dim:]
        
        pos_reshaped = pos.view(batch_size, self.n_particles, self.dim)
        
        # Compute forces
        forces = self.model(pos_reshaped)
        accel = forces / self.mass
        
        return torch.cat([vel, accel.view(batch_size, -1)], dim=-1)

class DifferentiableSimulator(nn.Module):
    def __init__(self, model, mass=1.0, method='rk4', dim=2, n_particles=3):
        super().__init__()
        self.odefunc = ODEFunc(model, mass, dim, n_particles)
        self.method = method

    def forward(self, x0, t):
        return odeint(self.odefunc, x0, t, method=self.method)
