import torch
import torch.nn as nn

class InvariantLayer(nn.Module):
    """Projects coordinates into invariant radial distances."""
    def forward(self, pos):
        # pos: (batch, n, dim)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1) # (batch, n, n, dim)
        dist = torch.norm(diff, dim=-1, keepdim=True) # (batch, n, n, 1)
        return dist, diff

class ConservativeForceField(nn.Module):
    """
    Ensures energy conservation by predicting a potential V(r) 
     and deriving forces via F = -grad(V).
    """
    def __init__(self, potential_net):
        super().__init__()
        self.projection = InvariantLayer()
        self.potential_net = potential_net

    def forward(self, pos):
        # Enable gradient tracking for pos to compute forces
        pos = pos.requires_grad_(True)
        dist, diff = self.projection(pos)
        
        # Mask out self-interaction for potential
        mask = (~torch.eye(pos.shape[1], device=pos.device).bool()).unsqueeze(0).unsqueeze(-1)
        
        # Predict pairwise potentials
        # We only need upper triangle for total energy, or we can just sum all and divide by 2
        v_pair = self.potential_net(dist) * mask
        total_energy = torch.sum(v_pair) * 0.5
        
        # Force is -grad(Energy)
        forces = -torch.autograd.grad(total_energy, pos, create_graph=True)[0]
        return forces

    def predict_mag(self, r):
        # For symbolic distillation, we need the magnitude of the force
        # F(r) = -dV/dr
        r = r.requires_grad_(True)
        v = self.potential_net(r)
        dv_dr = torch.autograd.grad(v.sum(), r, create_graph=True)[0]
        return -dv_dr
