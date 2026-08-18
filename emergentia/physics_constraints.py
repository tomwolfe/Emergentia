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

    def _pairwise_potential(self, dist):
        """Compute pairwise potentials from distances using the inner network."""
        feat = self.potential_net._get_features(dist)
        return self.potential_net.net(feat)

    def forward(self, pos):
        if not pos.requires_grad:
            pos = pos.clone().requires_grad_(True)
        dist, diff = self.projection(pos)
        
        mask = (~torch.eye(pos.shape[1], device=pos.device).bool()).unsqueeze(0).unsqueeze(-1)
        
        v_pair = self._pairwise_potential(dist) * mask
        total_energy = torch.sum(v_pair) * 0.5
        
        forces = -torch.autograd.grad(total_energy, pos, create_graph=True)[0]
        return forces

    def predict_mag(self, r):
        """Predict force magnitude F(r) = -dV/dr for 1D symbolic distillation.
        
        Args:
            r: Tensor of shape (num_points, 1)
        """
        r = r.view(-1, 1)
        if not r.requires_grad:
            r = r.clone().requires_grad_(True)
        v = self._pairwise_potential(r)
        dv_dr = torch.autograd.grad(v.sum(), r, create_graph=True)[0]
        return -dv_dr

    def get_potential_energy(self, pos):
        """Compute scalar total potential energy for logging/conservation checks.
        
        Args:
            pos: Tensor of shape (batch, n_particles, dim)
        Returns:
            Scalar total potential energy
        """
        if not pos.requires_grad:
            pos = pos.clone().requires_grad_(True)
        dist, diff = self.projection(pos)
        mask = (~torch.eye(pos.shape[1], device=pos.device).bool()).unsqueeze(0).unsqueeze(-1)
        v_pair = self._pairwise_potential(dist) * mask
        return torch.sum(v_pair) * 0.5
