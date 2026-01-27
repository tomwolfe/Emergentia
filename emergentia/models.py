import torch
import torch.nn as nn

class TrajectoryScaler:
    def __init__(self, mode='spring'):
        self.mode = mode
        self.p_scale = 1.0
        self.f_scale = 1.0
        
    def fit(self, p, f):
        self.p_scale = torch.max(torch.abs(p)).item() if torch.max(torch.abs(p)).item() > 0 else 1.0
        self.f_scale = torch.max(torch.abs(f)).item() if torch.max(torch.abs(f)).item() > 0 else 1.0
        
    def transform(self, p, f): 
        return p / self.p_scale, f / self.f_scale
    
    def inverse_transform_f(self, f_scaled): 
        return f_scaled * self.f_scale

class DiscoveryNet(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        # Input features: [r, 1/r] - Zero Hinting
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )

    def _get_features(self, dist):
        # Only distance and its inverse. No high-power exponents.
        dist_safe = torch.clamp(dist, min=1e-4, max=50.0)
        inv_r = 1.0 / dist_safe
        return torch.cat([dist_safe, inv_r], dim=-1)

    def forward(self, pos_scaled):
        # pos_scaled: (batch, n_particles, 2)
        diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1) # (batch, n, n, 2)
        dist = torch.norm(diff, dim=-1, keepdim=True) # (batch, n, n, 1)
        
        feat = self._get_features(dist) # (batch, n, n, 2)
        mag = self.net(feat) # (batch, n, n, 1)
        
        # Mask out self-interaction
        mask = (~torch.eye(pos_scaled.shape[1], device=pos_scaled.device).bool()).unsqueeze(0).unsqueeze(-1)
        
        # F_ij = mag_ij * (r_i - r_j) / |r_i - r_j|
        pair_forces = mag * (diff / torch.clamp(dist, min=1e-6)) * mask
        
        # Net force on each particle: F_i = sum_j F_ij
        return torch.sum(pair_forces, dim=2)

    def predict_mag(self, r_scaled):
        # r_scaled: (num_points, 1)
        return self.net(self._get_features(r_scaled))
