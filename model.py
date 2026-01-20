import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torchdiffeq import odeint

class GNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        tmp = torch.cat([x_i, x_j], dim=1)
        return self.mlp(tmp)

class HierarchicalPooling(nn.Module):
    """
    Learned soft-assignment pooling to preserve spatial locality 
    by aggregating nodes into a fixed number of super-nodes.
    Uses Gumbel-Softmax for 'harder' assignments.
    """
    def __init__(self, in_channels, n_super_nodes):
        super(HierarchicalPooling, self).__init__()
        self.n_super_nodes = n_super_nodes
        self.assign_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, n_super_nodes)
        )

    def forward(self, x, batch, tau=1.0, hard=False):
        # x: [N, in_channels], batch: [N]
        # Compute assignment logits
        logits = self.assign_mlp(x)
        
        # Use Gumbel-Softmax for harder, more distinct assignments
        s = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        
        # 1. Sharpness Loss (Minimize entropy per node)
        entropy = -torch.mean(torch.sum(s * torch.log(s + 1e-9), dim=1))
        
        # 2. Diversity Loss (Maximize entropy of average assignment)
        avg_s = s.mean(dim=0)
        diversity_loss = torch.sum(avg_s * torch.log(avg_s + 1e-9))
        
        # Total assignment loss
        assign_loss = entropy + diversity_loss
        
        # Weighted aggregation: [N, 1, in_channels] * [N, n_super_nodes, 1]
        x_expanded = x.unsqueeze(1) * s.unsqueeze(2) 
        
        # Scatter to batch: [Batch_Size, n_super_nodes, in_channels]
        out = scatter(x_expanded, batch, dim=0, reduce='sum')
        return out, s, assign_loss

class GNNEncoder(nn.Module):
    def __init__(self, node_features, hidden_dim, latent_dim, n_super_nodes):
        super(GNNEncoder, self).__init__()
        self.gnn1 = GNNLayer(node_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        
        # Spatially-aware pooling instead of global_mean_pool
        self.pooling = HierarchicalPooling(hidden_dim, n_super_nodes)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, edge_index, batch, tau=1.0):
        x = self.ln1(torch.relu(self.gnn1(x, edge_index)))
        x = self.ln2(torch.relu(self.gnn2(x, edge_index)))
        
        # Pool to K super-nodes preserving spatial features
        pooled, s, entropy = self.pooling(x, batch, tau=tau) # [batch_size, n_super_nodes, hidden_dim], [N, n_super_nodes]
        latent = self.output_layer(pooled) # [batch_size, n_super_nodes, latent_dim]
        return latent, s, entropy

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=64):
        super(LatentODEFunc, self).__init__()
        self.input_dim = latent_dim * n_super_nodes
        self.latent_dim = latent_dim
        self.n_super_nodes = n_super_nodes
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, self.input_dim)
        )

    def forward(self, t, y):
        # y: [batch_size, latent_dim * n_super_nodes]
        # t is a scalar or [1], ignored for autonomous dynamics
        return self.net(y)

class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_features):
        super(GNNDecoder, self).__init__()
        self.ln = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features)
        )

    def forward(self, z, s, batch):
        # z: [batch_size, n_super_nodes, latent_dim]
        # s: [N_total, n_super_nodes]
        
        # We need to map z back to nodes using s.
        # Since s is global for the batch, we use batch index to select z
        # batch: [N_total] contains indices of which batch item each node belongs to.
        
        # Expand z to match s: [N_total, n_super_nodes, latent_dim]
        z_expanded = z[batch]
        
        # Weighted sum: [N_total, n_super_nodes, 1] * [N_total, n_super_nodes, latent_dim]
        # sum over n_super_nodes
        node_features_latent = torch.sum(s.unsqueeze(-1) * z_expanded, dim=1)
        
        # Apply LayerNorm before final output layer (inside the MLP's first part effectively)
        # But instructions said "before the final output", let's be precise.
        h = torch.relu(self.mlp[0](node_features_latent))
        h = self.ln(h)
        return self.mlp[2](h)

class DiscoveryEngineModel(nn.Module):
    def __init__(self, n_particles, n_super_nodes, node_features=4, latent_dim=4, hidden_dim=64):
        super(DiscoveryEngineModel, self).__init__()
        self.encoder = GNNEncoder(node_features, hidden_dim, latent_dim, n_super_nodes)
        self.ode_func = LatentODEFunc(latent_dim, n_super_nodes, hidden_dim)
        self.decoder = GNNDecoder(latent_dim, hidden_dim, node_features)
        
    def encode(self, x, edge_index, batch, tau=1.0):
        return self.encoder(x, edge_index, batch, tau=tau)
    
    def decode(self, z, s, batch):
        return self.decoder(z, s, batch)
    
    def get_ortho_loss(self, s):
        # s: [N, n_super_nodes]
        # Encourage orthogonality between super-node assignment vectors
        # s^T * s should be close to identity (weighted by number of particles)
        dots = torch.matmul(s.t(), s)
        identity = torch.eye(s.size(1), device=s.device) * (s.size(0) / s.size(1))
        return torch.mean((dots - identity)**2)
    
    def forward_dynamics(self, z0, t):
        # z0: [batch_size, n_super_nodes, latent_dim]
        z0_flat = z0.view(z0.size(0), -1).to(torch.float32)
        t = t.to(torch.float32)

        # Use the device of the ode_func parameters (may be CPU even if z0 is MPS)
        ode_device = next(self.ode_func.parameters()).device
        original_device = z0.device
        
        y0 = z0_flat.to(ode_device)
        t_ode = t.to(ode_device)

        # Explicitly use float32 for rtol/atol to avoid issues on some devices
        rtol = torch.tensor(1e-3, dtype=torch.float32, device=ode_device)
        atol = torch.tensor(1e-4, dtype=torch.float32, device=ode_device)

        zt_flat = odeint(self.ode_func, y0, t_ode, rtol=rtol, atol=atol)

        # Move back to original device
        zt_flat = zt_flat.to(original_device)

        # zt_flat: [len(t), batch_size, latent_dim * n_super_nodes]
        return zt_flat.view(zt_flat.size(0), zt_flat.size(1), -1, z0.size(-1))
