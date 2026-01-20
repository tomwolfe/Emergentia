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
    """
    def __init__(self, in_channels, n_super_nodes):
        super(HierarchicalPooling, self).__init__()
        self.n_super_nodes = n_super_nodes
        self.assign_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, n_super_nodes)
        )

    def forward(self, x, batch):
        # x: [N, in_channels], batch: [N]
        # Compute soft-assignment matrix s: [N, n_super_nodes]
        s = torch.softmax(self.assign_mlp(x), dim=-1)
        
        # Weighted aggregation: [N, 1, in_channels] * [N, n_super_nodes, 1]
        x_expanded = x.unsqueeze(1) * s.unsqueeze(2) 
        
        # Scatter to batch: [Batch_Size, n_super_nodes, in_channels]
        out = scatter(x_expanded, batch, dim=0, reduce='sum')
        return out, s

class GNNEncoder(nn.Module):
    def __init__(self, node_features, hidden_dim, latent_dim, n_super_nodes):
        super(GNNEncoder, self).__init__()
        self.gnn1 = GNNLayer(node_features, hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, hidden_dim)
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        
        # Spatially-aware pooling instead of global_mean_pool
        self.pooling = HierarchicalPooling(hidden_dim, n_super_nodes)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.gnn1(x, edge_index))
        x = torch.relu(self.gnn2(x, edge_index))
        
        # Pool to K super-nodes preserving spatial features
        pooled, s = self.pooling(x, batch) # [batch_size, n_super_nodes, hidden_dim], [N, n_super_nodes]
        latent = self.output_layer(pooled) # [batch_size, n_super_nodes, latent_dim]
        return latent, s

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=64):
        super(LatentODEFunc, self).__init__()
        self.input_dim = latent_dim * n_super_nodes
        self.net = nn.Sequential(
            nn.Linear(self.input_dim + 1, hidden_dim), # +1 for time t
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.input_dim)
        )

    def forward(self, t, y):
        # y: [batch_size, latent_dim * n_super_nodes]
        # t is a scalar or [1]
        t_vec = torch.ones((y.size(0), 1), device=y.device) * t
        y_t = torch.cat([y, t_vec], dim=-1)
        return self.net(y_t)

class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_features):
        super(GNNDecoder, self).__init__()
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
        
        return self.mlp(node_features_latent)

class DiscoveryEngineModel(nn.Module):
    def __init__(self, n_particles, n_super_nodes, node_features=4, latent_dim=4, hidden_dim=64):
        super(DiscoveryEngineModel, self).__init__()
        self.encoder = GNNEncoder(node_features, hidden_dim, latent_dim, n_super_nodes)
        self.ode_func = LatentODEFunc(latent_dim, n_super_nodes, hidden_dim)
        self.decoder = GNNDecoder(latent_dim, hidden_dim, node_features)
        
    def encode(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)
    
    def decode(self, z, s, batch):
        return self.decoder(z, s, batch)
    
    def forward_dynamics(self, z0, t):
        # z0: [batch_size, n_super_nodes, latent_dim]
        z0_flat = z0.view(z0.size(0), -1)
        zt_flat = odeint(self.ode_func, z0_flat, t)
        # zt_flat: [len(t), batch_size, latent_dim * n_super_nodes]
        return zt_flat.view(zt_flat.size(0), zt_flat.size(1), -1, z0.size(-1))
