import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
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

class GNNEncoder(nn.Module):
    def __init__(self, node_features, hidden_dim, latent_dim, n_super_nodes):
        super(GNNEncoder, self).__init__()
        self.gnn1 = GNNLayer(node_features, hidden_dim)
        self.gnn2 = GNNLayer(hidden_dim, hidden_dim)
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        
        # Mapping to K super-nodes
        self.super_node_map = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * n_super_nodes)
        )

    def forward(self, x, edge_index, batch):
        x = torch.relu(self.gnn1(x, edge_index))
        x = torch.relu(self.gnn2(x, edge_index))
        
        # Instead of global pooling to one vector, we want to cluster/pool into K nodes
        # For simplicity, we can use a learned projection or global pooling + splitting
        # A more robust way is top-k pooling or assigned clusters, 
        # but here we'll use a fixed number of latent states derived from global info for now,
        # or average across partitions if we had them.
        # Let's use global pooling and then expand to K super-nodes for the "bottleneck".
        pooled = global_mean_pool(x, batch) # [batch_size, hidden_dim]
        latent = self.super_node_map(pooled) # [batch_size, latent_dim * n_super_nodes]
        return latent.view(-1, self.n_super_nodes, self.latent_dim)

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=64):
        super(LatentODEFunc, self).__init__()
        self.input_dim = latent_dim * n_super_nodes
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.input_dim)
        )

    def forward(self, t, y):
        # y: [batch_size, latent_dim * n_super_nodes]
        return self.net(y)

class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, n_super_nodes, hidden_dim, out_features, n_particles):
        super(GNNDecoder, self).__init__()
        self.n_particles = n_particles
        self.input_dim = latent_dim * n_super_nodes
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_particles * out_features)
        )

    def forward(self, z):
        # z: [batch_size, n_super_nodes, latent_dim]
        z_flat = z.view(z.size(0), -1)
        out = self.net(z_flat)
        return out.view(z.size(0), self.n_particles, -1)

class DiscoveryEngineModel(nn.Module):
    def __init__(self, n_particles, n_super_nodes, node_features=4, latent_dim=4, hidden_dim=64):
        super(DiscoveryEngineModel, self).__init__()
        self.encoder = GNNEncoder(node_features, hidden_dim, latent_dim, n_super_nodes)
        self.ode_func = LatentODEFunc(latent_dim, n_super_nodes, hidden_dim)
        self.decoder = GNNDecoder(latent_dim, n_super_nodes, hidden_dim, node_features, n_particles)
        
    def encode(self, x, edge_index, batch):
        return self.encoder(x, edge_index, batch)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward_dynamics(self, z0, t):
        # z0: [batch_size, n_super_nodes, latent_dim]
        z0_flat = z0.view(z0.size(0), -1)
        zt_flat = odeint(self.ode_func, z0_flat, t)
        # zt_flat: [len(t), batch_size, latent_dim * n_super_nodes]
        return zt_flat.view(zt_flat.size(0), zt_flat.size(1), -1, z0.size(-1))
