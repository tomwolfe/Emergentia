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
        self.scaling = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, batch, pos=None, tau=1.0, hard=False):
        # x: [N, in_channels], batch: [N], pos: [N, 2]
        # Compute assignment logits with scaling for sharpness
        logits = self.assign_mlp(x) * self.scaling
        
        # Use Gumbel-Softmax for harder, more distinct assignments
        s = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        
        # 1. Sharpness Loss (Minimize entropy per node)
        entropy = -torch.mean(torch.sum(s * torch.log(s + 1e-9), dim=1))
        
        # 2. Diversity Loss (Maximize entropy of average assignment)
        avg_s = s.mean(dim=0)
        diversity_loss = torch.sum(avg_s * torch.log(avg_s + 1e-9))
        
        # 2b. Pruning/Sparsity Loss: Encourage some super-nodes to have near-zero average assignment
        # This allows the model to 'turn off' unused super-nodes.
        # We use an L1-style penalty on the average assignment of each super-node.
        pruning_loss = torch.mean(torch.abs(avg_s))

        # 3. Spatial Locality Penalty
        spatial_loss = torch.tensor(0.0, device=x.device)
        if pos is not None:
            # For each super-node, calculate weighted variance of positions
            # s: [N, K], pos: [N, 2]
            s_sum = s.sum(dim=0, keepdim=True) + 1e-9
            s_norm = s / s_sum # [N, K]
            
            # Weighted mean position for each super-node
            mu = torch.matmul(s_norm.t(), pos) # [K, 2]
            
            # Weighted variance: sum_i s_norm[i, k] * ||pos[i] - mu[k]||^2
            # (pos[i] - mu[k])^2 = pos[i]^2 - 2*pos[i]*mu[k] + mu[k]^2
            pos_sq = (pos**2).sum(dim=1, keepdim=True) # [N, 1]
            mu_sq = (mu**2).sum(dim=1) # [K]
            
            var = torch.matmul(s_norm.t(), pos_sq).squeeze() - 2 * (mu * torch.matmul(s_norm.t(), pos)).sum(dim=1) + mu_sq
            spatial_loss = var.mean()

        # Total assignment loss components
        assign_losses = {
            'entropy': entropy,
            'diversity': diversity_loss,
            'spatial': spatial_loss,
            'pruning': pruning_loss
        }
        
        # Calculate mu (weighted mean position) for super-nodes
        super_node_mu = None
        if pos is not None:
            # s: [N, K], pos: [N, 2]
            # out_mu: [batch_size, K, 2]
            s_pos_expanded = pos.unsqueeze(1) * s.unsqueeze(2) # [N, K, 2]
            sum_s_pos = scatter(s_pos_expanded, batch, dim=0, reduce='sum') # [B, K, 2]
            sum_s = scatter(s, batch, dim=0, reduce='sum').unsqueeze(-1) + 1e-9 # [B, K, 1]
            super_node_mu = sum_s_pos / sum_s

        # 4. Spatial Separation Loss (Centroids should be spread out)
        if super_node_mu is not None:
            # mu: [B, K, 2]. We want to maximize distance between centroids.
            # Using a simple repulsive potential: 1 / (dist^2 + eps)
            mu = super_node_mu
            dist_sq = torch.sum((mu.unsqueeze(1) - mu.unsqueeze(2))**2, dim=-1) # [B, K, K]
            # Mask out diagonal
            mask = torch.eye(self.n_super_nodes, device=x.device).unsqueeze(0)
            repulsion = 1.0 / (dist_sq + 1.0)
            separation_loss = (repulsion * (1 - mask)).sum() / (self.n_super_nodes * (self.n_super_nodes - 1) + 1e-9)
            assign_losses['separation'] = separation_loss

        # Weighted aggregation: [N, 1, in_channels] * [N, n_super_nodes, 1]
        x_expanded = x.unsqueeze(1) * s.unsqueeze(2) 
        
        # Scatter to batch: [Batch_Size, n_super_nodes, in_channels]
        out = scatter(x_expanded, batch, dim=0, reduce='sum')
        
        return out, s, assign_losses, super_node_mu

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
        # Store initial positions for spatial pooling
        pos = x[:, :2] 
        
        x = self.ln1(torch.relu(self.gnn1(x, edge_index)))
        x = self.ln2(torch.relu(self.gnn2(x, edge_index)))
        
        # Pool to K super-nodes preserving spatial features
        pooled, s, assign_losses, mu = self.pooling(x, batch, pos=pos, tau=tau) # [batch_size, n_super_nodes, hidden_dim], [N, n_super_nodes], mu: [B, K, 2]
        latent = self.output_layer(pooled) # [batch_size, n_super_nodes, latent_dim]
        return latent, s, assign_losses, mu

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
        return self.net(y)

class HamiltonianODEFunc(nn.Module):
    """
    Enforces Hamiltonian constraints: dq/dt = dH/dp, dp/dt = -dH/dq.
    Optionally includes a learnable dissipation term: dp/dt = -dH/dq - gamma * p.
    """
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=64, dissipative=True):
        super(HamiltonianODEFunc, self).__init__()
        assert latent_dim % 2 == 0, "Latent dim must be even for Hamiltonian dynamics (q, p)"
        self.latent_dim = latent_dim
        self.n_super_nodes = n_super_nodes
        self.dissipative = dissipative
        
        self.H_net = nn.Sequential(
            nn.Linear(latent_dim * n_super_nodes, hidden_dim),
            nn.Tanh(), # Tanh ensures second-order differentiability for dH
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        if dissipative:
            # Small initial dissipation
            self.gamma = nn.Parameter(torch.full((n_super_nodes, 1), -5.0)) # log space

    def forward(self, t, y):
        # y: [batch_size, latent_dim * n_super_nodes]
        with torch.set_grad_enabled(True):
            y = y.requires_grad_(True)
            H = self.H_net(y).sum()
            dH = torch.autograd.grad(H, y, create_graph=True)[0]
        
        # dH is [batch_size, n_super_nodes * latent_dim]
        # Reshape to [batch_size, n_super_nodes, 2, latent_dim // 2]
        d_sub = self.latent_dim // 2
        dH_view = dH.view(-1, self.n_super_nodes, 2, d_sub)
        
        dq = dH_view[:, :, 1]  # dH/dp
        dp = -dH_view[:, :, 0] # -dH/dq
        
        if self.dissipative:
            # y: [B, K * D] -> [B, K, 2, D/2]
            y_view = y.view(-1, self.n_super_nodes, 2, d_sub)
            p = y_view[:, :, 1] # momentum
            gamma = torch.exp(self.gamma)
            dp = dp - gamma * p
        
        return torch.cat([dq, dp], dim=-1).view(y.shape[0], -1)

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
        
        # Expand z to match s: [N_total, n_super_nodes, latent_dim]
        z_expanded = z[batch]
        
        # Weighted sum: [N_total, n_super_nodes, 1] * [N_total, n_super_nodes, latent_dim]
        node_features_latent = torch.sum(s.unsqueeze(-1) * z_expanded, dim=1)
        
        h = torch.relu(self.mlp[0](node_features_latent))
        h = self.ln(h)
        return self.mlp[2](h)

class DiscoveryEngineModel(nn.Module):
    def __init__(self, n_particles, n_super_nodes, node_features=4, latent_dim=4, hidden_dim=64, hamiltonian=False, dissipative=True):
        super(DiscoveryEngineModel, self).__init__()
        self.encoder = GNNEncoder(node_features, hidden_dim, latent_dim, n_super_nodes)
        
        if hamiltonian:
            self.ode_func = HamiltonianODEFunc(latent_dim, n_super_nodes, hidden_dim, dissipative=dissipative)
        else:
            self.ode_func = LatentODEFunc(latent_dim, n_super_nodes, hidden_dim)
            
        self.decoder = GNNDecoder(latent_dim, hidden_dim, node_features)
        self.hamiltonian = hamiltonian

        # Learnable loss log-variances for automatic loss balancing
        # 0: rec, 1: cons, 2: assign, 3: ortho, 4: l2, 5: lvr, 6: align, 7: pruning, 8: sep, 9: conn
        self.log_vars = nn.Parameter(torch.zeros(10)) 
        
    def encode(self, x, edge_index, batch, tau=1.0):
        return self.encoder(x, edge_index, batch, tau=tau)
    
    def decode(self, z, s, batch):
        return self.decoder(z, s, batch)
    
    def get_ortho_loss(self, s):
        # s: [N, n_super_nodes]
        # Encourage orthogonality between super-node assignment vectors
        # s^T * s should be close to identity (weighted by number of particles)
        n_nodes, k = s.shape
        dots = torch.matmul(s.t(), s)
        # Scaling identity by N/K ensures that if all nodes are perfectly split, loss is 0
        identity = torch.eye(k, device=s.device).mul_(n_nodes / k)
        return torch.mean((dots - identity)**2)
    
    def get_connectivity_loss(self, s, edge_index):
        """
        Encourages nodes assigned to the same super-node to be connected.
        Minimizes (s_i - s_j)^2 for connected nodes (i, j).
        """
        row, col = edge_index
        s_i = s[row]
        s_j = s[col]
        return torch.mean((s_i - s_j)**2)
    
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
