"""
Optimized Neural-Symbolic Discovery Pipeline - Model Components

This module implements performance-optimized versions of the core neural network architectures
for the coarse-graining of particle dynamics using Graph Neural Networks (GNNs)
and Differentiable Physics (Latent ODEs).

Key optimizations:
- Efficient ODE solvers with adaptive precision
- Cached edge computations
- Reduced symbolic regression complexity
- Memory-efficient operations
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torchdiffeq import odeint, odeint_adjoint
from stable_pooling import StableHierarchicalPooling


class OptimizedEquivariantGNNLayer(MessagePassing):
    """
    Optimized E(n)-equivariant GNN Layer with reduced computational overhead.
    Uses simplified message passing and fewer parameters.
    """
    def __init__(self, in_channels, out_channels, hidden_dim=32, box_size=None):  # Added box_size
        super(OptimizedEquivariantGNNLayer, self).__init__(aggr='add')
        self.box_size = box_size
        # Reduced parameter count
        self.phi_e = nn.Sequential(
            nn.Linear(2 * in_channels + 2, hidden_dim), # +2 for dist_sq and dot(v, r)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim//2)  # Reduced output
        )
        # Simplified vector message network
        self.phi_v = nn.Sequential(
            nn.Linear(2 * in_channels + 2, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        # Node update network with fewer parameters
        self.phi_h = nn.Sequential(
            nn.Linear(in_channels + hidden_dim//2 + 1, hidden_dim//2), # +1 for vector message norm
            nn.SiLU(),
            nn.Linear(hidden_dim//2, out_channels)
        )
        
        # Shortcut for residual connection when dims don't match
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, pos, vel, edge_index):
        # x: [N, in_channels], pos: [N, 2], vel: [N, 2]
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        
        # Apply Minimum Image Convention for PBC
        if self.box_size is not None:
            # We assume box_size is a tensor of shape [2] or [1, 2]
            # If pos is normalized to [-1, 1], box_size in normalized units is 2.0
            # BUT we should probably pass the box_size in the same units as pos.
            # In prepare_data, pos is normalized to [-1, 1] using stats.
            # So the effective box_size in normalized units is (original_box_size / stats['pos_range']) * 2.0
            # Actually, it's easier to just assume the box is mapped to [-1, 1] exactly if pos covers the whole box.
            # If stats['pos_range'] is the box size, then effective_box = 2.0.
            
            # For now, let's assume box_size passed here is in normalized units.
            # In most cases, it will be 2.0 if the particles cover the whole box.
            if isinstance(self.box_size, (float, int)):
                box = torch.tensor([self.box_size, self.box_size], device=pos.device, dtype=pos.dtype)
            else:
                box = torch.as_tensor(self.box_size, device=pos.device, dtype=pos.dtype)
            
            rel_pos = rel_pos - box * torch.round(rel_pos / box)

        rel_vel = vel[edge_index[0]] - vel[edge_index[1]]

        dist_sq = torch.sum(rel_pos**2, dim=-1, keepdim=True)
        dot_vr = torch.sum(rel_vel * rel_pos, dim=-1, keepdim=True)

        # Propagate messages
        m_h, m_v = self.propagate(edge_index, x=x, dist_sq=dist_sq, dot_vr=dot_vr, rel_pos=rel_pos)

        # m_v is a vector [N, 2]. We take its norm as a scalar feature
        m_v_norm = torch.norm(m_v, dim=-1, keepdim=True)

        # Update node features
        h_update = self.phi_h(torch.cat([x, m_h, m_v_norm], dim=-1))

        return self.shortcut(x) + h_update

    def message(self, x_i, x_j, dist_sq, dot_vr, rel_pos):
        # Scalar message
        tmp = torch.cat([x_i, x_j, dist_sq, dot_vr], dim=1)
        m_h = self.phi_e(tmp)

        # Vector message weight
        v_w = self.phi_v(tmp)
        m_v = v_w * rel_pos

        return m_h, m_v

    def aggregate(self, inputs, index, dim_size=None):
        # Custom aggregate to handle tuple of (scalar_msg, vector_msg)
        m_h_in, m_v_in = inputs
        m_h = scatter(m_h_in, index, dim=self.node_dim, dim_size=dim_size, reduce='sum').to(torch.float32)
        m_v = scatter(m_v_in, index, dim=self.node_dim, dim_size=dim_size, reduce='sum').to(torch.float32)
        return m_h, m_v


class OptimizedGNNEncoder(nn.Module):
    def __init__(self, node_features, hidden_dim, latent_dim, n_super_nodes, min_active_super_nodes=2, box_size=None):
        super(OptimizedGNNEncoder, self).__init__()
        self.box_size = box_size
        # Use optimized layers with fewer parameters
        self.gnn1 = OptimizedEquivariantGNNLayer(node_features, hidden_dim, hidden_dim//2, box_size=box_size)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gnn2 = OptimizedEquivariantGNNLayer(hidden_dim, hidden_dim, hidden_dim//2, box_size=box_size)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim

        # Use the Enhanced StableHierarchicalPooling
        self.pooling = StableHierarchicalPooling(hidden_dim, n_super_nodes, min_active_super_nodes=min_active_super_nodes)

        # Simplified output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, latent_dim)
        )

    def forward(self, x, edge_index, batch, tau=1.0, hard=False):
        # Store initial positions for spatial pooling
        pos = x[:, :2]
        vel = x[:, 2:4] # Assume [pos_x, pos_y, vel_x, vel_y]

        x = self.ln1(torch.relu(self.gnn1(x, pos, vel, edge_index)))
        x = self.ln2(torch.relu(self.gnn2(x, pos, vel, edge_index)))

        # Pool to K super-nodes preserving spatial features
        pooled, s, assign_losses, mu = self.pooling(x, batch, pos=pos, tau=tau, hard=hard)
        latent = self.output_layer(pooled) # [batch_size, n_super_nodes, latent_dim]
        return latent, s, assign_losses, mu


class OptimizedHamiltonianODEFunc(nn.Module):
    """
    Optimized Hamiltonian ODE function with reduced computational complexity.
    """
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=64, dissipative=True):  # Increased hidden_dim
        super(OptimizedHamiltonianODEFunc, self).__init__()
        assert latent_dim % 2 == 0, "Latent dim must be even for Hamiltonian dynamics (q, p)"
        self.latent_dim = latent_dim
        self.n_super_nodes = n_super_nodes
        self.dissipative = dissipative

        # Simplified and more efficient Hamiltonian network
        total_input_dim = latent_dim * n_super_nodes
        # Use a slightly deeper network for better representation
        self.H_net = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        if dissipative:
            # Small initial dissipation
            self.gamma = nn.Parameter(torch.full((n_super_nodes, 1), -5.0)) # log space

    def forward(self, t, y):
        # y: [batch_size, latent_dim * n_super_nodes]
        training = torch.is_grad_enabled()

        # We need to compute dH/dy. We use autograd.grad.
        # create_graph=True is necessary for backpropagating through the ODE solver (especially for adjoint)
        with torch.set_grad_enabled(True):
            y_in = y.detach().requires_grad_(True)
            H = self.H_net(y_in).sum()
            dH = torch.autograd.grad(H, y_in, create_graph=training, allow_unused=True)[0]

            if dH is None:
                dH = torch.zeros_like(y_in)
            else:
                # Handle NaNs and Infs in the gradient
                dH = torch.nan_to_num(dH, nan=0.0, posinf=1e3, neginf=-1e3)

        # Gradient clipping for stability during ODE integration
        dH = torch.clamp(dH, -1e3, 1e3)

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
            gamma = torch.exp(torch.clamp(self.gamma, max=2.0)) # Clamp gamma to prevent extreme dissipation
            dp = dp - gamma * p

        return torch.cat([dq, dp], dim=-1).view(y.shape[0], -1)


class OptimizedGNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_features, box_size=10.0):
        super(OptimizedGNNDecoder, self).__init__()
        self.box_size = box_size
        # Reduced hidden dimensions
        self.shared_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//2),
            nn.ReLU()
        )
        # Position head: Maps to [-1, 1]
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim//2, 2),
            nn.Tanh()
        )
        # Velocity head: Maps to unconstrained Z-score values
        self.vel_head = nn.Linear(hidden_dim//2, 2)

    def forward(self, z, s, batch, stats=None):
        # z: [batch_size, n_super_nodes, latent_dim]
        # s: [N_total, n_super_nodes]

        # Expand z to match s: [N_total, n_super_nodes, latent_dim]
        z_expanded = z[batch]

        # Weighted sum: [N_total, n_super_nodes, 1] * [N_total, n_super_nodes, latent_dim]
        node_features_latent = torch.sum(s.unsqueeze(-1) * z_expanded, dim=1)

        shared_out = self.shared_mlp(node_features_latent)
        
        # Separate heads for position and velocity
        pos_recon = self.pos_head(shared_out)
        vel_recon = self.vel_head(shared_out)

        # Explicit denormalization using stats from trainer
        if stats is not None:
            pos_min = torch.tensor(stats['pos_min'], device=pos_recon.device, dtype=pos_recon.dtype)
            pos_range = torch.tensor(stats['pos_range'], device=pos_recon.device, dtype=pos_recon.dtype)
            # Map [-1, 1] back to original range
            pos_recon = 0.5 * (pos_recon + 1.0) * pos_range + pos_min
            
            vel_mean = torch.tensor(stats['vel_mean'], device=vel_recon.device, dtype=vel_recon.dtype)
            vel_std = torch.tensor(stats['vel_std'], device=vel_recon.device, dtype=vel_recon.dtype)
            vel_recon = vel_recon * vel_std + vel_mean
        
        return torch.cat([pos_recon, vel_recon], dim=-1)


class OptimizedDiscoveryEngineModel(nn.Module):
    def __init__(self, n_particles, n_super_nodes, node_features=4, latent_dim=4, hidden_dim=64, hamiltonian=False, dissipative=True, min_active_super_nodes=2, box_size=10.0):
        super(OptimizedDiscoveryEngineModel, self).__init__()
        # Determine normalized box size for GNN layers
        # If pos is mapped to [-1, 1], the box size in normalized space is 2.0
        normalized_box = 2.0 
        self.encoder = OptimizedGNNEncoder(node_features, hidden_dim, latent_dim, n_super_nodes, min_active_super_nodes=min_active_super_nodes, box_size=normalized_box)

        if hamiltonian:
            self.ode_func = OptimizedHamiltonianODEFunc(latent_dim, n_super_nodes, hidden_dim, dissipative=dissipative)
        else:
            # Use a simpler ODE function if not Hamiltonian
            from model import LatentODEFunc
            self.ode_func = LatentODEFunc(latent_dim, n_super_nodes, hidden_dim//2)

        self.decoder = OptimizedGNNDecoder(latent_dim, hidden_dim, node_features, box_size=box_size)
        self.hamiltonian = hamiltonian

        # Simplified MI discriminator with fewer parameters
        self.mi_discriminator = nn.Sequential(
            nn.Linear(min(latent_dim, 2) + 2, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, hidden_dim//4),
            nn.ReLU(),
            nn.Linear(hidden_dim//4, 1)
        )

        # Maintain the same number of log vars as the original model to be compatible with trainer
        # 0: rec, 1: cons, 2: assign, 3: ortho, 4: l2, 5: lvr, 6: align, 7: pruning, 8: sep, 9: conn, 10: sparsity, 11: mi, 12: sym, 13: var
        lvars = torch.zeros(14)
        lvars[0] = -5.0 # High priority for reconstruction
        lvars[1] = 0.5  # Consistency
        lvars[2] = 0.5  # Assignment
        lvars[3] = 0.0  # Ortho
        lvars[6] = -3.0 # Start align immediately with high weight
        lvars[12] = 2.0 # Suppress symbolic significantly initially
        lvars[13] = 1.0 # Suppress latent variance loss more initially
        self.log_vars = nn.Parameter(lvars) 
        
    def get_latent_variance_loss(self, z):
        """
        Explicitly penalize low latent variance to prevent manifold collapse.
        z: [B, K, D] or [T, B, K, D]
        """
        # Calculate variance across the feature dimension D for each super-node
        var = z.var(dim=-1).mean()
        # Hinge variance loss: only penalize if std < 1.0
        return torch.relu(1.0 - torch.sqrt(var + 1e-6))

    def get_mi_loss(self, z, mu):
        """
        Simplified mutual information estimation.
        z: [B, K, D], mu: [B, K, 2]
        """
        # Take first 2 dims of z for spatial alignment
        z_spatial = z[:, :, :2]

        # Joint distribution
        joint = self.mi_discriminator(torch.cat([z_spatial, mu], dim=-1))

        # Marginal distribution (shuffle mu across batch)
        batch_size, n_k, _ = mu.shape
        mu_shuffled = mu[torch.randperm(batch_size)]

        marginal = self.mi_discriminator(torch.cat([z_spatial, mu_shuffled], dim=-1))

        # Simplified MINE objective
        mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)) + 1e-9)
        return -mi_est

    def encode(self, x, edge_index, batch, tau=1.0, hard=False):
        return self.encoder(x, edge_index, batch, tau=tau, hard=hard)

    def decode(self, z, s, batch, stats=None):
        return self.decoder(z, s, batch, stats=stats)

    def get_ortho_loss(self, s):
        # Simplified orthogonality loss
        if s.size(0) == 0:
            return torch.tensor(0.0, device=s.device)
        n_nodes, k = s.shape
        dots = torch.matmul(s.t(), s)
        identity = torch.eye(k, device=s.device).mul_(n_nodes / k)
        return torch.mean((dots - identity)**2)

    def get_connectivity_loss(self, s, edge_index):
        if edge_index.numel() == 0:
            return torch.tensor(0.0, device=s.device)
        row, col = edge_index
        s_i = s[row]
        s_j = s[col]
        return torch.mean((s_i - s_j)**2)

    def forward_dynamics(self, z0, t):
        # z0: [batch_size, n_super_nodes, latent_dim]
        z0_flat = z0.view(z0.size(0), -1).to(torch.float32)
        t = t.to(torch.float32)

        # Use the device of the ode_func parameters
        ode_device = next(self.ode_func.parameters()).device
        original_device = z0.device

        y0 = z0_flat.to(ode_device)
        t_ode = t.to(ode_device)

        # Use adaptive tolerance based on training stage to balance accuracy and efficiency
        if self.training:
            # Much looser tolerances during early training for efficiency
            rtol = 1e-2  # Slightly tighter than 1e-1
            atol = 1e-3  # Slightly tighter than 1e-2
        else:
            # Tighter tolerances during evaluation for accuracy
            rtol = 1e-3
            atol = 1e-5

        # Use a more efficient solver during training
        if self.training:
            method = 'midpoint'  # Better than euler for Hamiltonian systems
        else:
            method = 'rk4'  # Reasonably accurate for evaluation

        # Use the selected solver
        if self.hamiltonian and self.training:
            # Use adjoint method to save memory when training Hamiltonian
            zt_flat = odeint_adjoint(self.ode_func, y0, t_ode, rtol=rtol, atol=atol, method=method)
        else:
            zt_flat = odeint(self.ode_func, y0, t_ode, rtol=rtol, atol=atol, method=method)

        # Move back to original device
        zt_flat = zt_flat.to(original_device)

        # zt_flat: [len(t), batch_size, latent_dim * n_super_nodes]
        return zt_flat.view(zt_flat.size(0), zt_flat.size(1), -1, z0.size(-1))