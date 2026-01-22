"""
Neural-Symbolic Discovery Pipeline - Model Components

This module implements the core neural network architectures for the
coarse-graining of particle dynamics using Graph Neural Networks (GNNs)
and Differentiable Physics (Latent ODEs).

The architecture includes:
- GNN-based encoder with hierarchical soft-assignment pooling
- Hamiltonian-constrained ODE dynamics
- GNN-based decoder for reconstruction
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torchdiffeq import odeint, odeint_adjoint
from stable_pooling import StableHierarchicalPooling


class EquivariantGNNLayer(MessagePassing):
    """
    Enhanced E(n)-equivariant GNN Layer.
    Uses relative distances, relative velocities, and vector-valued messages
    to better capture physical interactions like angular momentum.
    """
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super(EquivariantGNNLayer, self).__init__(aggr='mean')
        # Scalar message network
        self.phi_e = nn.Sequential(
            nn.Linear(2 * in_channels + 2, hidden_dim), # +2 for dist_sq and dot(v, r)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Vector message network (outputs a scalar weight for the rel_pos vector)
        self.phi_v = nn.Sequential(
            nn.Linear(2 * in_channels + 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Node update network
        self.phi_h = nn.Sequential(
            nn.Linear(in_channels + hidden_dim + 1, hidden_dim), # +1 for vector message norm
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels)
        )
        
        # Shortcut for residual connection when dims don't match
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, pos, vel, edge_index):
        # x: [N, in_channels], pos: [N, 2], vel: [N, 2]
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        rel_vel = vel[edge_index[0]] - vel[edge_index[1]]
        
        dist_sq = torch.sum(rel_pos**2, dim=-1, keepdim=True)
        dot_vr = torch.sum(rel_vel * rel_pos, dim=-1, keepdim=True)
        
        # Propagate messages
        m_h, m_v = self.propagate(edge_index, x=x, dist_sq=dist_sq, dot_vr=dot_vr, rel_pos=rel_pos)
        
        # m_v is a vector [N, 2]. We take its norm as a scalar feature
        m_v_norm = torch.norm(m_v, dim=-1, keepdim=True)
        
        # Update node features
        h_update = self.phi_h(torch.cat([x, m_h, m_v_norm], dim=-1))
        
        # In a full equivariant GNN we would also update velocities, 
        # but here we focus on latent representation h.
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

class HierarchicalPooling(nn.Module):
    """
    Learned soft-assignment pooling to preserve spatial locality
    by aggregating nodes into a fixed number of super-nodes.
    Uses Gumbel-Softmax for 'harder' assignments and includes pruning.

    This module implements hierarchical soft-assignment pooling that:
    1. Learns to assign nodes to super-nodes based on their features
    2. Preserves spatial relationships between nodes
    3. Dynamically prunes unused super-nodes to find optimal meso-scale resolution
    4. Computes various regularization losses to encourage meaningful assignments
    """
    def __init__(self, in_channels, n_super_nodes, pruning_threshold=0.01):
        """
        Initialize the hierarchical pooling layer.

        Args:
            in_channels (int): Number of input feature channels per node
            n_super_nodes (int): Number of super-nodes to pool to
            pruning_threshold (float): Threshold for pruning super-nodes
        """
        super(HierarchicalPooling, self).__init__()
        self.n_super_nodes = n_super_nodes
        self.pruning_threshold = pruning_threshold
        self.assign_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, n_super_nodes)
        )
        self.scaling = nn.Parameter(torch.tensor(1.0))
        # Mask to track active super-nodes (not directly optimized by backprop)
        self.register_buffer('active_mask', torch.ones(n_super_nodes))

    def forward(self, x, batch, pos=None, tau=1.0, hard=False):
        """
        Forward pass of hierarchical pooling.

        Args:
            x (Tensor): Node features [N, in_channels]
            batch (Tensor): Batch assignment [N]
            pos (Tensor, optional): Node positions [N, 2]
            tau (float): Temperature for Gumbel-Softmax
            hard (bool): Whether to use hard sampling

        Returns:
            Tuple of (pooled_features, assignment_matrix, losses, super_node_positions)
        """
        # x: [N, in_channels], batch: [N], pos: [N, 2]
        if x.size(0) == 0:
            return torch.zeros((0, self.n_super_nodes, x.size(1)), device=x.device), \
                   torch.zeros((0, self.n_super_nodes), device=x.device), \
                   {'entropy': torch.tensor(0.0, device=x.device),
                    'diversity': torch.tensor(0.0, device=x.device),
                    'spatial': torch.tensor(0.0, device=x.device),
                    'pruning': torch.tensor(0.0, device=x.device)}, \
                   None

        logits = self.assign_mlp(x) * self.scaling

        # Apply active_mask to logits (soft mask to allow for revival)
        mask = self.active_mask.unsqueeze(0)
        
        if self.training:
            # Softer mask during training to allow reactivation
            logits = logits + (mask - 1.0) * 10.0
        else:
            # Hard mask during inference
            logits = logits.masked_fill(mask == 0, -1e9)

        s = nn.functional.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        avg_s = s.mean(dim=0)

        # Update active_mask if training and not hard (hard usually for eval)
        if self.training and not hard:
            # Moving average update for the mask to avoid rapid flickering
            current_active = (avg_s > self.pruning_threshold).float()
            self.active_mask.copy_(0.95 * self.active_mask + 0.05 * current_active)
            # Ensure at least one is always active to avoid collapse
            if self.active_mask.sum() == 0:
                self.active_mask[torch.argmax(avg_s)] = 1.0

        entropy = -torch.mean(torch.sum(s * torch.log(s + 1e-9), dim=1))
        diversity_loss = torch.sum(avg_s * torch.log(avg_s + 1e-9))
        pruning_loss = torch.mean(torch.abs(avg_s * (1 - self.active_mask))) # Penalize usage of "inactive" nodes

        # Sparsity loss to encourage finding the minimal scale
        sparsity_loss = torch.sum(self.active_mask) / self.n_super_nodes

        spatial_loss = torch.tensor(0.0, device=x.device)
        if pos is not None:
            s_sum = s.sum(dim=0, keepdim=True) + 1e-9
            s_norm = s / s_sum
            mu = torch.matmul(s_norm.t(), pos)
            pos_sq = (pos**2).sum(dim=1, keepdim=True)
            mu_sq = (mu**2).sum(dim=1)
            var = torch.matmul(s_norm.t(), pos_sq).squeeze() - 2 * (mu * torch.matmul(s_norm.t(), pos)).sum(dim=1) + mu_sq
            spatial_loss = var.mean()

        assign_losses = {
            'entropy': entropy,
            'diversity': diversity_loss,
            'spatial': spatial_loss,
            'pruning': pruning_loss,
            'sparsity': sparsity_loss
        }

        super_node_mu = None
        if pos is not None and batch is not None:
            s_pos_expanded = pos.unsqueeze(1) * s.unsqueeze(2)
            sum_s_pos = scatter(s_pos_expanded, batch, dim=0, reduce='sum').to(torch.float32)
            sum_s = scatter(s, batch, dim=0, reduce='sum').to(torch.float32).unsqueeze(-1) + 1e-9
            super_node_mu = sum_s_pos / sum_s

        if super_node_mu is not None:
            mu = super_node_mu
            dist_sq = torch.sum((mu.unsqueeze(1) - mu.unsqueeze(2))**2, dim=-1)
            mask_eye = torch.eye(self.n_super_nodes, device=x.device).unsqueeze(0)
            repulsion = 1.0 / (dist_sq + 1.0)
            separation_loss = (repulsion * (1 - mask_eye)).sum() / (self.n_super_nodes * (self.n_super_nodes - 1) + 1e-9)
            assign_losses['separation'] = separation_loss

        x_expanded = x.unsqueeze(1) * s.unsqueeze(2)
        if batch is not None:
            out = scatter(x_expanded, batch, dim=0, reduce='sum').to(torch.float32)
        else:
            # Return appropriate shape when batch is None
            out = torch.zeros((0, self.n_super_nodes, x.size(1)), device=x.device, dtype=x.dtype)

        return out, s, assign_losses, mu

class GNNEncoder(nn.Module):
    def __init__(self, node_features, hidden_dim, latent_dim, n_super_nodes, min_active_super_nodes=2):
        super(GNNEncoder, self).__init__()
        self.gnn1 = EquivariantGNNLayer(node_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gnn2 = EquivariantGNNLayer(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim

        # Spatially-aware pooling instead of global_mean_pool
        # Use the Enhanced StableHierarchicalPooling
        self.pooling = StableHierarchicalPooling(hidden_dim, n_super_nodes, min_active_super_nodes=min_active_super_nodes)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Initialize weights using Xavier uniform
        self._initialize_weights()

        # Add a learnable gain parameter to prevent vanishing latents
        self.latent_gain = nn.Parameter(torch.ones(latent_dim) * 1.0)

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch, tau=1.0, hard=False, prev_assignments=None):
        # Store initial positions for spatial pooling
        pos = x[:, :2]
        vel = x[:, 2:4] # Assume [pos_x, pos_y, vel_x, vel_y]

        x = self.ln1(torch.relu(self.gnn1(x, pos, vel, edge_index)))
        x = self.ln2(torch.relu(self.gnn2(x, pos, vel, edge_index)))

        # Pool to K super-nodes preserving spatial features
        pooled, s, assign_losses, mu = self.pooling(x, batch, pos=pos, tau=tau, hard=hard, prev_assignments=prev_assignments) # [batch_size, n_super_nodes, hidden_dim], [N, n_super_nodes], mu: [B, K, 2]
        latent = self.output_layer(pooled) # [batch_size, n_super_nodes, latent_dim]

        # Apply learnable gain to prevent vanishing latents
        latent = latent * self.latent_gain

        return latent, s, assign_losses, mu

class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=64):
        super(LatentODEFunc, self).__init__()
        self.input_dim = latent_dim * n_super_nodes
        self.latent_dim = latent_dim
        self.n_super_nodes = n_super_nodes
        self.hidden_dim = hidden_dim

        # Further simplified and faster architecture
        # Process each super-node individually (phi network)
        self.phi = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim//4),  # Further reduced hidden size
            nn.SiLU(),  # Faster activation
            nn.Linear(hidden_dim//4, hidden_dim//4)
        )

        # Aggregation function (rho network) - mean pooling is permutation invariant
        # This aggregates information from all super-nodes
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//4),
            nn.SiLU(),  # Faster activation
            nn.Linear(hidden_dim//4, latent_dim * n_super_nodes)
        )

    def forward(self, t, y):
        # y: [batch_size, latent_dim * n_super_nodes]
        batch_size = y.size(0)

        # Reshape to [batch_size, n_super_nodes, latent_dim]
        y_reshaped = y.view(batch_size, self.n_super_nodes, self.latent_dim)

        # Apply phi network to each super-node individually
        # [batch_size, n_super_nodes, hidden_dim]
        phi_outputs = self.phi(y_reshaped)

        # Aggregate across super-nodes using mean (permutation invariant)
        # [batch_size, hidden_dim]
        aggregated = phi_outputs.mean(dim=1)

        # Apply rho network to get final output
        # [batch_size, latent_dim * n_super_nodes]
        output = self.rho(aggregated)

        return output

class HamiltonianODEFunc(nn.Module):
    """
    Enforces Hamiltonian constraints: dq/dt = dH/dp, dp/dt = -dH/dq.
    Optionally includes a learnable dissipation term: dp/dt = -dH/dq - gamma * p.

    This module implements Hamiltonian dynamics by learning a Hamiltonian function H
    and computing its gradients to determine the time derivatives of position and momentum.
    The Hamiltonian structure preserves phase-space volume (Liouville's Theorem),
    which is critical for long-term stability in symbolic integration.
    """
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=64, dissipative=True):
        """
        Initialize the Hamiltonian ODE function.

        Args:
            latent_dim (int): Dimension of latent space (must be even for (q,p) pairs)
            n_super_nodes (int): Number of super-nodes
            hidden_dim (int): Hidden dimension for the Hamiltonian network
            dissipative (bool): Whether to include learnable dissipation terms
        """
        super(HamiltonianODEFunc, self).__init__()
        assert latent_dim % 2 == 0, "Latent dim must be even for Hamiltonian dynamics (q, p)"
        self.latent_dim = latent_dim
        self.n_super_nodes = n_super_nodes
        self.dissipative = dissipative

        # Further simplified and faster Hamiltonian network
        self.H_net = nn.Sequential(
            nn.Linear(latent_dim * n_super_nodes, hidden_dim//4),  # Further reduced hidden size
            nn.SiLU(), # Faster activation function
            nn.Linear(hidden_dim//4, 1)
        )

        if dissipative:
            # Small initial dissipation
            self.gamma = nn.Parameter(torch.full((n_super_nodes, 1), -5.0)) # log space

    def forward(self, t, y):
        """
        Compute time derivatives using Hamiltonian mechanics.

        Args:
            t (Tensor): Time (unused in autonomous systems)
            y (Tensor): State vector [batch_size, latent_dim * n_super_nodes]

        Returns:
            Tensor: Time derivatives [batch_size, latent_dim * n_super_nodes]
        """
        # y: [batch_size, latent_dim * n_super_nodes]
        training = torch.is_grad_enabled()

        # We need to compute dH/dy. We use autograd.grad.
        # create_graph=True is necessary for backpropagating through the ODE solver (especially for adjoint)
        with torch.set_grad_enabled(True):
            y_in = y.detach().requires_grad_(True)
            H_raw = self.H_net(y_in)
            # Add small L2 regularization on the output to prevent scaling to infinity
            H = H_raw.sum() + 0.01 * torch.sum(H_raw**2)
            dH = torch.autograd.grad(H, y_in, create_graph=True, allow_unused=True)[0]
            
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

class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_features, box_size=10.0):
        super(GNNDecoder, self).__init__()
        self.box_size = box_size
        self.shared_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Add LayerNorm before the heads to prevent large latent values from saturating Tanh
        self.pre_pos_norm = nn.LayerNorm(hidden_dim)
        self.pre_vel_norm = nn.LayerNorm(hidden_dim)

        # Position head: Maps to [-1, 1]
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Tanh()
        )
        # Initialize pos_head weights to prevent initial saturation
        nn.init.normal_(self.pos_head[0].weight, std=0.01)
        nn.init.zeros_(self.pos_head[0].bias)

        # Velocity head: Maps to unconstrained Z-score values
        self.vel_head = nn.Linear(hidden_dim, 2)

        # Initialize weights using Xavier uniform
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, s, batch, stats=None):
        # z: [batch_size, n_super_nodes, latent_dim]
        # s: [N_total, n_super_nodes]

        # Expand z to match s: [N_total, n_super_nodes, latent_dim]
        z_expanded = z[batch]

        # Weighted sum: [N_total, n_super_nodes, 1] * [N_total, n_super_nodes, latent_dim]
        node_features_latent = torch.sum(s.unsqueeze(-1) * z_expanded, dim=1)

        shared_out = self.shared_mlp(node_features_latent)

        # Apply LayerNorm before the heads to prevent large latent values from saturating Tanh
        pos_input = self.pre_pos_norm(shared_out)
        vel_input = self.pre_vel_norm(shared_out)

        # Separate heads for position and velocity
        pos_recon = self.pos_head(pos_input)
        vel_recon = self.vel_head(vel_input)

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

class MIDiscriminator(nn.Module):
    """
    Discriminator for Mutual Information Estimation (MINE-style).
    Distinguishes between (latent, physical) pairs and shuffled ones.
    """
    def __init__(self, latent_dim, physical_dim, hidden_dim=64):
        super(MIDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + physical_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, z, p):
        # z: [B, K, D], p: [B, K, 2]
        # Cat along last dim
        zp = torch.cat([z, p], dim=-1)
        return self.net(zp)

class DiscoveryEngineModel(nn.Module):
    def __init__(self, n_particles, n_super_nodes, node_features=4, latent_dim=4, hidden_dim=64, hamiltonian=False, dissipative=True, min_active_super_nodes=2, box_size=10.0):
        super(DiscoveryEngineModel, self).__init__()
        self.encoder = GNNEncoder(node_features, hidden_dim, latent_dim, n_super_nodes, min_active_super_nodes=min_active_super_nodes)
        
        if hamiltonian:
            self.ode_func = HamiltonianODEFunc(latent_dim, n_super_nodes, hidden_dim, dissipative=dissipative)
        else:
            self.ode_func = LatentODEFunc(latent_dim, n_super_nodes, hidden_dim)
            
        self.decoder = GNNDecoder(latent_dim, hidden_dim, node_features, box_size=box_size)
        self.hamiltonian = hamiltonian

        # Mutual Information alignment components
        # Align first 2 dims of latent to mu (physical CoM)
        self.mi_discriminator = MIDiscriminator(min(latent_dim, 2), 2, hidden_dim)

        # Learnable loss log-variances for automatic loss balancing
        # 0: rec, 1: cons, 2: assign, 3: ortho, 4: l2, 5: lvr, 6: align, 7: pruning, 8: sep, 9: conn, 10: sparsity, 11: mi, 12: sym, 13: var, 14: hinge, 15: smooth
        self.log_vars = nn.Parameter(torch.zeros(16)) 
        
    def get_latent_variance_loss(self, z):
        """
        Explicitly penalize low latent variance to prevent manifold collapse.
        z: [B, K, D] or [T, B, K, D]
        """
        from common_losses import get_latent_variance_loss as common_latent_variance_loss
        return common_latent_variance_loss(z)

    def get_mi_loss(self, z, mu):
        """
        Unsupervised alignment loss via Mutual Information Maximization (MINE).
        z: [B, K, D], mu: [B, K, 2]
        """
        from common_losses import get_mi_loss as common_mi_loss
        return common_mi_loss(z, mu, self.mi_discriminator)

    def encode(self, x, edge_index, batch, tau=1.0, hard=False, prev_assignments=None):
        return self.encoder(x, edge_index, batch, tau=tau, hard=hard, prev_assignments=prev_assignments)
    
    def decode(self, z, s, batch, stats=None):
        return self.decoder(z, s, batch, stats=stats)
    
    def get_ortho_loss(self, s):
        # s: [N, n_super_nodes]
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
        
        # MPS fix: torchdiffeq is unstable on MPS. If original or ode device is MPS, use CPU.
        is_mps = (str(original_device) == 'mps' or str(ode_device) == 'mps')
        target_device = torch.device('cpu') if is_mps else ode_device

        y0 = z0_flat.to(target_device)
        t_ode = t.to(target_device)
        
        # Ensure ode_func is on the correct device
        self.ode_func.to(target_device)

        # Use adaptive tolerance based on training stage
        eps = 1e-3 if is_mps else 0.0
        
        if self.training:
            # Looser tolerances during training for efficiency
            rtol = 1e-1 + eps
            atol = 1e-2 + eps
        else:
            # Tighter tolerances during evaluation for accuracy
            rtol = 1e-3 + eps
            atol = 1e-5 + eps

        # Use a more efficient solver during training, but accurate one for evaluation
        if self.training:
            method = 'euler'  # Fastest fixed-step method during early training
        else:
            method = 'rk4'  # Faster than dopri5 but still reasonably accurate

        # Use the selected solver
        if self.hamiltonian and self.training:
            # Use adjoint method to save memory when training Hamiltonian
            # The adjoint method is more memory efficient for backpropagating through ODEs
            zt_flat = odeint_adjoint(self.ode_func, y0, t_ode, rtol=rtol, atol=atol, method=method)
        else:
            zt_flat = odeint(self.ode_func, y0, t_ode, rtol=rtol, atol=atol, method=method)

        # Move back to original device
        zt_flat = zt_flat.to(original_device)

        # zt_flat: [len(t), batch_size, latent_dim * n_super_nodes]
        return zt_flat.view(zt_flat.size(0), zt_flat.size(1), -1, z0.size(-1))
