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
        # +5 for: dist_sq, dot(v, r), 1/r^2, 1/r^6, 1/r^12
        self.phi_e = nn.Sequential(
            nn.Linear(2 * in_channels + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Vector message network (outputs a scalar weight for the rel_pos vector)
        self.phi_v = nn.Sequential(
            nn.Linear(2 * in_channels + 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1)
        )
        # Node update network
        self.phi_h = nn.Sequential(
            nn.Linear(in_channels + hidden_dim + 1, hidden_dim), # +1 for vector message norm
            nn.LayerNorm(hidden_dim),
            nn.Softplus(),
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
        m_v_norm = torch.norm(m_v + 1e-8, dim=-1, keepdim=True)
        
        # Update node features
        h_update = self.phi_h(torch.cat([x, m_h, m_v_norm], dim=-1))
        
        # In a full equivariant GNN we would also update velocities, 
        # but here we focus on latent representation h.
        return self.shortcut(x) + h_update

    def message(self, x_i, x_j, dist_sq, dot_vr, rel_pos):
        # Physics-informed features with stability protection
        # Clamp instead of tanh to allow for sharper signals (like 1/r^12)
        r2_inv = torch.clamp(1.0 / (dist_sq + 0.05), max=20.0)
        r6_inv = torch.clamp(r2_inv**3, max=400.0)
        r12_inv = torch.clamp(r6_inv**2, max=160000.0)
        
        # Scalar message
        tmp = torch.cat([x_i, x_j, dist_sq, dot_vr, r2_inv, r6_inv, r12_inv], dim=1)
        m_h = self.phi_e(tmp)
        
        # Vector message weight
        v_w = self.phi_v(tmp)
        m_v = v_w * rel_pos
        
        return m_h, m_v

    def aggregate(self, inputs, index, dim_size=None):
        # Custom aggregate to handle tuple of (scalar_msg, vector_msg)
        m_h_in, m_v_in = inputs
        m_h = scatter(m_h_in, index, dim=self.node_dim, dim_size=dim_size, reduce='mean').to(torch.float32)
        m_v = scatter(m_v_in, index, dim=self.node_dim, dim_size=dim_size, reduce='mean').to(torch.float32)
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
    def __init__(self, node_features, hidden_dim, latent_dim, n_super_nodes, n_particles=None, min_active_super_nodes=4):
        super(GNNEncoder, self).__init__()
        self.gnn1 = EquivariantGNNLayer(node_features, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gnn2 = EquivariantGNNLayer(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.n_particles = n_particles

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

        # Add a learnable gain parameter to prevent vanishing latents - REDUCED FROM 5.0 TO 1.0
        self.latent_gain = nn.Parameter(torch.ones(latent_dim) * 1.0)

        # Initialize output layer to spread out initial latent states
        nn.init.normal_(self.output_layer[-1].weight, mean=0.0, std=1.0)
        nn.init.zeros_(self.output_layer[-1].bias)

    def _initialize_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch, tau=1.0, hard=False, prev_assignments=None, current_epoch=None, total_epochs=None):
        # Store initial positions for spatial pooling
        pos = x[:, :2]
        vel = x[:, 2:4] # Assume [pos_x, pos_y, vel_x, vel_y]

        x = self.ln1(torch.relu(self.gnn1(x, pos, vel, edge_index)))
        x = self.ln2(torch.relu(self.gnn2(x, pos, vel, edge_index)))

        # Pool to K super-nodes preserving spatial features
        pooled, s, assign_losses, mu = self.pooling(x, batch, pos=pos, tau=tau, hard=hard, prev_assignments=prev_assignments, current_epoch=current_epoch, total_epochs=total_epochs) # [batch_size, n_super_nodes, hidden_dim], [N, n_super_nodes], mu: [B, K, 2]
        latent = self.output_layer(pooled) # [batch_size, n_super_nodes, latent_dim]

        # Apply learnable gain to prevent vanishing latents
        latent = latent * self.latent_gain

        # Remove tanh activation to prevent the 'wall' effect seen in the trajectories
        # Instead, use LayerNorm followed by a small Linear layer with std=0.1 initialization
        latent = nn.functional.layer_norm(latent, normalized_shape=(self.latent_dim,))

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
            nn.Softplus(),  # Smoother activation for better derivatives
            nn.Linear(hidden_dim//4, hidden_dim//4)
        )

        # Aggregation function (rho network) - mean pooling is permutation invariant
        # This aggregates information from all super-nodes
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim//4, hidden_dim//4),
            nn.Softplus(),  # Smoother activation for better derivatives
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
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=64, dissipative=True, separable=True):
        """
        Initialize the Hamiltonian ODE function.

        Args:
            latent_dim (int): Dimension of latent space (must be even for (q,p) pairs)
            n_super_nodes (int): Number of super-nodes
            hidden_dim (int): Hidden dimension for the Hamiltonian network
            dissipative (bool): Whether to include learnable dissipation terms
            separable (bool): Whether to enforce H = V(q) + sum(p^2/2)
        """
        super(HamiltonianODEFunc, self).__init__()
        assert latent_dim % 2 == 0, "Latent dim must be even for Hamiltonian dynamics (q, p)"
        self.latent_dim = latent_dim
        self.n_super_nodes = n_super_nodes
        self.dissipative = dissipative
        self.separable = separable

        # Potential network V(q)
        # If separable, input is (latent_dim // 2) * n_super_nodes
        # Else, input is latent_dim * n_super_nodes
        v_input_dim = (latent_dim // 2) * n_super_nodes if separable else latent_dim * n_super_nodes
        
        self.V_net = nn.Sequential(
            nn.Linear(v_input_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Softplus(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # For non-separable, we keep H_net for compatibility if needed, 
        # but we'll use V_net as the primary engine.
        self.H_net = self.V_net if not separable else None

        # Initialize weights with smaller values
        self._initialize_weights_small()

        # Add LayerNorm at the input to stabilize gradients
        self.input_norm = nn.LayerNorm(v_input_dim)

        if dissipative:
            # Small initial dissipation
            self.gamma = nn.Parameter(torch.full((n_super_nodes, 1), -5.0)) # log space

        # NEW: Dynamic Mass Matrix
        # Initialize masses to 1.0 (log-space for positivity)
        self.log_masses = nn.Parameter(torch.zeros(n_super_nodes, 1))

    def _initialize_weights_small(self):
        """Initialize weights using orthogonal initialization."""
        for m in self.V_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Explicitly make the last layer even smaller to start with a flat potential
        if isinstance(self.V_net[-1], nn.Linear):
            nn.init.orthogonal_(self.V_net[-1].weight, gain=0.01)

    def get_masses(self):
        """Returns the diagonal elements of the mass matrix M."""
        return torch.exp(self.log_masses)

    def hamiltonian(self, y):
        """Compute the total Hamiltonian H(q, p)."""
        d_sub = self.latent_dim // 2
        y_view = y.view(-1, self.n_super_nodes, 2, d_sub)
        q = y_view[:, :, 0].reshape(y.size(0), -1)
        p = y_view[:, :, 1] # [B, K, D_sub]

        if self.separable:
            # H = V(q) + T(p)
            # T = 1/2 * p^T * M^-1 * p
            q_norm = self.input_norm(q)
            V = self.V_net(q_norm)
            
            # Dynamic Mass Matrix (diagonal)
            M_diag = self.get_masses().view(1, self.n_super_nodes, 1)
            T = 0.5 * torch.sum((p**2) / M_diag, dim=(1, 2)).unsqueeze(-1)
            return V + T
        else:
            # Full H(q, p)
            y_norm = self.input_norm(y)
            return self.V_net(y_norm)

    def forward(self, t, y):
        """
        Compute time derivatives using Hamiltonian mechanics.
        """
        # y: [batch_size, latent_dim * n_super_nodes]
        d_sub = self.latent_dim // 2
        
        # Ensure we are in a grad-enabled context
        with torch.set_grad_enabled(True):
            # Optimization: Only detach and set requires_grad if not already present
            # This preserves the graph for higher-order derivatives
            y_in = y if y.requires_grad else y.detach().requires_grad_(True)
            H_val = self.hamiltonian(y_in)
            H = H_val.sum()
            
            # Use create_graph=True to allow for higher-order derivatives (Jacobians)
            dH = torch.autograd.grad(H, y_in, create_graph=True, retain_graph=True, allow_unused=True)[0]
            
            if dH is None:
                dH = torch.zeros_like(y_in)
            else:
                dH = torch.nan_to_num(dH, nan=0.0, posinf=1e2, neginf=-1e2)

        dH = torch.clamp(dH, -1e2, 1e2)
        dH_view = dH.view(-1, self.n_super_nodes, 2, d_sub)

        dq = dH_view[:, :, 1]  # dH/dp
        dp = -dH_view[:, :, 0] # -dH/dq

        if self.dissipative:
            y_view = y.view(-1, self.n_super_nodes, 2, d_sub)
            p = y_view[:, :, 1]
            gamma = torch.exp(torch.clamp(self.gamma, max=2.0))
            dp = dp - gamma * p

        result = torch.cat([dq, dp], dim=-1).view(y.shape[0], -1)
        return torch.clamp(result, -1e2, 1e2)

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return x + self.net(x)

class GNNDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_features, box_size=10.0):
        super(GNNDecoder, self).__init__()
        self.box_size = box_size
        # ENHANCED: Double hidden_dim and use Residual architecture
        hidden_dim = hidden_dim * 2
        self.shared_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )

        # Add LayerNorm before the heads to prevent large latent values from saturating Tanh
        self.pre_pos_norm = nn.LayerNorm(hidden_dim)
        self.pre_vel_norm = nn.LayerNorm(hidden_dim)

        # ENHANCED: Use deeper heads for position and velocity reconstruction
        self.pos_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
            nn.Tanh()
        )
        # Initialize pos_head weights to prevent initial large offsets
        for layer in self.pos_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

        # Velocity head: Maps to unconstrained Z-score values
        self.vel_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2)
        )
        for layer in self.vel_head:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)

        # Initialize weights using Xavier uniform with higher variance for better reconstruction
        self._initialize_weights()

        # Initialize decoder layers with standard weights for better initial reconstruction
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
    def __init__(self, n_particles, n_super_nodes, node_features=4, latent_dim=4, hidden_dim=128, hamiltonian=False, dissipative=True, min_active_super_nodes=4, box_size=10.0):
        super(DiscoveryEngineModel, self).__init__()
        self.encoder = GNNEncoder(node_features, hidden_dim, latent_dim, n_super_nodes, n_particles=n_particles, min_active_super_nodes=min_active_super_nodes)
        
        if hamiltonian:
            self.ode_func = HamiltonianODEFunc(latent_dim, n_super_nodes, hidden_dim, dissipative=dissipative)
        else:
            self.ode_func = LatentODEFunc(latent_dim, n_super_nodes, hidden_dim)
            
        self.decoder = GNNDecoder(latent_dim, hidden_dim, node_features, box_size=box_size)
        self.hamiltonian = hamiltonian

        # Mutual Information alignment components
        # Align first 2 dims of latent to mu (physical CoM)
        self.mi_discriminator = MIDiscriminator(min(latent_dim, 2), 2, hidden_dim)
        
        # NEW: Linear Alignment Layer for Soft CCA
        # This layer is used ONLY for alignment loss and does not affect discovery
        self.linear_aligner = nn.ModuleList([nn.Linear(latent_dim, 2) for _ in range(n_super_nodes)])

        # Learnable loss log-variances for automatic loss balancing (one per head)
        # 0: Reconstruction, 1: Structural, 2: Physicality, 3: Symbolic
        self.log_vars = nn.Parameter(torch.zeros(4)) 
        
    def get_latent_variance_loss(self, z):
        """
        Explicitly penalize low latent variance to prevent manifold collapse.
        z: [B, K, D] or [T, B, K, D]
        """
        from common_losses import get_latent_variance_loss as common_latent_variance_loss
        return common_latent_variance_loss(z)

    def get_activity_penalty(self, z):
        """
        Encourage the latent variables to have non-zero standard deviation over the temporal sequence.
        z: [T, B, K, D] where T is time steps
        """
        # Calculate std across time dimension with epsilon for stability
        temporal_std = torch.sqrt(torch.var(z, dim=0) + 1e-8)  # [B, K, D]
        # Encourage non-zero standard deviation to prevent static latents
        # Increased threshold to 5.0 and using squared penalty for a much stronger "push"
        activity_penalty = torch.mean(torch.pow(torch.relu(5.0 - temporal_std), 2))
        return activity_penalty

    def get_mi_loss(self, z, mu):
        """
        Unsupervised alignment loss via Mutual Information Maximization (MINE).
        Combined with a stronger linear correlation objective and linear reconstruction.
        z: [B, K, D], mu: [B, K, 2]
        """
        from common_losses import get_mi_loss as common_mi_loss
        from common_losses import get_correlation_loss as common_corr_loss
        
        mi_loss = common_mi_loss(z, mu, self.mi_discriminator)
        corr_loss = common_corr_loss(z, mu)
        
        # NEW: Linear Alignment Loss (Soft CCA)
        linear_loss = 0.0
        for k in range(len(self.linear_aligner)):
            mu_pred = self.linear_aligner[k](z[:, k, :])
            linear_loss += torch.nn.functional.mse_loss(mu_pred, mu[:, k, :])
        
        return mi_loss + 2.0 * corr_loss + 5.0 * linear_loss

    def encode(self, x, edge_index, batch, tau=1.0, hard=False, prev_assignments=None, current_epoch=None, total_epochs=None):
        return self.encoder(x, edge_index, batch, tau=tau, hard=hard, prev_assignments=prev_assignments, current_epoch=current_epoch, total_epochs=total_epochs)
    
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
    
    def forward_dynamics(self, z0, t, step_size=None, chunk_size=None):
        """
        Compute time derivatives with optimized device management and chunking.
        """
        # z0: [batch_size, n_super_nodes, latent_dim]
        z0_shape = z0.shape
        z0_flat = z0.reshape(z0.size(0), -1).to(torch.float32)
        t = t.to(torch.float32)

        # Optimization: Cache the target device and is_mps flag
        if not hasattr(self, '_cached_target_device'):
            ode_device = next(self.ode_func.parameters()).device
            is_mps = (str(z0.device) == 'mps' or str(ode_device) == 'mps')
            self._cached_target_device = torch.device('cpu') if is_mps else ode_device
            self._is_mps_fix = is_mps
        
        target_device = self._cached_target_device
        original_device = z0.device

        # Determine chunk size if not provided
        if chunk_size is None:
            chunk_size = 20 if self._is_mps_fix else 100
            
        y0 = z0_flat if z0_flat.device == target_device else z0_flat.to(target_device)
        t_ode = t if t.device == target_device else t.to(target_device)

        # Ensure ode_func is on target_device
        if next(self.ode_func.parameters()).device != target_device:
            self.ode_func.to(target_device)

        if step_size is None:
            if len(t) > 1:
                dt_span = (t[1] - t[0]).item()
                # Use larger steps for training, slightly smaller for evaluation
                step_size = dt_span / 2 if self.training else dt_span / 5
            else:
                step_size = 0.01

        method = 'midpoint' if self.training else 'rk4'
        options = {'step_size': step_size}
        solver_fn = odeint_adjoint if (self.hamiltonian and self.training) else odeint

        # Chunked Integration Loop
        if len(t) > chunk_size:
            zt_flat_list = []
            curr_y0 = y0
            for i in range(0, len(t) - 1, chunk_size - 1):
                end_idx = min(i + chunk_size, len(t))
                t_chunk = t_ode[i:end_idx]
                
                # Adjust t_chunk to be relative to 0 if necessary for the solver, 
                # but odeint usually handles absolute t.
                out_chunk = solver_fn(self.ode_func, curr_y0, t_chunk, method=method, options=options)
                
                # Avoid duplicating the overlapping point
                if i == 0:
                    zt_flat_list.append(out_chunk)
                else:
                    zt_flat_list.append(out_chunk[1:])
                
                curr_y0 = out_chunk[-1]
            zt_flat = torch.cat(zt_flat_list, dim=0)
        else:
            zt_flat = solver_fn(self.ode_func, y0, t_ode, method=method, options=options)

        # Final cleanup and return
        if zt_flat.device != original_device:
            zt_flat = zt_flat.to(original_device)

        if torch.isnan(zt_flat).any():
            zt_flat = torch.nan_to_num(zt_flat, nan=0.0, posinf=1e2, neginf=-1e2)

        return zt_flat.reshape(zt_flat.size(0), zt_flat.size(1), z0_shape[1], z0_shape[2])
