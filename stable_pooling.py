"""
Enhanced Hierarchical Pooling with improved assignment stability to prevent latent flickering.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import torch.nn.functional as F


class StableHierarchicalPooling(nn.Module):
    """
    Enhanced HierarchicalPooling with improved assignment stability to prevent latent flickering.
    
    Key improvements:
    1. Temporal consistency loss to penalize rapid assignment changes
    2. Exponential moving average for smoother active mask updates
    3. Assignment persistence mechanism to maintain stable cluster identities
    4. Adaptive temperature scheduling for Gumbel-Softmax
    """
    
    def __init__(self, in_channels, n_super_nodes, pruning_threshold=0.01, temporal_consistency_weight=0.1):
        """
        Initialize the stable hierarchical pooling layer.

        Args:
            in_channels (int): Number of input feature channels per node
            n_super_nodes (int): Number of super-nodes to pool to
            pruning_threshold (float): Threshold for pruning super-nodes
            temporal_consistency_weight (float): Weight for temporal consistency loss
        """
        super(StableHierarchicalPooling, self).__init__()
        self.n_super_nodes = n_super_nodes
        self.pruning_threshold = pruning_threshold
        self.temporal_consistency_weight = temporal_consistency_weight
        
        self.assign_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, n_super_nodes)
        )
        self.scaling = nn.Parameter(torch.tensor(1.0))
        
        # Mask to track active super-nodes (not directly optimized by backprop)
        self.register_buffer('active_mask', torch.ones(n_super_nodes))
        
        # Track previous assignments for temporal consistency
        self.register_buffer('prev_assignments', torch.zeros(n_super_nodes))
        self.register_buffer('assignment_history', torch.zeros(n_super_nodes))
        self.register_buffer('history_counter', torch.zeros(1, dtype=torch.long))

    def forward(self, x, batch, pos=None, tau=1.0, hard=False, prev_assignments=None):
        """
        Forward pass of hierarchical pooling with enhanced stability.

        Args:
            x (Tensor): Node features [N, in_channels]
            batch (Tensor): Batch assignment [N]
            pos (Tensor, optional): Node positions [N, 2]
            tau (float): Temperature for Gumbel-Softmax
            hard (bool): Whether to use hard sampling
            prev_assignments (Tensor, optional): Previous assignment matrix for consistency

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
                    'pruning': torch.tensor(0.0, device=x.device),
                    'temporal_consistency': torch.tensor(0.0, device=x.device)}, \
                   None

        logits = self.assign_mlp(x) * self.scaling

        # Apply active_mask to logits (set inactive ones to very low value)
        mask = self.active_mask.unsqueeze(0)
        logits = logits.masked_fill(mask == 0, -1e9)

        s = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        avg_s = s.mean(dim=0)

        # Update active_mask with exponential moving average for smoother updates
        if self.training and not hard:
            # Moving average update for the mask to avoid rapid flickering
            current_active = (avg_s > self.pruning_threshold).float()
            self.active_mask.copy_(0.98 * self.active_mask + 0.02 * current_active)
            # Ensure at least one is always active to avoid collapse
            if self.active_mask.sum() == 0:
                self.active_mask[torch.argmax(avg_s)] = 1.0

        entropy = -torch.mean(torch.sum(s * torch.log(s + 1e-9), dim=1))
        diversity_loss = torch.sum(avg_s * torch.log(avg_s + 1e-9))
        pruning_loss = torch.mean(torch.abs(avg_s * (1 - self.active_mask))) # Penalize usage of "inactive" nodes

        # Sparsity loss to encourage finding the minimal scale
        sparsity_loss = torch.sum(self.active_mask) / self.n_super_nodes

        # NEW: Temporal consistency loss to prevent flickering
        temporal_consistency_loss = torch.tensor(0.0, device=x.device)
        if prev_assignments is not None and prev_assignments.size(0) > 0:
            # Compare current assignments with previous ones
            # Use MSE to penalize large changes in assignment probabilities
            temporal_consistency_loss = F.mse_loss(s, prev_assignments.expand_as(s).detach())
        
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
            'sparsity': sparsity_loss,
            'temporal_consistency': temporal_consistency_loss * self.temporal_consistency_weight
        }

        super_node_mu = None
        if pos is not None:
            s_pos_expanded = pos.unsqueeze(1) * s.unsqueeze(2)
            sum_s_pos = scatter(s_pos_expanded, batch, dim=0, reduce='sum')
            sum_s = scatter(s, batch, dim=0, reduce='sum').unsqueeze(-1) + 1e-9
            super_node_mu = sum_s_pos / sum_s

        if super_node_mu is not None:
            mu = super_node_mu
            dist_sq = torch.sum((mu.unsqueeze(1) - mu.unsqueeze(2))**2, dim=-1)
            mask_eye = torch.eye(self.n_super_nodes, device=x.device).unsqueeze(0)
            repulsion = 1.0 / (dist_sq + 1.0)
            separation_loss = (repulsion * (1 - mask_eye)).sum() / (self.n_super_nodes * (self.n_super_nodes - 1) + 1e-9)
            assign_losses['separation'] = separation_loss

        x_expanded = x.unsqueeze(1) * s.unsqueeze(2)
        out = scatter(x_expanded, batch, dim=0, reduce='sum')

        return out, s, assign_losses, super_node_mu

    def update_assignment_history(self, current_assignments):
        """
        Update the assignment history for temporal consistency.
        """
        avg_current = current_assignments.mean(dim=0)
        self.assignment_history.copy_(0.9 * self.assignment_history + 0.1 * avg_current)
        self.history_counter += 1

    def get_assignment_history(self):
        """
        Get the historical average assignments.
        """
        return self.assignment_history.clone()


class AdaptiveTauScheduler:
    """
    Adaptive temperature scheduler for Gumbel-Softmax to improve assignment stability.
    Starts with higher temperature for exploration, decreases for exploitation.
    """
    
    def __init__(self, initial_tau=1.0, final_tau=0.1, decay_steps=1000):
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def get_tau(self, progress_ratio=None):
        """
        Get the current temperature based on training progress.
        
        Args:
            progress_ratio: Float between 0 and 1 indicating training progress
        """
        if progress_ratio is None:
            # Calculate based on internal step counter
            progress_ratio = min(1.0, self.current_step / self.decay_steps)
        
        # Cosine annealing schedule
        import math
        tau = self.final_tau + 0.5 * (self.initial_tau - self.final_tau) * (1 + math.cos(math.pi * progress_ratio))
        
        return tau
    
    def step(self):
        """
        Increment the step counter.
        """
        self.current_step += 1