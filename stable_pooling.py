"""
Enhanced Hierarchical Pooling with improved assignment stability to prevent latent flickering
and resolution collapse.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import torch.nn.functional as F


class StableHierarchicalPooling(nn.Module):
    """
    Enhanced HierarchicalPooling with improved assignment stability to prevent latent flickering
    and resolution collapse.

    Key improvements:
    1. Temporal consistency loss to penalize rapid assignment changes
    2. Exponential moving average for smoother active mask updates
    3. Assignment persistence mechanism to maintain stable cluster identities
    4. Adaptive temperature scheduling for Gumbel-Softmax
    5. Enhanced resolution collapse prevention mechanisms
    6. Dynamic loss balancing to prevent single super-node dominance
    """

    def __init__(self, in_channels, n_super_nodes, pruning_threshold=0.01,
                 temporal_consistency_weight=0.1, collapse_prevention_weight=1.0,
                 sparsity_weight=0.01, min_active_super_nodes=1):
        """
        Initialize the stable hierarchical pooling layer.

        Args:
            in_channels (int): Number of input feature channels per node
            n_super_nodes (int): Number of super-nodes to pool to
            pruning_threshold (float): Threshold for pruning super-nodes
            temporal_consistency_weight (float): Weight for temporal consistency loss
            collapse_prevention_weight (float): Weight for collapse prevention loss
            sparsity_weight (float): Initial weight for sparsity loss
            min_active_super_nodes (int): Minimum number of super-nodes that must remain active
        """
        super(StableHierarchicalPooling, self).__init__()
        self.n_super_nodes = n_super_nodes
        self.pruning_threshold = pruning_threshold
        self.temporal_consistency_weight = temporal_consistency_weight
        self.collapse_prevention_weight = collapse_prevention_weight
        self.sparsity_weight = sparsity_weight
        self.min_active_super_nodes = min(max(1, min_active_super_nodes), n_super_nodes)

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

        # Track assignment distribution to detect collapse
        self.register_buffer('assignment_distribution', torch.ones(n_super_nodes) / n_super_nodes)
        self.register_buffer('assignment_variance', torch.zeros(n_super_nodes))
        
        # Adaptive sparsity scheduling
        self.current_sparsity_weight = sparsity_weight

    def set_sparsity_weight(self, weight):
        """Manually set or update the sparsity weight."""
        self.current_sparsity_weight = weight

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
                    'temporal_consistency': torch.tensor(0.0, device=x.device),
                    'collapse_prevention': torch.tensor(0.0, device=x.device),
                    'balance': torch.tensor(0.0, device=x.device)}, \
                   None

        logits = self.assign_mlp(x) * self.scaling

        # Apply active_mask to logits (soft mask to allow for revival)
        mask = self.active_mask.unsqueeze(0)
        
        # During training, use a softer mask to allow potential reactivation
        if self.training:
            # -10.0 is small enough to suppress but large enough to allow gradient flow
            logits = logits + (mask - 1.0) * 10.0
        else:
            # Harder mask during inference
            logits = logits.masked_fill(mask == 0, -1e9)

        s = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        avg_s = s.mean(dim=0)

        # Update active_mask with exponential moving average for smoother updates
        if self.training and not hard:
            # Moving average update for the mask to avoid rapid flickering
            current_active = (avg_s > self.pruning_threshold).float()
            self.active_mask.copy_(0.98 * self.active_mask + 0.02 * current_active)

            # Ensure minimum number of super-nodes remain active to prevent total collapse
            n_active_now = current_active.sum().item()
            if n_active_now < self.min_active_super_nodes:
                # Activate the least used super-nodes to meet minimum requirement
                _, least_used_indices = torch.topk(avg_s, self.min_active_super_nodes, largest=False)
                new_active = current_active.clone()
                new_active[least_used_indices] = 1.0
                self.active_mask.copy_(0.98 * self.active_mask + 0.02 * new_active)

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

        # NEW: Collapse prevention loss to ensure assignments are distributed
        # This penalizes situations where most assignments go to a single super-node
        collapse_prevention_loss = self._compute_collapse_prevention_loss(avg_s)

        # NEW: Balance loss to encourage equal usage of super-nodes
        balance_loss = self._compute_balance_loss(avg_s)

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
            'sparsity': sparsity_loss * self.current_sparsity_weight,
            'temporal_consistency': temporal_consistency_loss * self.temporal_consistency_weight,
            'collapse_prevention': collapse_prevention_loss * self.collapse_prevention_weight,
            'balance': balance_loss
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

    def _compute_collapse_prevention_loss(self, avg_assignments):
        """
        Compute loss to prevent all assignments from collapsing to a single super-node.

        Args:
            avg_assignments: Average assignment probabilities [n_super_nodes]

        Returns:
            collapse_loss: Scalar loss value
        """
        # Compute variance of assignment probabilities
        # High variance indicates collapse (one dominates)
        variance = torch.var(avg_assignments)

        # Also penalize max probability being too high (indicating dominance)
        max_prob = torch.max(avg_assignments)

        # Combine both measures
        collapse_loss = variance + torch.relu(max_prob - 1.0/self.n_super_nodes)

        return collapse_loss

    def _compute_balance_loss(self, avg_assignments):
        """
        Compute loss to encourage balanced usage of super-nodes.

        Args:
            avg_assignments: Average assignment probabilities [n_super_nodes]

        Returns:
            balance_loss: Scalar loss value
        """
        # Target is uniform distribution
        uniform_prob = 1.0 / self.n_super_nodes
        uniform_dist = torch.full_like(avg_assignments, uniform_prob)

        # Use KL divergence to encourage uniform distribution
        kl_div = torch.sum(avg_assignments * torch.log(avg_assignments / (uniform_dist + 1e-9)))

        return kl_div

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
    """

    def __init__(self, initial_tau=1.0, final_tau=0.1, decay_steps=1000,
                 adaptive_collapse_protection=True):
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.decay_steps = decay_steps
        self.current_step = 0
        self.adaptive_collapse_protection = adaptive_collapse_protection

    def get_tau(self, progress_ratio=None, assignment_stats=None):
        """
        Get the current temperature based on training progress and assignment statistics.

        Args:
            progress_ratio: Float between 0 and 1 indicating training progress
            assignment_stats: Dict with assignment statistics for adaptive adjustment
        """
        if progress_ratio is None:
            # Calculate based on internal step counter
            progress_ratio = min(1.0, self.current_step / self.decay_steps)

        # Cosine annealing schedule
        import math
        tau = self.final_tau + 0.5 * (self.initial_tau - self.final_tau) * (1 + math.cos(math.pi * progress_ratio))

        # If adaptive collapse protection is enabled and we have assignment stats
        if self.adaptive_collapse_protection and assignment_stats is not None:
            # Increase temperature if assignments are becoming too peaked (collapse risk)
            avg_assignments = assignment_stats.get('avg_assignments', None)
            if avg_assignments is not None:
                max_assignment = torch.max(avg_assignments).item()
                # If max assignment probability is too high, increase temperature to encourage exploration
                if max_assignment > 0.8:  # Threshold for collapse detection
                    tau *= 1.5  # Increase temperature to encourage more uniform assignments

        return tau

    def step(self):
        """
        Increment the step counter.
        """
        self.current_step += 1


class DynamicLossBalancer:
    """
    Dynamic loss balancer that adjusts weights based on training progress and loss magnitudes.
    This helps prevent resolution collapse by dynamically adjusting the importance of different losses.
    """

    def __init__(self, initial_weights=None, adaptation_rate=0.01):
        """
        Initialize the dynamic loss balancer.

        Args:
            initial_weights: Dict of initial loss weights
            adaptation_rate: Rate at which weights adapt (smaller = slower adaptation)
        """
        self.initial_weights = initial_weights or {}
        self.adaptation_rate = adaptation_rate
        self.current_weights = initial_weights.copy() if initial_weights else {}
        self.loss_history = {}
        self.step_count = 0

    def update_weights(self, current_losses):
        """
        Update loss weights based on current loss values.

        Args:
            current_losses: Dict of current loss values
        """
        self.step_count += 1

        for loss_name, loss_value in current_losses.items():
            # Initialize if this is the first time seeing this loss
            if loss_name not in self.loss_history:
                self.loss_history[loss_name] = []
                self.current_weights[loss_name] = self.initial_weights.get(loss_name, 1.0)

            # Add current loss to history
            self.loss_history[loss_name].append(loss_value.item())

            # Keep only recent history (last 100 steps)
            if len(self.loss_history[loss_name]) > 100:
                self.loss_history[loss_name] = self.loss_history[loss_name][-100:]

            # If we have enough history, adapt the weight
            if len(self.loss_history[loss_name]) >= 10:
                recent_avg = sum(self.loss_history[loss_name][-10:]) / 10
                initial_avg = sum(self.loss_history[loss_name][:10]) / min(10, len(self.loss_history[loss_name]))

                # If this loss is decreasing much slower than others, increase its weight
                if initial_avg > 0 and recent_avg / initial_avg > 0.9:  # Still high relative to initial
                    self.current_weights[loss_name] *= (1 + self.adaptation_rate)
                elif recent_avg < initial_avg * 0.1:  # Decreasing very fast, decrease weight
                    self.current_weights[loss_name] *= (1 - self.adaptation_rate)

                # Clamp weights to reasonable range
                self.current_weights[loss_name] = max(0.1, min(10.0, self.current_weights[loss_name]))

    def get_balanced_losses(self, raw_losses):
        """
        Apply current weights to raw losses.

        Args:
            raw_losses: Dict of raw loss values

        Returns:
            weighted_losses: Dict of weighted loss values
        """
        self.update_weights(raw_losses)

        weighted_losses = {}
        for loss_name, loss_value in raw_losses.items():
            weight = self.current_weights.get(loss_name, 1.0)
            weighted_losses[loss_name] = loss_value * weight

        return weighted_losses


class SparsityScheduler:
    """
    Sparsity scheduler that gradually increases sparsity pressure.
    This prevents resolution collapse by allowing the model to find a good
    representation before aggressively pruning super-nodes.
    """

    def __init__(self, initial_weight=0.0, target_weight=0.1, warmup_steps=1000, 
                 max_steps=5000, schedule_type='sigmoid'):
        self.initial_weight = initial_weight
        self.target_weight = target_weight
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.schedule_type = schedule_type
        self.current_step = 0

    def step(self):
        self.current_step += 1
        return self.get_weight()

    def get_weight(self):
        if self.current_step < self.warmup_steps:
            return self.initial_weight
        
        progress = min(1.0, (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps))
        
        if self.schedule_type == 'linear':
            weight = self.initial_weight + progress * (self.target_weight - self.initial_weight)
        elif self.schedule_type == 'cosine':
            import math
            weight = self.target_weight + 0.5 * (self.initial_weight - self.target_weight) * (1 + math.cos(math.pi * progress))
        elif self.schedule_type == 'sigmoid':
            # Sigmoid schedule for smoother transition
            steepness = 10
            midpoint = 0.5
            import math
            sig = 1 / (1 + math.exp(-steepness * (progress - midpoint)))
            weight = self.initial_weight + sig * (self.target_weight - self.initial_weight)
        else:
            weight = self.target_weight
            
        return weight
