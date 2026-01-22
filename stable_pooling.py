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
                 temporal_consistency_weight=0.1, collapse_prevention_weight=2.0,
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
        self.temporal_consistency_weight = temporal_consistency_weight * 5  # Increased by factor of 5 as requested
        self.collapse_prevention_weight = collapse_prevention_weight
        self.sparsity_weight = sparsity_weight
        self.min_active_super_nodes = min(max(1, min_active_super_nodes), n_super_nodes)

        self.assign_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, n_super_nodes)
        )
        self.scaling = nn.Parameter(torch.tensor(10.0)) # Increased to 10.0 for stronger symmetry breaking
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

        # NEW: Assignment Persistence - bias logits by previous assignments to stabilize flickering
        if prev_assignments is not None and prev_assignments.size(0) == x.size(0):
            # persistence_gain = 2.0 to strongly favor previous identity
            logits = logits + 2.0 * prev_assignments.detach()

        # Apply active_mask to logits (soft mask to allow for revival)
        # Use detach() to prevent inplace modification errors during backward pass
        mask = self.active_mask.detach().unsqueeze(0)
        
        # During training, use a softer mask to allow potential reactivation
        if self.training:
            # -5.0 is enough to suppress but allows much more gradient flow than -10.0
            # Also add a small random exploration factor to logits of inactive nodes
            exploration = torch.randn_like(logits) * 0.01
            logits = logits + (mask - 1.0) * 5.0 + exploration
        else:
            # Harder mask during inference
            logits = logits.masked_fill(mask == 0, -1e9)

        s = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)

        avg_s = s.mean(dim=0)

        # Update active_mask with exponential moving average for smoother updates
        # We allow updates during both soft and hard training to maintain consistency
        if self.training:
            # Moving average update for the mask to avoid rapid flickering
            current_active = (avg_s > self.pruning_threshold).float()

            # STOCHASTIC REVIVAL: Occasionally give inactive nodes a chance to revive
            # if they show even minor signs of life (e.g. avg_s > pruning_threshold / 5)
            revival_threshold = self.pruning_threshold / 5.0
            revival_candidate = (avg_s > revival_threshold).float()

            # Combine current_active with a small probability of reviving revival_candidates
            revival_mask = (torch.rand_like(self.active_mask) < 0.1).float() * revival_candidate # Increased prob from 0.05
            effective_active = torch.clamp(current_active + revival_mask, 0, 1)

            # Use a much slower EMA for smoothness as requested
            ema_rate = 0.001 # Extremely smooth EMA (0.999 weight on previous)
            if hard:
                ema_rate *= 0.5 # Even slower updates during hard sampling

            self.active_mask.copy_((1.0 - ema_rate) * self.active_mask + ema_rate * effective_active)

            # Ensure minimum number of super-nodes remain active to prevent total collapse
            n_active_now = (self.active_mask > 0.5).sum().item()
            if n_active_now < self.min_active_super_nodes:
                # Force-revive the nodes with highest average logits to meet minimum requirement
                avg_logits = logits.mean(dim=0)
                _, most_needed_indices = torch.topk(avg_logits, self.min_active_super_nodes, largest=True)
                new_active = self.active_mask.clone()
                new_active[most_needed_indices] = 1.0
                self.active_mask.copy_(new_active)  # Direct assignment instead of EMA to enforce constraint

            # Double-check that the constraint is met after update
            n_active_final = (self.active_mask > 0.5).sum().item()
            if n_active_final < self.min_active_super_nodes:
                # If still below minimum, force activation of top nodes
                _, forced_indices = torch.topk(avg_s, self.min_active_super_nodes, largest=True)
                self.active_mask[forced_indices] = 1.0

        entropy = -torch.mean(torch.sum(s * torch.log(s + 1e-9), dim=1))
        
        # STABILITY FIX: Change diversity_loss from raw entropy to KL(Uniform || avg_s)
        # This makes the loss positive-definite and prevents the "negative divergence" trap.
        uniform_p = torch.full_like(avg_s, 1.0 / self.n_super_nodes)
        diversity_loss = torch.sum(uniform_p * torch.log(uniform_p / (avg_s + 1e-9)))
        
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

        return out, s, assign_losses, super_node_mu

    def _compute_collapse_prevention_loss(self, avg_assignments):
        """
        Compute loss to prevent all assignments from collapsing to a single super-node.
        Uses a combination of variance and an entropy-based penalty.
        """
        from common_losses import compute_collapse_prevention_loss as common_collapse_loss
        return common_collapse_loss(avg_assignments, self.n_super_nodes)

    def _compute_balance_loss(self, avg_assignments):
        """
        Compute loss to encourage balanced usage of super-nodes using KL divergence.
        """
        from common_losses import compute_balance_loss as common_balance_loss
        return common_balance_loss(avg_assignments, self.n_super_nodes)

    def apply_hard_revival(self):
        """
        Forcefully revive inactive nodes by re-initializing their assignment MLP weights
        if they have been inactive (active_mask near zero).
        Called every 50-100 epochs.
        """
        if not self.training:
            return

        inactive_indices = torch.where(self.active_mask < 0.1)[0]
        if len(inactive_indices) > 0:
            print(f"  [Hard Revival] Reviving super-nodes: {inactive_indices.tolist()}")
            # Re-initialize the final layer weights for inactive nodes to force new exploration
            with torch.no_grad():
                # self.assign_mlp[2] is the Linear(in_channels, n_super_nodes) layer
                last_layer = self.assign_mlp[2]
                for idx in inactive_indices:
                    # Re-initialize the weights for the corresponding output row
                    # We use a larger std to force exploration
                    nn.init.normal_(last_layer.weight[idx], mean=0.0, std=0.5)
                    nn.init.constant_(last_layer.bias[idx], 0.0)

                # Reset active mask for these nodes to allow them to compete immediately
                self.active_mask[inactive_indices] = 1.0


class DynamicLossBalancer:
    """
    Enhanced dynamic loss balancer that adjusts weights based on training progress,
    loss magnitudes, and relative importance of stability vs reconstruction.
    """

    def __init__(self, initial_weights=None, adaptation_rate=0.02, priority_losses=None):
        """
        Args:
            initial_weights: Dict of initial loss weights
            adaptation_rate: Rate at which weights adapt
            priority_losses: List of losses that should be prioritized if they don't decrease
        """
        self.initial_weights = initial_weights or {}
        self.adaptation_rate = adaptation_rate
        self.priority_losses = priority_losses or ['collapse_prevention', 'balance', 'entropy']
        self.current_weights = initial_weights.copy() if initial_weights else {}
        self.loss_history = {}
        self.step_count = 0

    def update_weights(self, current_losses):
        self.step_count += 1

        for loss_name, loss_value in current_losses.items():
            val = loss_value.item()
            if loss_name not in self.loss_history:
                self.loss_history[loss_name] = []
                self.current_weights[loss_name] = self.initial_weights.get(loss_name, 1.0)

            self.loss_history[loss_name].append(val)
            if len(self.loss_history[loss_name]) > 50:
                self.loss_history[loss_name].pop(0)

            # Weight adaptation logic
            if len(self.loss_history[loss_name]) >= 20:
                recent_avg = sum(self.loss_history[loss_name][-10:]) / 10
                older_avg = sum(self.loss_history[loss_name][:10]) / 10
                
                if older_avg > 0:
                    ratio = recent_avg / older_avg
                    
                    # If loss is not decreasing (ratio close to 1.0 or higher)
                    if ratio > 0.95:
                        # Prioritize stability and structural losses more aggressively
                        boost = 1.0 + self.adaptation_rate
                        if loss_name in self.priority_losses:
                            boost += self.adaptation_rate * 2.0
                        self.current_weights[loss_name] *= boost
                    # If loss is decreasing very fast, reduce weight
                    elif ratio < 0.2:
                        self.current_weights[loss_name] *= (1.0 - self.adaptation_rate)

                # Clamp weights
                min_w = 0.05 if loss_name not in self.priority_losses else 0.5
                max_w = 20.0
                self.current_weights[loss_name] = max(min_w, min(max_w, self.current_weights[loss_name]))

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
        
        # Calculate progress relative to the remaining steps after warmup
        progress = min(1.0, (self.current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps))
        
        if self.schedule_type == 'linear':
            weight = self.initial_weight + progress * (self.target_weight - self.initial_weight)
        elif self.schedule_type == 'cosine':
            import math
            weight = self.target_weight + 0.5 * (self.initial_weight - self.target_weight) * (1 + math.cos(math.pi * progress))
        elif self.schedule_type == 'sigmoid':
            # Sigmoid schedule for smoother transition
            # Stays low for longer if we shift the midpoint or increase steepness
            steepness = 12.0
            midpoint = 0.5
            import math
            # Shifted sigmoid to stay near zero longer
            sig = 1 / (1 + math.exp(-steepness * (progress - midpoint)))
            # Re-normalize sigmoid to [0, 1] range within the progress window
            sig_min = 1 / (1 + math.exp(-steepness * (0 - midpoint)))
            sig_max = 1 / (1 + math.exp(-steepness * (1 - midpoint)))
            sig = (sig - sig_min) / (sig_max - sig_min)
            
            weight = self.initial_weight + sig * (self.target_weight - self.initial_weight)
        else:
            weight = self.target_weight
            
        return weight
