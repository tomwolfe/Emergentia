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
                 sparsity_weight=0.01, min_active_super_nodes=4):
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
        self.temporal_consistency_weight = temporal_consistency_weight * 2 # Reduced from 50x to 2x to prevent gradient interference
        self.collapse_prevention_weight = collapse_prevention_weight
        self.sparsity_weight = sparsity_weight
        self.min_active_super_nodes = min(max(1, min_active_super_nodes), n_super_nodes)

        self.assign_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, n_super_nodes)
        )
        self.scaling = nn.Parameter(torch.tensor(5.0)) # Kept at 5.0 to prevent softmax saturation
        self.register_buffer('active_mask', torch.ones(n_super_nodes))

        # Track previous assignments for temporal consistency
        self.register_buffer('prev_assignments', torch.zeros(n_super_nodes))
        self.register_buffer('assignment_history', torch.zeros(n_super_nodes))
        self.register_buffer('history_counter', torch.zeros(1, dtype=torch.long))

        # Track assignment distribution to detect collapse
        self.register_buffer('assignment_distribution', torch.ones(n_super_nodes) / n_super_nodes)
        self.register_buffer('assignment_variance', torch.zeros(n_super_nodes))
        
        # NEW: Logit smoothing EMA buffer
        self.register_buffer('logits_ema', None)
        self.logit_ema_alpha = 0.5 # Smoothing factor
        
        # Adaptive sparsity scheduling
        self.current_sparsity_weight = sparsity_weight

    def set_sparsity_weight(self, weight):
        """Manually set or update the sparsity weight."""
        self.current_sparsity_weight = weight

    def sinkhorn_knopp(self, logits, tau=1.0, iterations=20):
        """
        Sinkhorn-Knopp algorithm to find a doubly stochastic assignment matrix.
        Ensures each node is assigned to super-nodes (row sum=1) and each super-node
        receives approximately N/K nodes (column sum=N/K).
        Reduced iterations to 20 for speed.
        """
        N, K = logits.shape
        # Use log-space for stability
        P = logits / tau
        
        # Target column sum: N/K
        log_target_col_sum = torch.log(torch.tensor(N / K, device=logits.device, dtype=logits.dtype) + 1e-9)
        
        for _ in range(iterations):
            # Row normalization: log(P) = log(P) - logsumexp(log(P), dim=1)
            P = P - torch.logsumexp(P, dim=1, keepdim=True)
            # Column normalization: log(P) = log(P) - logsumexp(log(P), dim=0) + log(target)
            P = P - torch.logsumexp(P, dim=0, keepdim=True) + log_target_col_sum
            
        return torch.exp(P)

    def forward(self, x, batch, pos=None, tau=1.0, hard=False, prev_assignments=None, current_epoch=None, total_epochs=None):
        """
        Forward pass of hierarchical pooling with Sinkhorn-Knopp stability.
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
                    'balance': torch.tensor(0.0, device=x.device)},	\
                   None

        # Ensure minimum number of super-nodes remain active
        n_active_pre = (self.active_mask > 0.5).sum().item()
        if n_active_pre < self.min_active_super_nodes:
            if self.history_counter > 0:
                _, most_needed_indices = torch.topk(self.assignment_history, self.min_active_super_nodes, largest=True)
            else:
                most_needed_indices = torch.arange(self.min_active_super_nodes, device=x.device)
            self.active_mask.zero_()
            self.active_mask[most_needed_indices] = 1.0

        logits = self.assign_mlp(x) * self.scaling

        # Logit Smoothing (EMA) to prevent "banding" and temporal chatter
        if self.logits_ema is not None and self.logits_ema.size(0) == logits.size(0):
            logits = self.logit_ema_alpha * self.logits_ema.detach() + (1.0 - self.logit_ema_alpha) * logits
        self.logits_ema = logits.detach()

        # Assignment Persistence
        if prev_assignments is not None and prev_assignments.size(0) == x.size(0):
            logits = logits + 5.0 * prev_assignments.detach()

        # Apply Sinkhorn-Knopp instead of Gumbel-Softmax
        s = self.sinkhorn_knopp(logits, tau=tau, iterations=20)

        if hard or (current_epoch is not None and total_epochs is not None and current_epoch/total_epochs >= 0.8):
            # Straight-through estimator for hard assignments
            s_hard = torch.zeros_like(s).scatter_(-1, torch.argmax(s, dim=-1, keepdim=True), 1.0)
            s = (s_hard - s).detach() + s

        avg_s = s.mean(dim=0)

        # COMPETITIVE DROPOUT: Randomly zero out most active super-node during training to force distribution
        if self.training and torch.rand(1).item() < 0.1:
            most_active_idx = torch.argmax(avg_s)
            dropout_mask = torch.ones_like(s)
            dropout_mask[:, most_active_idx] = 0.0
            s = s * dropout_mask
            s = s / (s.sum(dim=-1, keepdim=True) + 1e-9)
            avg_s = s.mean(dim=0)

        self.assignment_history.copy_(0.9 * self.assignment_history + 0.1 * avg_s.detach())
        self.history_counter += 1

        if self.training:
            current_active = (avg_s > self.pruning_threshold).float()
            revival_threshold = self.pruning_threshold / 5.0
            revival_candidate = (avg_s > revival_threshold).float()
            revival_mask = (torch.rand_like(self.active_mask) < 0.2).float() * revival_candidate
            effective_active = torch.clamp(current_active + revival_mask, 0, 1)

            ema_rate = 0.05
            self.active_mask.copy_((1.0 - ema_rate) * self.active_mask + ema_rate * effective_active)

            n_active_now = (self.active_mask > 0.5).sum().item()
            if n_active_now < self.min_active_super_nodes:
                _, most_needed_indices = torch.topk(avg_s, self.min_active_super_nodes, largest=True)
                new_active = self.active_mask.clone()
                new_active.zero_()
                new_active[most_needed_indices] = 1.0
                self.active_mask.copy_(new_active)

        final_check = (self.active_mask > 0.5).sum().item()
        if final_check < self.min_active_super_nodes:
            _, emergency_indices = torch.topk(avg_s, self.min_active_super_nodes, largest=True)
            self.active_mask.zero_()
            self.active_mask[emergency_indices] = 1.0

        entropy = -torch.mean(torch.sum(s * torch.log(s + 1e-9), dim=1))
        uniform_p = torch.full_like(avg_s, 1.0 / self.n_super_nodes)
        diversity_loss = torch.sum(uniform_p * torch.log(uniform_p / (avg_s + 1e-9)))
        pruning_loss = torch.mean(torch.abs(avg_s * (1 - self.active_mask)))
        sparsity_loss = torch.sum(self.active_mask) / self.n_super_nodes

        temporal_consistency_loss = torch.tensor(0.0, device=x.device)
        if prev_assignments is not None and prev_assignments.size(0) > 0:
            temporal_consistency_loss = F.mse_loss(s, prev_assignments.expand_as(s).detach())

        collapse_prevention_loss = self._compute_collapse_prevention_loss(avg_s)
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
            'diversity': diversity_loss * 1.0,
            'spatial': spatial_loss,
            'pruning': pruning_loss,
            'sparsity': sparsity_loss * self.current_sparsity_weight,
            'temporal_consistency': temporal_consistency_loss * self.temporal_consistency_weight,
            'collapse_prevention': collapse_prevention_loss * (self.collapse_prevention_weight * 5.0),
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
            out = torch.zeros((0, self.n_super_nodes, x.size(1)), device=x.device, dtype=x.dtype)

        return out, s, assign_losses, super_node_mu

    def _compute_collapse_prevention_loss(self, avg_assignments):
        from common_losses import compute_collapse_prevention_loss as common_collapse_loss
        return common_collapse_loss(avg_assignments, self.n_super_nodes)

    def _compute_balance_loss(self, avg_assignments):
        from common_losses import compute_balance_loss as common_balance_loss
        loss = common_balance_loss(avg_assignments, self.n_super_nodes)
        loss += 5.0 * (avg_assignments.max() - avg_assignments.min())
        return loss

    def apply_hard_revival(self):
        if not self.training:
            return
        print(f"  [Sinkhorn Revival] Resetting active mask to encourage redistribution...")
        self.active_mask.fill_(1.0)
        self.logits_ema = None


class DynamicLossBalancer:
    def __init__(self, initial_weights=None, adaptation_rate=0.02, priority_losses=None):
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
            if len(self.loss_history[loss_name]) >= 20:
                recent_avg = sum(self.loss_history[loss_name][-10:]) / 10
                older_avg = sum(self.loss_history[loss_name][:10]) / 10
                if older_avg > 0:
                    ratio = recent_avg / older_avg
                    if ratio > 0.95:
                        boost = 1.0 + self.adaptation_rate
                        if loss_name in self.priority_losses:
                            boost += self.adaptation_rate * 2.0
                        self.current_weights[loss_name] *= boost
                    elif ratio < 0.2:
                        self.current_weights[loss_name] *= (1.0 - self.adaptation_rate)
                min_w = 0.05 if loss_name not in self.priority_losses else 0.5
                max_w = 20.0
                self.current_weights[loss_name] = max(min_w, min(max_w, self.current_weights[loss_name]))

    def get_balanced_losses(self, raw_losses):
        self.update_weights(raw_losses)
        weighted_losses = {}
        for loss_name, loss_value in raw_losses.items():
            weight = self.current_weights.get(loss_name, 1.0)
            weighted_losses[loss_name] = loss_value * weight
        return weighted_losses


class SparsityScheduler:
    def __init__(self, initial_weight=0.0, target_weight=0.1, warmup_steps=1000, 
                 max_steps=5000, schedule_type='sigmoid'):
        self.initial_weight = initial_weight
        self.target_weight = target_weight
        self.base_target_weight = target_weight
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.schedule_type = schedule_type
        self.current_step = 0
        self.last_snr = 1.0

    def step(self):
        self.current_step += 1
        return self.get_weight()

    def adjust_to_snr(self, snr):
        self.last_snr = snr
        if snr < 1.0:
            reduction_factor = max(0.1, snr)
            self.target_weight = self.base_target_weight * reduction_factor
        else:
            self.target_weight = self.base_target_weight

    def get_weight(self):
        if self.current_step < self.warmup_steps:
            return self.initial_weight
        progress = min(1.0, (self.current_step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps))
        if self.schedule_type == 'linear':
            weight = self.initial_weight + progress * (self.target_weight - self.initial_weight)
        elif self.schedule_type == 'cosine':
            import math
            weight = self.target_weight + 0.5 * (self.initial_weight - self.target_weight) * (1 + math.cos(math.pi * progress))
        elif self.schedule_type == 'sigmoid':
            steepness = 12.0
            midpoint = 0.5
            import math
            sig = 1 / (1 + math.exp(-steepness * (progress - midpoint)))
            sig_min = 1 / (1 + math.exp(-steepness * (0 - midpoint)))
            sig_max = 1 / (1 + math.exp(-steepness * (1 - midpoint)))
            sig = (sig - sig_min) / (sig_max - sig_min)
            weight = self.initial_weight + sig * (self.target_weight - self.initial_weight)
        else:
            weight = self.target_weight
        return weight
