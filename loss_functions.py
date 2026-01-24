import torch
import torch.nn as nn
import numpy as np

class LossTracker:
    """Tracks running averages and full history of loss components to help with balancing and visualization."""
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.history = {}
        self.weights = {}
        self.history_list = {}  # Store full history for plotting
        self.weights_list = {}

    def update(self, components, weights=None):
        for k, v in components.items():
            val = v.item() if hasattr(v, 'item') else v
            if k not in self.history:
                self.history[k] = val
                self.history_list[k] = [val]
            else:
                self.history[k] = self.alpha * self.history[k] + (1 - self.alpha) * val
                self.history_list[k].append(val)
        
        if weights is not None:
            for k, v in weights.items():
                val = v.item() if hasattr(v, 'item') else v
                if k not in self.weights:
                    self.weights[k] = val
                    self.weights_list[k] = [val]
                else:
                    self.weights[k] = self.alpha * self.weights[k] + (1 - self.alpha) * val
                    self.weights_list[k].append(val)

    def get_stats(self):
        # For plotting, we return the history lists
        return {**self.history_list, **{f"w_{k}": v for k, v in self.weights_list.items()}}

    def get_running_averages(self):
        return {**self.history, **{f"w_{k}": v for k, v in self.weights.items()}}

class GradNormBalancer(nn.Module):
    """
    Implementation of GradNorm: Gradient Normalization for Multi-Task Learning.
    Balances task weights by normalizing gradient magnitudes.
    """
    def __init__(self, n_tasks, alpha=1.5, device='cpu'):
        super().__init__()
        self.n_tasks = n_tasks
        self.alpha = alpha
        # Initialize weights to 1.0
        self.weights = nn.Parameter(torch.ones(n_tasks, device=device))
        self.initial_losses = None
        
    def get_weights(self):
        # Ensure weights stay positive and sum to n_tasks
        w = torch.softmax(self.weights, dim=0) * self.n_tasks
        return w

    def update(self, losses, shared_weights, optimizer):
        """
        losses: List of loss tensors
        shared_weights: Parameters common to all tasks (e.g., GNN encoder)
        """
        if self.initial_losses is None:
            self.initial_losses = torch.tensor([l.item() for l in losses], device=self.weights.device)
            return

        # 1. Compute L2 norm of gradients for each task
        norms = []
        for i, loss in enumerate(losses):
            # Compute gradient of task loss with respect to shared weights
            grad = torch.autograd.grad(loss, shared_weights, retain_graph=True, allow_unused=True)[0]
            if grad is not None:
                norms.append(torch.norm(self.weights[i] * grad, p=2))
            else:
                norms.append(torch.tensor(0.0, device=self.weights.device))
        
        norms = torch.stack(norms)
        avg_norm = norms.mean() + 1e-6

        # 2. Compute relative inverse training rate
        loss_ratios = torch.tensor([l.item() for l in losses], device=self.weights.device) / (self.initial_losses + 1e-6)
        # Prioritize tasks that are far from convergence
        inverse_train_rates = loss_ratios / (loss_ratios.mean() + 1e-6)

        # 3. Target norm
        target_norms = (avg_norm * (inverse_train_rates ** self.alpha)).detach()

        # 4. GradNorm loss
        grad_norm_loss = nn.functional.l1_loss(norms, target_norms)
        
        # 5. Backward for weights
        self.weights.grad = torch.autograd.grad(grad_norm_loss, self.weights)[0]

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, loss_rec, loss_cons):
        return loss_rec + loss_cons

class StructuralLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, loss_assign, loss_ortho, loss_conn, loss_pruning, loss_sparsity, loss_sep):
        return loss_assign + loss_ortho + loss_conn + loss_pruning + loss_sparsity + loss_sep

class PhysicalityLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, loss_align, loss_mi, loss_anchor, loss_curv, loss_activity, loss_l2, loss_lvr, loss_hinge, loss_smooth):
        return loss_align + loss_mi + loss_anchor + loss_curv + loss_activity + loss_l2 + loss_lvr + loss_hinge + loss_smooth

class SymbolicConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, loss_sym, stage2_factor, symbolic_weight):
        return loss_sym * stage2_factor * max(1.0, symbolic_weight)

class LossFactory:
    """Centralizes loss creation and meta-grouping."""
    @staticmethod
    def get_loss_groups():
        return {
            'ReconstructionLoss': ['rec', 'cons'],
            'StructuralLoss': ['assign', 'ortho', 'conn', 'pruning', 'sparsity', 'sep'],
            'PhysicalityLoss': ['align', 'mi', 'anchor', 'curv', 'activity', 'l2', 'lvr', 'hinge', 'smooth'],
            'SymbolicConsistencyLoss': ['sym']
        }
    
    @staticmethod
    def create_loss_modules():
        return {
            'ReconstructionLoss': ReconstructionLoss(),
            'StructuralLoss': StructuralLoss(),
            'PhysicalityLoss': PhysicalityLoss(),
            'SymbolicConsistencyLoss': SymbolicConsistencyLoss()
        }
