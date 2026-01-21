"""
Enhanced Dynamic Loss Balancer with gradient-based weighting to address
the complex loss landscape hyper-dimensionality issue.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, deque
import math


class GradientBasedLossBalancer:
    """
    Advanced loss balancer that uses gradient information to balance losses,
    addressing the hyper-dimensional loss landscape problem.
    """
    
    def __init__(self, initial_weights=None, adaptation_rate=0.02, 
                 gradient_momentum=0.9, history_window=50):
        """
        Args:
            initial_weights: Dict of initial loss weights
            adaptation_rate: Rate at which weights adapt based on gradients
            gradient_momentum: Momentum for gradient norm tracking
            history_window: Window size for tracking loss history
        """
        self.initial_weights = initial_weights or {}
        self.adaptation_rate = adaptation_rate
        self.gradient_momentum = gradient_momentum
        self.history_window = history_window
        
        # Current weights for each loss term
        self.current_weights = initial_weights.copy() if initial_weights else {}
        
        # Track loss histories
        self.loss_histories = defaultdict(lambda: deque(maxlen=history_window))
        self.gradient_norm_histories = defaultdict(lambda: deque(maxlen=history_window))
        
        # Track running averages of gradient norms
        self.running_gradient_norms = {}
        self.step_count = 0
        
        # Priority losses that should maintain minimum influence
        self.priority_losses = {
            'collapse_prevention', 'balance', 'entropy', 'temporal_consistency'
        }
    
    def update_weights(self, current_losses, model_parameters=None):
        """
        Update loss weights based on current losses and gradients.
        
        Args:
            current_losses: Dict of current loss values
            model_parameters: Model parameters for gradient computation
        """
        self.step_count += 1
        
        # Update loss histories
        for loss_name, loss_value in current_losses.items():
            self.loss_histories[loss_name].append(loss_value.item())
        
        # Compute and track gradient norms if parameters are provided
        if model_parameters is not None:
            self._update_gradient_norms(current_losses, model_parameters)
        
        # Adapt weights based on loss trends and gradient information
        for loss_name in current_losses.keys():
            # Get recent loss trend
            recent_losses = list(self.loss_histories[loss_name])
            if len(recent_losses) < 10:
                continue
                
            # Calculate trend (slope of last 10 losses)
            recent_x = list(range(len(recent_losses[-10:])))
            recent_y = recent_losses[-10:]
            if len(set(recent_y)) > 1:  # Only if there's variation
                slope = np.polyfit(recent_x, recent_y, 1)[0]
            else:
                slope = 0.0
            
            # Get gradient norm for this loss (if available)
            grad_norm = self.running_gradient_norms.get(loss_name, 1.0)
            
            # Base weight adjustment based on loss trend
            if slope > 0:  # Loss is increasing
                adjustment = 1.0 + self.adaptation_rate * 2.0  # Increase weight more aggressively
            elif slope < -0.1:  # Loss is decreasing rapidly
                adjustment = max(0.9, 1.0 - self.adaptation_rate)  # Decrease weight moderately
            else:  # Loss is stable
                adjustment = 1.0
            
            # Adjust based on gradient norm (relative to other losses)
            if len(self.running_gradient_norms) > 1:
                avg_grad_norm = np.mean(list(self.running_gradient_norms.values()))
                if avg_grad_norm > 0:
                    grad_ratio = grad_norm / avg_grad_norm
                    # If this loss has smaller gradients, increase its weight to compensate
                    grad_adjustment = 1.0 / max(grad_ratio, 0.1)
                    adjustment *= grad_adjustment
            
            # Apply priority adjustments for critical losses
            if loss_name in self.priority_losses:
                # Priority losses should not be reduced too much
                adjustment = max(adjustment, 0.8)
            
            # Update weight with the adjustment
            current_weight = self.current_weights.get(loss_name, 
                                                   self.initial_weights.get(loss_name, 1.0))
            new_weight = current_weight * adjustment
            
            # Apply bounds to prevent extreme weights
            min_weight = 0.01 if loss_name not in self.priority_losses else 0.1
            max_weight = 100.0
            new_weight = max(min_weight, min(max_weight, new_weight))
            
            self.current_weights[loss_name] = new_weight
    
    def _update_gradient_norms(self, current_losses, model_parameters):
        """Update running averages of gradient norms for each loss."""
        for loss_name, loss_value in current_losses.items():
            # Compute gradient norm for this specific loss
            if loss_value.requires_grad:
                try:
                    # Detach and clone parameters to avoid modifying gradients
                    params_copy = []
                    for param in model_parameters:
                        if param.grad is not None:
                            params_copy.append(param.grad.detach().clone().flatten())
                    
                    if params_copy:
                        all_grads = torch.cat(params_copy)
                        grad_norm = torch.norm(all_grads).item()
                        
                        # Update running average with momentum
                        if loss_name in self.running_gradient_norms:
                            self.running_gradient_norms[loss_name] = (
                                self.gradient_momentum * self.running_gradient_norms[loss_name] + 
                                (1.0 - self.gradient_momentum) * grad_norm
                            )
                        else:
                            self.running_gradient_norms[loss_name] = grad_norm
                except:
                    # If gradient computation fails, use a default value
                    if loss_name not in self.running_gradient_norms:
                        self.running_gradient_norms[loss_name] = 1.0
            else:
                # If loss doesn't require grad, use a default value
                if loss_name not in self.running_gradient_norms:
                    self.running_gradient_norms[loss_name] = 1.0
    
    def get_balanced_losses(self, raw_losses, model_parameters=None):
        """
        Apply current weights to raw losses with gradient-based adjustments.
        
        Args:
            raw_losses: Dict of raw loss values
            model_parameters: Model parameters for gradient computation
            
        Returns:
            Dict of weighted loss values
        """
        # Update weights based on current state
        self.update_weights(raw_losses, model_parameters)
        
        weighted_losses = {}
        for loss_name, loss_value in raw_losses.items():
            weight = self.current_weights.get(loss_name, 1.0)
            weighted_losses[loss_name] = loss_value * weight
        
        return weighted_losses


class MultiScaleLossBalancer:
    """
    Loss balancer that operates at multiple scales (component, group, total)
    to handle the complex multi-component loss landscape.
    """
    
    def __init__(self, component_groups=None, adaptation_rates=None):
        """
        Args:
            component_groups: Dict mapping group names to lists of loss components
            adaptation_rates: Dict mapping group names to adaptation rates
        """
        self.component_groups = component_groups or {
            'reconstruction': ['rec', 'l2'],
            'structural': ['entropy', 'diversity', 'sparsity', 'pruning', 'balance'],
            'stability': ['temporal_consistency', 'collapse_prevention', 'cons'],
            'alignment': ['align', 'mi', 'ortho']
        }
        
        self.adaptation_rates = adaptation_rates or {
            'reconstruction': 0.01,
            'structural': 0.02,
            'stability': 0.03,
            'alignment': 0.01
        }
        
        # Individual component balancers
        self.component_balancers = {}
        for group_name, components in self.component_groups.items():
            rate = self.adaptation_rates.get(group_name, 0.02)
            self.component_balancers[group_name] = GradientBasedLossBalancer(
                adaptation_rate=rate
            )
        
        # Group-level balancer
        self.group_balancer = GradientBasedLossBalancer(adaptation_rate=0.01)
        
        # Track group losses
        self.group_losses = {}
    
    def get_balanced_losses(self, raw_losses, model_parameters=None):
        """
        Apply multi-scale balancing to raw losses.
        
        Args:
            raw_losses: Dict of raw loss values
            model_parameters: Model parameters for gradient computation
            
        Returns:
            Dict of hierarchically weighted loss values
        """
        # First, balance within each group
        group_weighted_losses = {}
        for group_name, components in self.component_groups.items():
            # Extract losses for this group
            group_raw_losses = {k: v for k, v in raw_losses.items() if k in components}
            
            if group_raw_losses:
                # Balance within the group
                balancer = self.component_balancers[group_name]
                group_weighted = balancer.get_balanced_losses(
                    group_raw_losses, model_parameters
                )
                
                # Store for group-level balancing
                group_total = sum(group_weighted.values())
                self.group_losses[group_name] = group_total
                
                # Add to overall results
                group_weighted_losses.update(group_weighted)
        
        # Then, balance between groups
        if self.group_losses:
            group_level_weighted = self.group_balancer.get_balanced_losses(
                self.group_losses, model_parameters
            )
            
            # Apply group-level weights to individual components
            final_losses = {}
            for group_name, components in self.component_groups.items():
                if group_name in group_level_weighted:
                    group_multiplier = group_level_weighted[group_name] / max(self.group_losses[group_name], 1e-8)
                    for comp_name in components:
                        if comp_name in group_weighted_losses:
                            final_losses[comp_name] = group_weighted_losses[comp_name] * group_multiplier
                        elif comp_name in raw_losses:
                            # Apply group multiplier to ungrouped losses too
                            final_losses[comp_name] = raw_losses[comp_name] * group_multiplier
            return final_losses
        else:
            # Fallback to component-level balancing only
            return group_weighted_losses


class AdaptiveWarmupScheduler:
    """
    Adaptive warmup scheduler that adjusts the training schedule based on
    loss convergence and stability metrics.
    """
    
    def __init__(self, total_steps, warmup_ratio=0.1, stability_threshold=0.05):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.stability_threshold = stability_threshold
        
        self.loss_tracker = defaultdict(lambda: deque(maxlen=20))
        self.step = 0
    
    def get_current_phase(self, current_losses):
        """
        Determine current training phase based on loss stability.
        
        Returns:
            float: Phase indicator (0.0 to 1.0)
        """
        self.step += 1
        
        # Update loss trackers
        for name, value in current_losses.items():
            self.loss_tracker[name].append(value.item())
        
        # Calculate stability metric
        stability_score = 1.0
        if len(list(self.loss_tracker.values())[0]) >= 10 if self.loss_tracker else False:
            for loss_values in self.loss_tracker.values():
                if len(loss_values) >= 10:
                    recent_values = list(loss_values)[-10:]
                    volatility = np.std(recent_values) / (np.mean(np.abs(recent_values)) + 1e-8)
                    stability_score = min(stability_score, 1.0 - min(volatility / self.stability_threshold, 1.0))
        
        # Base phase on step count but modulate with stability
        base_phase = min(1.0, self.step / self.total_steps)
        
        # If we're past warmup but losses are still unstable, extend the adjustment period
        if self.step > self.warmup_steps and stability_score < 0.7:
            # Slow down progression if losses are unstable
            adjusted_phase = max(base_phase * 0.7, (self.step - self.warmup_steps) / (self.total_steps * 0.5))
            return min(adjusted_phase, 1.0)
        
        return base_phase
    
    def is_warmup_complete(self, current_losses):
        """Check if warmup phase is complete."""
        if self.step < self.warmup_steps:
            return False
        
        # Check if key losses have stabilized
        key_losses = ['rec', 'cons', 'entropy', 'diversity']
        for loss_name in key_losses:
            if loss_name in self.loss_tracker:
                values = list(self.loss_tracker[loss_name])
                if len(values) >= 10:
                    recent_values = values[-10:]
                    if np.std(recent_values) > self.stability_threshold:
                        return False
        
        return True


def create_enhanced_loss_balancer(strategy='gradient_based', **kwargs):
    """
    Factory function to create an enhanced loss balancer.
    
    Args:
        strategy: Type of balancer ('gradient_based', 'multi_scale', 'adaptive')
        **kwargs: Additional arguments for the specific balancer
        
    Returns:
        Appropriate loss balancer instance
    """
    if strategy == 'multi_scale':
        return MultiScaleLossBalancer(**kwargs)
    elif strategy == 'adaptive':
        return AdaptiveWarmupScheduler(**kwargs)
    else:
        return GradientBasedLossBalancer(**kwargs)