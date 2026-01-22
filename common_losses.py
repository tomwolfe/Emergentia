"""
Common loss functions extracted from multiple modules to reduce duplication.
This addresses the DRY violations identified in the codebase.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


def get_latent_variance_loss(z):
    """
    Explicitly penalize low latent variance to prevent manifold collapse.
    z: [B, K, D] or [T, B, K, D]
    """
    # Calculate variance across the feature dimension D for each super-node
    # We want the latent space to be utilized, so we penalize low variance
    var = z.var(dim=-1).mean()
    # Hinge variance loss: only penalize if std < 1.0
    # This prevents the infinite gradient singularity of -log(var)
    return torch.relu(1.0 - torch.sqrt(var + 1e-6))


def get_mi_loss(z, mu, mi_discriminator):
    """
    Unsupervised alignment loss via Mutual Information Maximization (MINE).
    z: [B, K, D], mu: [B, K, 2]
    """
    # Take first 2 dims of z for spatial alignment
    z_spatial = z[:, :, :2]

    # Joint distribution
    joint = mi_discriminator(z_spatial, mu)

    # Marginal distribution (shuffle mu across batch and super-nodes)
    batch_size, n_k, _ = mu.shape
    mu_shuffled = mu[torch.randperm(batch_size)]
    # Also shuffle super-nodes to break local correlation
    mu_shuffled = mu_shuffled[:, torch.randperm(n_k)]

    marginal = mi_discriminator(z_spatial, mu_shuffled)

    # MINE objective: I(Z; MU) >= E[joint] - log(E[exp(marginal)])
    # We want to maximize this, so we minimize the negative
    mi_est = torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal)) + 1e-9)
    return -mi_est


def get_ortho_loss(s):
    """
    Orthogonality loss to encourage independence between super-nodes.
    s: [N, n_super_nodes]
    """
    if s.size(0) == 0:
        return torch.tensor(0.0, device=s.device)
    n_nodes, k = s.shape
    dots = torch.matmul(s.t(), s)
    identity = torch.eye(k, device=s.device).mul_(n_nodes / k)
    return torch.mean((dots - identity)**2)


def get_connectivity_loss(s, edge_index):
    """
    Connectivity loss to encourage connected nodes to have similar assignments.
    s: [N, n_super_nodes]
    edge_index: [2, E] where E is number of edges
    """
    if edge_index.numel() == 0:
        return torch.tensor(0.0, device=s.device)
    row, col = edge_index
    s_i = s[row]
    s_j = s[col]
    return torch.mean((s_i - s_j)**2)


def compute_collapse_prevention_loss(avg_assignments, n_super_nodes):
    """
    Compute loss to prevent all assignments from collapsing to a single super-node.
    Uses a combination of variance and an entropy-based penalty.
    
    Args:
        avg_assignments: Average assignment probabilities [n_super_nodes]
        n_super_nodes: Number of super nodes
    
    Returns:
        Collapse prevention loss value
    """
    # Variance penalty: high variance means one node dominates
    variance = torch.var(avg_assignments)

    # Maximum probability penalty: prevent any single node from taking too much share
    max_prob = torch.max(avg_assignments)
    max_penalty = torch.pow(torch.relu(max_prob - 0.8), 2) * 10.0

    # Entropy of the average assignments: should be high for diversity
    # Note: avg_assignments is already summed/averaged across batch
    ent_avg = -torch.sum(avg_assignments * torch.log(avg_assignments + 1e-9))
    target_ent = torch.log(torch.tensor(float(n_super_nodes), device=avg_assignments.device))
    ent_penalty = torch.pow(torch.relu(0.5 * target_ent - ent_avg), 2)

    return variance + max_penalty + ent_penalty


def compute_balance_loss(avg_assignments, n_super_nodes):
    """
    Compute loss to encourage balanced usage of super-nodes using KL divergence.
    
    Args:
        avg_assignments: Average assignment probabilities [n_super_nodes]
        n_super_nodes: Number of super nodes
    
    Returns:
        Balance loss value
    """
    # Target is uniform distribution
    uniform_prob = 1.0 / n_super_nodes
    uniform_dist = torch.full_like(avg_assignments, uniform_prob)

    # Use KL divergence to encourage uniform distribution
    # Higher weight when we are far from uniform
    kl_div = torch.sum(uniform_dist * torch.log(uniform_dist / (avg_assignments + 1e-9)))

    return kl_div


def compute_assignment_loss(losses_t, entropy_weight):
    """
    Compute assignment loss from individual components.
    
    Args:
        losses_t: Dictionary containing individual loss components
        entropy_weight: Weight for entropy loss
    
    Returns:
        Combined assignment loss
    """
    return (
        entropy_weight * losses_t['entropy'] +
        5.0 * losses_t['diversity'] +
        1.0 * losses_t['spatial'] + # Increased 10x
        2.0 * losses_t.get('collapse_prevention', 0.0) +
        2.0 * losses_t.get('balance', 0.0) +
        5.0 * losses_t.get('temporal_consistency', 0.0)
    )


def compute_hinge_loss(z_preds):
    """
    Compute hinge loss to prevent latent variable shrinkage.
    
    Args:
        z_preds: Predicted latent variables
    
    Returns:
        Hinge loss value
    """
    # Hinge loss to prevent latent variable shrinkage (force them to have some minimum magnitude)
    # Calculates norm for each [Batch, K] and applies hinge at 0.1
    return torch.mean(torch.relu(0.1 - torch.norm(z_preds, dim=-1)))


def compute_regularization_losses(z_preds, dt):
    """
    Compute L2 and LVR (latent velocity regularization) losses.
    
    Args:
        z_preds: Predicted latent variables
        dt: Time step
    
    Returns:
        Tuple of (L2 loss, LVR loss)
    """
    loss_l2 = torch.mean(z_preds**2)
    z_vel = (z_preds[1:] - z_preds[:-1]) / dt
    loss_lvr = torch.mean((z_vel[1:] - z_vel[:-1])**2) if len(z_vel) > 1 else torch.tensor(0.0, device=z_preds.device)
    loss_lvr += 0.1 * torch.mean(z_vel**2) if len(z_vel) > 0 else torch.tensor(0.0, device=z_preds.device)
    return loss_l2, loss_lvr


def compute_temporal_consistency_loss(criterion, s_t, mu_t, z_t_target, s_prev, mu_prev, z_enc_prev):
    """
    Compute temporal consistency loss.
    
    Args:
        criterion: Loss function to use
        s_t, mu_t, z_t_target: Current time step values
        s_prev, mu_prev, z_enc_prev: Previous time step values
    
    Returns:
        Temporal consistency loss value
    """
    return criterion(s_t, s_prev) + 10.0 * criterion(mu_t, mu_prev) + 0.5 * criterion(z_t_target, z_enc_prev)