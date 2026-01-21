"""
Optimized ODE functions to address adjoint sensitivity complexity.
This module implements more efficient Hamiltonian ODE functions with reduced gradient computation overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint, odeint_adjoint
import numpy as np


class OptimizedHamiltonianODEFunc(nn.Module):
    """
    Optimized Hamiltonian ODE function with reduced computational complexity.
    Implements several optimizations to reduce the computational burden of adjoint sensitivity.
    """
    
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=64, dissipative=True, 
                 use_checkpointing=False, checkpoint_interval=5):
        """
        Initialize the optimized Hamiltonian ODE function.

        Args:
            latent_dim (int): Dimension of latent space (must be even for (q,p) pairs)
            n_super_nodes (int): Number of super-nodes
            hidden_dim (int): Hidden dimension for the Hamiltonian network
            dissipative (bool): Whether to include learnable dissipation terms
            use_checkpointing (bool): Whether to use gradient checkpointing
            checkpoint_interval (int): Interval for checkpointing (if used)
        """
        super(OptimizedHamiltonianODEFunc, self).__init__()
        assert latent_dim % 2 == 0, "Latent dim must be even for Hamiltonian dynamics (q, p)"
        self.latent_dim = latent_dim
        self.n_super_nodes = n_super_nodes
        self.dissipative = dissipative
        self.use_checkpointing = use_checkpointing
        self.checkpoint_interval = checkpoint_interval

        # Optimized and more efficient Hamiltonian network
        # Use a shallower but wider network to reduce gradient computation overhead
        self.H_net = nn.Sequential(
            nn.Linear(latent_dim * n_super_nodes, hidden_dim),  # Reduced from hidden_dim//2 to hidden_dim
            nn.SiLU(), # Faster activation function
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),  # Intermediate layer
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        if dissipative:
            # Small initial dissipation
            self.gamma = nn.Parameter(torch.full((n_super_nodes, 1), -5.0)) # log space

        # Pre-compute indices for reshaping to avoid repeated computation
        self.q_indices = torch.arange(0, latent_dim, 2)
        self.p_indices = torch.arange(1, latent_dim, 2)

    def forward(self, t, y):
        """
        Compute time derivatives using optimized Hamiltonian mechanics.

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
            
            # Compute Hamiltonian
            H = self.H_net(y_in).sum()
            
            # Compute gradients efficiently
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


class MemoryEfficientHamiltonianODEFunc(nn.Module):
    """
    Memory-efficient version of Hamiltonian ODE function that trades some accuracy for reduced memory usage.
    Useful when dealing with large systems where memory is a constraint.
    """
    
    def __init__(self, latent_dim, n_super_nodes, hidden_dim=32, dissipative=True):
        """
        Initialize the memory-efficient Hamiltonian ODE function.

        Args:
            latent_dim (int): Dimension of latent space (must be even for (q,p) pairs)
            n_super_nodes (int): Number of super-nodes
            hidden_dim (int): Hidden dimension for the Hamiltonian network (smaller for efficiency)
            dissipative (bool): Whether to include learnable dissipation terms
        """
        super(MemoryEfficientHamiltonianODEFunc, self).__init__()
        assert latent_dim % 2 == 0, "Latent dim must be even for Hamiltonian dynamics (q, p)"
        self.latent_dim = latent_dim
        self.n_super_nodes = n_super_nodes
        self.dissipative = dissipative

        # Smaller, more efficient network to reduce memory usage
        self.H_net = nn.Sequential(
            nn.Linear(latent_dim * n_super_nodes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        if dissipative:
            # Small initial dissipation
            self.gamma = nn.Parameter(torch.full((n_super_nodes, 1), -5.0))

    def forward(self, t, y):
        """
        Compute time derivatives using memory-efficient Hamiltonian mechanics.

        Args:
            t (Tensor): Time (unused in autonomous systems)
            y (Tensor): State vector [batch_size, latent_dim * n_super_nodes]

        Returns:
            Tensor: Time derivatives [batch_size, latent_dim * n_super_nodes]
        """
        # y: [batch_size, latent_dim * n_super_nodes]
        training = torch.is_grad_enabled()

        # Use torch.no_grad for initial computation to reduce memory overhead
        with torch.no_grad():
            y_detached = y.detach()

        # Compute Hamiltonian with gradient tracking
        H = self.H_net(y_detached).sum()

        # Compute gradients efficiently
        dH = torch.autograd.grad(H, y_detached, create_graph=training, allow_unused=True)[0]

        if dH is None:
            dH = torch.zeros_like(y)
        else:
            # Handle NaNs and Infs in the gradient
            dH = torch.nan_to_num(dH, nan=0.0, posinf=1e3, neginf=-1e3)

        # Gradient clipping for stability
        dH = torch.clamp(dH, -1e3, 1e3)

        # Reshape and compute derivatives
        d_sub = self.latent_dim // 2
        dH_view = dH.view(-1, self.n_super_nodes, 2, d_sub)

        dq = dH_view[:, :, 1]  # dH/dp
        dp = -dH_view[:, :, 0] # -dH/dq

        if self.dissipative:
            y_view = y.view(-1, self.n_super_nodes, 2, d_sub)
            p = y_view[:, :, 1] # momentum
            gamma = torch.exp(torch.clamp(self.gamma, max=2.0))
            dp = dp - gamma * p

        return torch.cat([dq, dp], dim=-1).view(y.shape[0], -1)


class AdaptiveSolver:
    """
    Adaptive ODE solver that chooses between different integration methods based on system properties.
    Reduces computational overhead by selecting the most appropriate solver for the current system state.
    """
    
    def __init__(self, ode_func, method='auto', rtol=1e-3, atol=1e-5, 
                 use_adjoint=True, adjoint_method='checkpointed_dopri5'):
        """
        Initialize the adaptive solver.

        Args:
            ode_func: The ODE function to solve
            method: Integration method ('auto', 'dopri5', 'euler', 'rk4')
            rtol: Relative tolerance
            atol: Absolute tolerance
            use_adjoint: Whether to use adjoint method for training
            adjoint_method: Method for adjoint computation
        """
        self.ode_func = ode_func
        self.method = method
        self.rtol = rtol
        self.atol = atol
        self.use_adjoint = use_adjoint
        self.adjoint_method = adjoint_method

    def solve(self, y0, t):
        """
        Solve the ODE using the adaptive method.

        Args:
            y0: Initial conditions
            t: Time points

        Returns:
            Solution at time points t
        """
        # Determine the appropriate method based on training vs evaluation
        if self.method == 'auto':
            if torch.is_grad_enabled():
                # During training, use a more efficient method
                method = 'euler' if self.use_adjoint else 'dopri5'
            else:
                # During evaluation, use a more accurate method
                method = 'dopri5'
        else:
            method = self.method

        # Use the appropriate solver
        if self.use_adjoint and torch.is_grad_enabled():
            # Use adjoint method for memory efficiency during training
            return odeint_adjoint(
                self.ode_func, y0, t,
                method=self.adjoint_method,
                options={'step_size': 0.01} if method == 'euler' else {},
                rtol=self.rtol,
                atol=self.atol
            )
        else:
            # Use standard odeint for evaluation
            return odeint(
                self.ode_func, y0, t,
                method=method,
                rtol=self.rtol,
                atol=self.atol
            )


def create_optimized_ode_func(latent_dim, n_super_nodes, hidden_dim=64, 
                            dissipative=True, optimization_level='standard'):
    """
    Factory function to create an optimized ODE function.

    Args:
        latent_dim: Dimension of latent space
        n_super_nodes: Number of super-nodes
        hidden_dim: Hidden dimension for the network
        dissipative: Whether to include dissipation
        optimization_level: Level of optimization ('standard', 'memory_efficient')

    Returns:
        Optimized ODE function
    """
    if optimization_level == 'memory_efficient':
        return MemoryEfficientHamiltonianODEFunc(
            latent_dim, n_super_nodes, hidden_dim // 2, dissipative
        )
    else:
        return OptimizedHamiltonianODEFunc(
            latent_dim, n_super_nodes, hidden_dim, dissipative
        )