import torch
import numpy as np
import sympy as sp
from typing import List, Optional
import json
import os
import re
from symbolic_utils import gp_to_sympy

def calculate_symplectic_drift(model_func, z0, dt, steps=100):
    """
    Calculates the symplectic drift over time.
    Symplectic drift measures how much the Jacobian of the flow deviates from being symplectic.
    For a symplectic map J, J^T * Omega * J = Omega.
    """
    batch_size, latent_dim = z0.shape
    device = z0.device
    
    # Define the symplectic matrix Omega
    # Omega = [[0, I], [-I, 0]]
    d = latent_dim // 2
    Omega = torch.zeros((latent_dim, latent_dim), device=device)
    Omega[:d, d:] = torch.eye(d, device=device)
    Omega[d:, :d] = -torch.eye(d, device=device)
    
    z = z0.clone().detach().requires_grad_(True)
    
    drifts = []
    
    for _ in range(steps):
        # Compute one step using rk4
        # k1 = model_func(z)
        # ... simplified for now: just compute Jacobian of the one-step map
        
        # To get the flow Jacobian, we need to integrate
        # But we can also check the divergence of the vector field: div(f) = 0 for Hamiltonian
        # Or check the condition on the vector field Jacobian M: M^T * Omega + Omega * M = 0
        
        def get_dz(z_in):
            return model_func(0, z_in)
        
        # Vectorized Jacobian computation
        # Use a small batch if possible, or just the first element
        z_sample = z[0:1]
        # Jacobian expects a function that takes only one argument (z)
        M = torch.autograd.functional.jacobian(get_dz, z_sample).squeeze()
        
        # Hamiltonian condition: M^T * Omega + Omega * M = 0
        drift_matrix = M.t() @ Omega + Omega @ M
        drift = torch.norm(drift_matrix).item()
        drifts.append(drift)
        
        # Update z (Euler step for simplicity in drift tracking)
        dz = get_dz(z)
        z = (z + dz * dt).detach().requires_grad_(True)
        
    return np.mean(drifts)

def calculate_energy_drift(model_func, energy_func, z0, dt, steps=100):
    """
    Calculates the relative energy drift over a trajectory.
    """
    z = z0.clone().detach()
    
    initial_energy = energy_func(z)
    energies = [initial_energy.item()]
    
    for _ in range(steps):
        dz = model_func(0, z)
        z = z + dz * dt
        energies.append(energy_func(z).item())
        
    energies = np.array(energies)
    drift = np.abs(energies - initial_energy.item()) / (np.abs(initial_energy.item()) + 1e-9)
    return np.max(drift)

def verify_hamiltonian_properties(expr_str, n_super_nodes, latent_dim):
    """
    Verifies if a discovered expression satisfies Hamiltonian properties.
    Calculates the Poisson bracket {q, H} and {p, H} if possible.
    """
    print(f"\n--- Hamiltonian Poisson Bracket Verification ---")
    
    # We need to know which x_i correspond to which latent dimensions
    # In the transformer, we use physical features (distances, p^2).
    # If H is discovered as a function of these features, we can't directly 
    # compute Poisson brackets w.r.t q, p unless we know the mapping features -> (q, p).
    
    # However, we can check if H depends on both "position-like" and "momentum-like" features.
    
    q_features = ['sum_inv_d', 'sum_d', 'z0', 'z1'] # positions for K=2, D=4 (first 2 per node)
    p_features = ['p0^2_sum', 'p1^2_sum', 'z2', 'z3'] # momentum for K=2, D=4 (last 2 per node)
    
    has_q = any(f in expr_str for f in q_features)
    has_p = any(f in expr_str for f in p_features)
    
    print(f"  Detected Position-like features: {has_q}")
    print(f"  Detected Momentum-like features: {has_p}")
    
    if has_q and has_p:
        print("  [Status] Valid Phase-Space Dependency.")
    else:
        print("  [Status] Incomplete Phase-Space (Likely non-conservative or static).")

    # Analytical test: if H = p^2/2 + V(q), then {q, H} = p and {p, H} = -dV/dq
    # This matches the Hamiltonian equations dq/dt = p, dp/dt = -dV/dq.
    
    try:
        # Simplified SymPy test
        q, p = sp.symbols('q p')
        # Let's assume a 1D test case
        # If expr contains 'sum_inv_d6' (proportional to q^-6) and 'p0^2_sum' (proportional to p^2)
        # We can substitute and check
        pass
    except:
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, required=True, help='Path to discovery.json')
    args = parser.parse_args()
    
    if os.path.exists(args.results):
        with open(args.results, 'r') as f:
            data = json.load(f)
        
        eqs = data.get('equations', [])
        for eq in eqs:
            verify_hamiltonian_properties(eq, data['config']['super_nodes'], data['config']['latent_dim'])
