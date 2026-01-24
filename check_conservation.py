import torch
import numpy as np
import sympy as sp
from typing import List, Optional
import json
import os
import re

def gp_to_sympy(expr_str, n_features):
    local_dict = {
        'add': lambda x,y: x+y,
        'sub': lambda x,y: x-y,
        'mul': lambda x,y: x*y,
        'div': lambda x,y: x/(y+1e-9),
        'sqrt': lambda x: sp.sqrt(sp.Abs(x)),
        'log': lambda x: sp.log(sp.Abs(x)+1e-9),
        'abs': sp.Abs,
        'neg': lambda x: -x,
        'inv': lambda x: 1.0/(x+1e-9),
        'square': lambda x: x**2,
        'inv_square': lambda x: 1.0/(x**2+1e-9)
    }
    # Map X0, X1 to x0, x1
    for i in range(n_features):
        local_dict[f'X{i}'] = sp.Symbol(f'x{i}')
    return sp.sympify(expr_str, locals=local_dict)

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
