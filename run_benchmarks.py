import torch
import numpy as np
import sympy as sp
import pandas as pd
import os
import multiprocessing
from test_single_mode import train_discovery, PhysicsSim

def calculate_dfs(discovered_expr, mode):
    r = sp.Symbol('r')
    if mode == 'spring':
        target = -10.0 * (r - 1.0)
    else:  # lj
        target = 48.0 * (r**-13) - 24.0 * (r**-7)
    
    try:
        diff = sp.simplify(discovered_expr - target)
        is_equivalent = (diff == 0)
        # If not exactly zero, maybe it's numerically very close
        if not is_equivalent:
            # Check if it's a small constant or something
            if diff.is_constant() and abs(float(sp.N(diff))) < 1e-4:
                is_equivalent = True
    except:
        is_equivalent = False
        
    return is_equivalent

def run_trial(mode, noise_std, trial_idx):
    seed = 42 + trial_idx
    print(f"\n>>> Starting Trial {trial_idx+1}/10 for {mode} (noise={noise_std}, seed={seed})")
    try:
        status, nn_loss, p_coeff, expr, mse = train_discovery(mode, noise_std=noise_std, seed=seed)
        is_equivalent = calculate_dfs(expr, mode)
        return {
            "Trial": trial_idx + 1,
            "Mode": mode,
            "Noise Level": noise_std,
            "Final Formula": str(expr),
            "MSE": mse,
            "Mathematical Equivalence": is_equivalent
        }
    except Exception as e:
        print(f"Error in trial {trial_idx+1}: {e}")
        return {
            "Trial": trial_idx + 1,
            "Mode": mode,
            "Noise Level": noise_std,
            "Final Formula": "ERROR",
            "MSE": 1e6,
            "Mathematical Equivalence": False
        }

def main():
    # Ensure multiprocessing is set up correctly
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    noise_level = 0.01
    all_results = []

    # Run trials for spring
    for i in range(10):
        res = run_trial('spring', noise_level, i)
        all_results.append(res)
        # Save intermediate results
        pd.DataFrame(all_results).to_csv('benchmark_results.csv', index=False)

    # Run trials for lj
    for i in range(10):
        res = run_trial('lj', noise_level, i)
        all_results.append(res)
        # Save intermediate results
        pd.DataFrame(all_results).to_csv('benchmark_results.csv', index=False)

    print("\n--- Benchmark Summary ---")
    df = pd.DataFrame(all_results)
    print(df.to_string())
    
    # Calculate success rates
    for mode in ['spring', 'lj']:
        mode_df = df[df['Mode'] == mode]
        success_rate = mode_df['Mathematical Equivalence'].mean() * 100
        print(f"{mode} Success Rate (Exact Equivalence): {success_rate:.1f}%")

if __name__ == "__main__":
    main()
