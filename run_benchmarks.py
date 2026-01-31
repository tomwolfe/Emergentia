import torch
import pandas as pd
import multiprocessing
import time
import os
from emergentia import PhysicsSim, DiscoveryPipeline
from emergentia.simulator import HarmonicPotential, LennardJonesPotential, MorsePotential, GravityPotential, CompositePotential, BuckinghamPotential, YukawaPotential

def run_trial(mode, potential, noise_std, trial_idx, dim=2):
    seed = 42 + trial_idx
    print(f"\n>>> Trial {trial_idx+1} | Mode: {mode} | Dim: {dim} | Noise: {noise_std} | Seed: {seed}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Select basis set based on mode
    basis_set = None
    if mode == 'gravity':
        basis_set = ['1', '1/r^2']
    elif mode == 'lj':
        basis_set = ['1', '1/r^7', '1/r^13']
    elif mode == 'spring':
        basis_set = ['1', 'r']
    elif mode == 'morse':
        basis_set = ['1', 'exp(-r)']
    elif mode == 'buckingham':
        basis_set = ['1', '1/r^7', 'exp(-r)']
    elif mode == 'yukawa':
        basis_set = ['1/r', '1/r^2', 'exp(-r)/r']
    elif mode == 'mixed':
        basis_set = ['1', 'r', '1/r^2']
        
    sim = PhysicsSim(n=3, dim=dim, potential=potential, seed=seed, device=device)
    pipeline = DiscoveryPipeline(mode=mode, potential=potential, device=device, seed=seed, basis_set=basis_set)
    
    start_time = time.perf_counter()
    try:
        # Balanced epochs for speed and accuracy
        result = pipeline.run(sim, nn_epochs=2000, noise_std=noise_std)
        duration = time.perf_counter() - start_time
        result['trial'] = trial_idx + 1
        result['noise_std'] = noise_std
        result['duration'] = duration
        result['dim'] = dim
        
        # Save report
        os.makedirs('results', exist_ok=True)
        report_path = f"results/report_{mode}_dim{dim}_noise{noise_std}_trial{trial_idx}.txt"
        with open(report_path, 'w') as f:
            f.write(f"Emergentia Discovery Report\n")
            f.write(f"===========================\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Dimension: {dim}\n")
            f.write(f"Noise Std: {noise_std}\n")
            f.write(f"Raw Formula: {result.get('raw_formula')}\n")
            f.write(f"Refined Formula: {result.get('formula')}\n")
            f.write(f"Success: {result.get('success')}\n")
            f.write(f"MSE: {result.get('mse'):.2e}\n")
            f.write(f"R2: {result.get('r2'):.4f}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            
        return result
    except Exception as e:
        print(f"Error in trial {trial_idx+1}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "mode": mode, "noise_std": noise_std, "nn_loss": 1.0, 
            "formula": "ERROR", "mse": 1e6, "r2": 0.0, "bic": 1e6,
            "success": False, "trial": trial_idx + 1, "duration": 0, "dim": dim
        }

def main():
    all_results = []
    
    potentials = {
        'gravity': GravityPotential(),
        'lj': LennardJonesPotential(),
        'morse': MorsePotential(),
        'buckingham': BuckinghamPotential(),
        'yukawa': YukawaPotential(),
        'mixed': CompositePotential([HarmonicPotential(k=10.0, r0=1.0), GravityPotential(G=1.0)])
    }
    
    noise_levels = [0.0, 0.01, 0.05]
    num_trials = 3
    dimensions = [3]
    
    for dim in dimensions:
        for mode, potential in potentials.items():
            for noise in noise_levels:
                for i in range(num_trials):
                    res = run_trial(mode, potential, noise, i, dim=dim)
                    all_results.append(res)
                
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("      EMERGENTIA ENHANCED BENCHMARK RESULTS")
    print("="*80)
    print(df[['mode', 'dim', 'noise_std', 'success', 'r2', 'mse', 'formula']].to_string(index=False))
    print("="*80)
    
    summary = df.groupby(['mode', 'noise_std'])['success'].mean() * 100
    print("\nSummary Success Rates (%):")
    print(summary)

    # Export benchmark summary to CSV
    summary_df = df.groupby(['mode', 'noise_std']).agg({
        'success': lambda x: (x.sum() / len(x)) * 100,  # Success rate percentage
        'r2': 'mean',
        'mse': 'mean'
    }).round(4)
    summary_df.columns = ['success_rate_pct', 'mean_r2', 'mean_mse']
    summary_df.reset_index(inplace=True)

    os.makedirs('results', exist_ok=True)
    summary_df.to_csv('results/benchmark_summary.csv', index=False)
    print(f"\nBenchmark summary saved to results/benchmark_summary.csv")
    print("\nDetailed Summary:")
    print(summary_df)

if __name__ == "__main__":
    main()