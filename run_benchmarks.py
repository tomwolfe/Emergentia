import torch
import pandas as pd
import multiprocessing
import time
from emergentia import PhysicsSim, DiscoveryPipeline
from emergentia.simulator import HarmonicPotential, LennardJonesPotential, MorsePotential

def run_trial(mode, potential, noise_std, trial_idx):
    seed = 42 + trial_idx
    print(f"\n>>> Trial {trial_idx+1} | Mode: {mode} | Noise: {noise_std} | Seed: {seed}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # print(f"Using device: {device}")
    
    sim = PhysicsSim(n=2, potential=potential, seed=seed, device=device)
    pipeline = DiscoveryPipeline(mode=mode, potential=potential, device=device, seed=seed)
    
    start_time = time.perf_counter()
    try:
        # Reduce epochs for faster benchmarking
        result = pipeline.run(sim, nn_epochs=3000, noise_std=noise_std)
        duration = time.perf_counter() - start_time
        result['trial'] = trial_idx + 1
        result['noise_std'] = noise_std
        result['duration'] = duration
        return result
    except Exception as e:
        print(f"Error in trial {trial_idx+1}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "mode": mode, "noise_std": noise_std, "nn_loss": 1.0, 
            "formula": "ERROR", "mse": 1e6, "r2": 0.0, "bic": 1e6,
            "success": False, "trial": trial_idx + 1, "duration": 0
        }

def main():
    all_results = []
    
    potentials = {
        'spring': HarmonicPotential(),
        'lj': LennardJonesPotential(),
        'morse': MorsePotential()
    }
    
    noise_levels = [0.0, 0.01, 0.05]
    num_trials = 1 # Keep small for initial run
    
    for mode, potential in potentials.items():
        for noise in noise_levels:
            for i in range(num_trials):
                res = run_trial(mode, potential, noise, i)
                all_results.append(res)
                
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*80)
    print("      EMERGENTIA ENHANCED BENCHMARK RESULTS")
    print("="*80)
    print(df[['mode', 'noise_std', 'success', 'r2', 'mse', 'duration', 'formula']].to_string(index=False))
    print("="*80)
    
    summary = df.groupby(['mode', 'noise_std'])['success'].mean() * 100
    print("\nSummary Success Rates (%):")
    print(summary)

if __name__ == "__main__":
    main()