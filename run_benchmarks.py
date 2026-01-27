import torch
import pandas as pd
import multiprocessing
import time
from emergentia import PhysicsSim, DiscoveryPipeline

def run_trial(mode, trial_idx):
    seed = 42 + trial_idx
    print(f"\n>>> Trial {trial_idx+1} | Mode: {mode} | Seed: {seed}")
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    sim = PhysicsSim(n=2, mode=mode, seed=seed, device=device)
    pipeline = DiscoveryPipeline(mode=mode, device=device, seed=seed)
    
    start_time = time.perf_counter()
    try:
        result = pipeline.run(sim)
        duration = time.perf_counter() - start_time
        result['trial'] = trial_idx + 1
        result['duration'] = duration
        result['speedup'] = 300.0 / duration # Baseline 300s
        return result
    except Exception as e:
        print(f"Error in trial {trial_idx+1}: {e}")
        return {"mode": mode, "nn_loss": 1.0, "formula": "ERROR", "mse": 1e6, "success": False, "trial": trial_idx + 1, "duration": 0, "speedup": 0}

def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    all_results = []
    modes = ['spring', 'lj']
    num_trials = 1
    for mode in modes:
        for i in range(num_trials):
            res = run_trial(mode, i)
            all_results.append(res)
    df = pd.DataFrame(all_results)
    print("\n" + "="*50)
    print("      EMERGENTIA BENCHMARK RESULTS")
    print("="*50)
    print(df[['mode', 'trial', 'nn_loss', 'mse', 'success', 'duration', 'speedup', 'formula']].to_string(index=False))
    print("="*50)
    for m in modes:
        sr = df[df['mode'] == m]['success'].mean() * 100
        avg_speedup = df[df['mode'] == m]['speedup'].mean()
        print(f"{m.upper()} Success Rate: {sr:.1f}% | Avg Speedup: {avg_speedup:.2f}x")

if __name__ == "__main__":
    main()
