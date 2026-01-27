import torch
import pandas as pd
import multiprocessing
from emergentia import PhysicsSim, DiscoveryPipeline

def run_trial(mode, trial_idx):
    seed = 42 + trial_idx
    print(f"\n>>> Trial {trial_idx+1} | Mode: {mode} | Seed: {seed}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sim = PhysicsSim(n=2, mode=mode, seed=seed, device=device)
    pipeline = DiscoveryPipeline(mode=mode, device=device, seed=seed)
    try:
        result = pipeline.run(sim)
        result['trial'] = trial_idx + 1
        return result
    except Exception as e:
        print(f"Error in trial {trial_idx+1}: {e}")
        return {"mode": mode, "nn_loss": 1.0, "formula": "ERROR", "mse": 1e6, "success": False, "trial": trial_idx + 1}

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
    print(df[['mode', 'trial', 'nn_loss', 'mse', 'success', 'formula']].to_string(index=False))
    print("="*50)
    for m in modes:
        sr = df[df['mode'] == m]['success'].mean() * 100
        print(f"{m.upper()} Success Rate: {sr:.1f}%")

if __name__ == "__main__":
    main()
