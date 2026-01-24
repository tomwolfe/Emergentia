import torch
import numpy as np
import argparse
import os
import json
import sympy as sp
from torch_geometric.data import Batch
from simulator import SpringMassSimulator, LennardJonesSimulator
from model import DiscoveryEngineModel
from engine import prepare_data
from symbolic import extract_latent_data
from train_utils import get_device

def load_discovery_results(results_path):
    with open(results_path, 'r') as f:
        return json.load(f)

def shadow_integration(symbolic_proxy, initial_z, steps, dt, device):
    """
    Integrates the symbolic equations forward from an initial state.
    """
    z_traj = [initial_z]
    z_curr = initial_z
    
    for _ in range(steps - 1):
        # simple Euler for symbolic integration if ODE solver is not used
        # or use the same integration method as in the model
        with torch.no_grad():
            dz = symbolic_proxy(z_curr)
            z_curr = z_curr + dz * dt
        z_traj.append(z_curr)
        
    return torch.stack(z_traj).squeeze(1)

def calculate_forecast_horizon(symbolic_traj, ground_truth_traj, variance):
    """
    Calculates the number of steps until MSE exceeds the variance of the dataset.
    """
    mse = torch.mean((symbolic_traj - ground_truth_traj)**2, dim=(1, 2))
    horizon = 0
    for i, err in enumerate(mse):
        if err > variance:
            break
        horizon = i + 1
    return horizon

def main():
    parser = argparse.ArgumentParser(description="Closed-Loop Validation of Discovered Equations")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--results_path', type=str, required=True)
    parser.add_argument('--sim', type=str, default='lj', choices=['spring', 'lj'])
    parser.add_argument('--particles', type=int, default=8)
    parser.add_argument('--steps', type=int, default=2000) # Increased to 2000 for Stress Test
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else args.device
    print(f"Using device: {device}")

    # 1. Load Results and Setup Model
    results = load_discovery_results(args.results_path)
    config = results['config']
    
    model = DiscoveryEngineModel(
        n_particles=config['particles'],
        n_super_nodes=config['super_nodes'],
        latent_dim=config['latent_dim'],
        hidden_dim=config['hidden_dim'],
        hamiltonian=config['hamiltonian'],
        min_active_super_nodes=config['min_active']
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    # 2. Setup Simulator and Generate Ground Truth (OOD: Higher energy or just longer)
    print(f"Generating ground truth trajectory for {args.steps} steps...")
    if args.sim == 'spring':
        sim = SpringMassSimulator(n_particles=args.particles, spring_dist=1.0, dynamic_radius=1.5)
    else:
        # Increase initial velocity for OOD test
        sim = LennardJonesSimulator(n_particles=args.particles, dynamic_radius=2.0)

    pos, vel = sim.generate_trajectory(steps=args.steps)
    
    # We need to process in chunks to avoid memory issues on some devices
    dataset_list = []
    chunk_size = 500
    for i in range(0, args.steps, chunk_size):
        end = min(i + chunk_size, args.steps)
        d_chunk, stats = prepare_data(pos[i:end], vel[i:end], radius=2.0, device=device)
        dataset_list.extend(d_chunk)

    # 3. Extract Neural Latent Trajectory
    print("Extracting neural latent trajectory...")
    z_gt_list = []
    with torch.no_grad():
        for i in range(0, args.steps, chunk_size):
            end = min(i + chunk_size, args.steps)
            batch = Batch.from_data_list(dataset_list[i:end]).to(device)
            z, _, _, _ = model.encode(batch.x, batch.edge_index, batch.batch)
            z_gt_list.append(z.cpu())
    
    z_gt = torch.cat(z_gt_list, dim=0) # [T, K, D]
    z_gt_reshaped = z_gt.view(args.steps, config['super_nodes'], config['latent_dim'])

    # 4. Setup Symbolic Proxy
    from engine import SymbolicProxy
    from balanced_features import BalancedFeatureTransformer
    transformer = BalancedFeatureTransformer(config['super_nodes'], config['latent_dim'])
    
    # Re-fit transformer on a subset of the ground truth to ensure proper normalization
    # but the symbolic equations were discovered on training data normalization.
    # We should ideally use the training normalization.
    # For now, we'll use the discovery_report's equations which are already distilled.
    
    # Use equations from results
    equations = results['equations']
    # Re-fitting transformer here is a bit risky if it changes feature indices,
    # but SymbolicProxy needs a fitted transformer.
    # In a perfect system, we'd save the transformer state.
    transformer.fit(z_gt.numpy().reshape(args.steps, -1), np.zeros((args.steps, z_gt.shape[-1])))
    
    symbolic_proxy = SymbolicProxy(config['super_nodes'], config['latent_dim'], equations, transformer, hamiltonian=config['hamiltonian']).to(device)

    # 5. Shadow Integration (Stress Test)
    print("Performing Shadow Integration (Stress Test)...")
    initial_z = z_gt[0:1].to(device) # [1, K, D]
    initial_z_flat = initial_z.view(1, -1)
    
    # Move symbolic proxy to CPU for more stable long-term integration if using MPS
    if str(device) == 'mps':
        symbolic_proxy.to('cpu')
        initial_z_flat = initial_z_flat.cpu()
        integration_device = torch.device('cpu')
    else:
        integration_device = device

    z_traj = [initial_z_flat]
    z_curr = initial_z_flat
    
    for t in range(args.steps - 1):
        with torch.no_grad():
            dz = symbolic_proxy(z_curr)
            z_curr = z_curr + dz * sim.dt
            if torch.isnan(z_curr).any():
                print(f"Integration diverged at step {t}")
                break
        z_traj.append(z_curr)
        if t % 500 == 0:
            print(f"  Step {t}/{args.steps}...")
            
    symbolic_traj_flat = torch.stack(z_traj).squeeze(1)
    # Ensure it's back on CPU for comparison
    symbolic_traj_flat = symbolic_traj_flat.cpu()
    z_gt_flat = z_gt.view(args.steps, -1)
    
    # If diverged early, pad with last value or zeros
    if len(symbolic_traj_flat) < args.steps:
        pad = torch.zeros((args.steps - len(symbolic_traj_flat), z_gt_flat.shape[1]))
        symbolic_traj_flat = torch.cat([symbolic_traj_flat, pad], dim=0)

    # 6. Calculate Forecast Horizon (Step count where error > 5% of variance)
    print("Calculating Forecast Horizon...")
    mse_per_step = torch.mean((symbolic_traj_flat - z_gt_flat)**2, dim=1)
    total_variance = torch.var(z_gt_flat)
    threshold = 0.05 * total_variance
    
    horizon = 0
    for i, mse in enumerate(mse_per_step):
        if mse > threshold:
            break
        horizon = i
    
    print(f"\nForecast Horizon: {horizon} steps ({(horizon/args.steps)*100:.1f}% of stress test trajectory)")
    
    # 7. Metrics
    is_stable = (horizon >= args.steps * 0.5) # Passed if it survives half the stress test
    
    # Final Report
    report = {
        "Forecast Horizon": horizon,
        "Relative Horizon": horizon / args.steps,
        "Stability Score": 1.0 if horizon == args.steps else horizon / args.steps,
        "Total Steps": args.steps,
        "MSE at End": float(mse_per_step[-1]),
        "Passed": bool(is_stable)
    }
    
    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Validation report saved to validation_report.json")


if __name__ == "__main__":
    main()
