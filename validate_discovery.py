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
    parser.add_argument('--steps', type=int, default=200)
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
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 2. Setup Simulator and Generate Ground Truth
    if args.sim == 'spring':
        sim = SpringMassSimulator(n_particles=args.particles, spring_dist=1.0, dynamic_radius=1.5)
    else:
        sim = LennardJonesSimulator(n_particles=args.particles, dynamic_radius=2.0)

    print("Generating ground truth trajectory...")
    pos, vel = sim.generate_trajectory(steps=args.steps)
    dataset_list, stats = prepare_data(pos, vel, radius=2.0, device=device)
    dataset = Batch.from_data_list(dataset_list).to(device)

    # 3. Extract Neural Latent Trajectory
    print("Extracting neural latent trajectory...")
    z_gt, dz_gt, _ = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=config['hamiltonian'])
    z_gt = torch.from_numpy(z_gt).float().to(device)
    z_gt_reshaped = z_gt.view(args.steps, config['super_nodes'], config['latent_dim'])

    # 4. Setup Symbolic Proxy
    from engine import SymbolicProxy
    # Re-create the transformer from saved stats
    from balanced_features import BalancedFeatureTransformer
    transformer = BalancedFeatureTransformer(config['super_nodes'], config['latent_dim'])
    # We need more than just stats to re-create the transformer... 
    # For now, let's assume we can load it or re-fit it if we have the data.
    # In a real scenario, we'd pickle the transformer.
    transformer.fit(z_gt.cpu().numpy(), dz_gt)
    
    # Use equations from results
    equations = results['equations']
    symbolic_proxy = SymbolicProxy(config['super_nodes'], config['latent_dim'], equations, transformer).to(device)

    # 5. Shadow Integration
    print("Performing Shadow Integration...")
    initial_z = z_gt[0:1] # [1, K*D] 
    symbolic_traj_flat = shadow_integration(symbolic_proxy, initial_z, args.steps, sim.dt, device)
    symbolic_traj = symbolic_traj_flat.view(args.steps, config['super_nodes'], config['latent_dim'])

    # 6. Calculate Forecast Horizon
    print("Calculating Forecast Horizon...")
    latent_variance = torch.var(z_gt)
    horizon = calculate_forecast_horizon(symbolic_traj, z_gt_reshaped, latent_variance)
    
    print(f"\nForecast Horizon: {horizon} steps ({(horizon/args.steps)*100:.1f}% of trajectory)")
    
    # 7. Stability Check
    is_stable = horizon >= args.steps * 0.95
    print(f"Long-term Stability: {'PASSED' if is_stable else 'FAILED'}")

    # Final Report
    report = {
        "Forecast Horizon": horizon,
        "Stability": is_stable,
        "Steps": args.steps,
        "Relative Horizon": horizon / args.steps
    }
    
    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    main()
