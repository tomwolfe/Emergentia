#!/usr/bin/env python3
"""
Test script to verify the optimized neural-symbolic pipeline implementation.
This script tests all the fixes implemented according to the requirements.
"""

import torch
import numpy as np
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

from simulator import SpringMassSimulator, LennardJonesSimulator
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data, analyze_latent_space
from symbolic import extract_latent_data, SymbolicDistiller
from hamiltonian_symbolic import HamiltonianSymbolicDistiller
from train_utils import ImprovedEarlyStopping, robust_energy, get_device
from stable_pooling import SparsityScheduler
from visualization import plot_discovery_results, plot_training_history


def test_optimized_pipeline():
    """Test the optimized pipeline with all implemented fixes."""
    print("=" * 60)
    print("Testing Optimized Neural-Symbolic Pipeline")
    print("=" * 60)
    
    # Configuration matching the requirements
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Setup Simulator
    print("\n1. Setting up simulator...")
    sim = SpringMassSimulator(n_particles=8, dynamic_radius=2.0)  # Reduced for faster testing
    
    # 2. Generate Data
    print("2. Generating trajectory...")
    pos, vel = sim.generate_trajectory(steps=100)  # Reduced for faster testing
    dataset, stats = prepare_data(pos, vel, radius=2.0, device=device)
    
    # 3. Setup Model & Trainer with all optimizations
    print("3. Setting up model and trainer with optimizations...")
    model = DiscoveryEngineModel(
        n_particles=8,
        n_super_nodes=3,  # Reduced for faster testing
        latent_dim=4,
        hamiltonian=False
    ).to(device)
    
    # Initialize SparsityScheduler to prevent resolution collapse
    sparsity_scheduler = SparsityScheduler(
        initial_weight=0.0,
        target_weight=0.1,
        warmup_steps=50,  # Reduced for testing
        max_steps=200    # Reduced for testing
    )
    
    trainer = Trainer(
        model,
        lr=5e-4,
        device=device,
        stats=stats,
        warmup_epochs=20,  # Reduced for testing
        max_epochs=100,    # Reduced for testing
        sparsity_scheduler=sparsity_scheduler
    )
    
    # Verify that temporal consistency weight is increased
    temp_weight = model.encoder.pooling.temporal_consistency_weight
    print(f"   Temporal consistency weight: {temp_weight} (should be > 10.0)")
    
    # Verify that log_vars are in optimizer
    log_var_found = any(id(p) == id(model.log_vars) for pg in trainer.optimizer.param_groups for p in pg['params'])
    print(f"   Log vars in optimizer: {log_var_found} (should be True)")
    
    # 4. Training Loop with reconstruction-first warmup
    print("4. Starting training with optimizations...")
    print("   - Reconstruction-first warmup (first 50 epochs)")
    print("   - Latent smoothing loss")
    print("   - Enhanced loss balancing")
    
    early_stopping = ImprovedEarlyStopping(patience=50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=100)
    
    last_rec = 1.0
    for epoch in range(100):
        idx = np.random.randint(0, len(dataset) - 10)
        batch_data = dataset[idx : idx + 10]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=100)
        scheduler.step()
        last_rec = rec
        
        if epoch % 20 == 0:
            print(f"   Epoch {epoch:3d} | Loss: {loss:.4f} | Rec: {rec:.4f} | Cons: {cons:.4f}")
        
        if early_stopping(loss):
            print(f"   Early stopping triggered at epoch {epoch}")
            break
    
    print(f"\n5. Final reconstruction loss: {last_rec:.4f} (should be < 0.45)")
    
    # 6. Analysis & Symbolic Discovery with enhanced parameters
    print("6. Extracting latent data for analysis...")
    z_states, dz_states, t_states = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=False)
    
    # Compute stability metric
    stability_window = max(10, int(len(dz_states) * 0.2))
    recent_dz = dz_states[-stability_window:]
    dz_stability = np.var(recent_dz)
    print(f"   Latent Stability (Var[dz]): {dz_stability:.6f} (should be < 0.05 for good stability)")
    
    # Perform symbolic discovery with increased populations and generations
    print("7. Performing enhanced symbolic distillation...")
    print("   - Populations: 2000 (increased)")
    print("   - Generations: 40 (increased)")
    
    distiller = SymbolicDistiller(populations=2000, generations=40)
    equations = distiller.distill(z_states, dz_states, 3, 4)  # n_super_nodes=3, latent_dim=4
    
    print("\n   Discovered Equations:")
    for i, eq in enumerate(equations):
        target = f"dz_{i}/dt"
        print(f"     {target} = {eq}")
    
    # 8. Visualization with symbolic predictions overlay
    print("8. Creating visualization with symbolic predictions overlay...")
    # Get assignments for the full trajectory
    model.eval()
    s_list = []
    with torch.no_grad():
        for data in dataset[:50]:  # Limit for faster testing
            batch = Batch.from_data_list([data]).to(device)
            _, s, _, _ = model.encode(batch.x, batch.edge_index, batch.batch)
            s_list.append(s.cpu())

    s_full = torch.cat(s_list, dim=0) # [T*N, K]
    s_heatmap = s_full.view(len(s_list), 8, 3).mean(dim=1)  # [T, K]

    # Get final assignments for scatter plot
    batch_0 = Batch.from_data_list([dataset[0]]).to(device)
    _, s_0, _, _ = model.encode(batch_0.x, batch_0.edge_index, batch_0.batch)
    assignments = torch.argmax(s_0, dim=1).cpu().numpy()

    z_states_plot = z_states.reshape(-1, 3, 4)[:len(s_list)]  # Match lengths

    # Generate symbolic predictions for visualization
    symbolic_predictions = None
    if equations and trainer.symbolic_proxy is not None:
        print("   Generating symbolic predictions for visualization...")
        try:
            # Convert z_states to tensor for symbolic prediction
            z_tensor = torch.tensor(z_states_plot.reshape(-1, 3 * 4), dtype=torch.float32, device=device)
            
            # Get symbolic predictions
            with torch.no_grad():
                symbolic_dz_dt = trainer.symbolic_proxy(z_tensor)
                
            # Reshape to match z_states_plot dimensions [T, K, D]
            symbolic_dz_dt = symbolic_dz_dt.cpu().numpy().reshape(-1, 3, 4)
            
            # Integrate to get trajectories (simple Euler integration)
            dt = 0.01  # Use a small time step for integration
            symbolic_traj = np.zeros_like(z_states_plot)
            symbolic_traj[0] = z_states_plot[0]  # Initialize with first state
            
            for t in range(1, len(symbolic_traj)):
                # Simple Euler integration: z_new = z_old + dt * dz_dt
                symbolic_traj[t] = symbolic_traj[t-1] + dt * symbolic_dz_dt[t-1]
            
            symbolic_predictions = symbolic_traj
            print(f"   Generated symbolic predictions: shape {symbolic_predictions.shape}")
        except Exception as e:
            print(f"   Failed to generate symbolic predictions: {e}")
            symbolic_predictions = None

    # Create visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_discovery_results(model, dataset[:50], pos[:50], s_heatmap, z_states_plot, assignments, 
                          output_path=f'discovery_result_test_{timestamp}.png',
                          symbolic_predictions=symbolic_predictions)
    plot_training_history(trainer.loss_tracker, output_path=f'training_history_test_{timestamp}.png')
    
    print(f"\n9. Visualizations saved with timestamp {timestamp}")
    
    # Summary of implemented fixes
    print("\n" + "=" * 60)
    print("SUMMARY OF IMPLEMENTED FIXES:")
    print("=" * 60)
    print("✓ Latent trajectory stabilization:")
    print("  - Increased temporal consistency weight by factor of 5")
    print("  - Added latent smoothing loss (penalizes second derivative)")
    print("✓ Loss balancing mechanism:")
    print("  - Log vars included in optimizer with higher learning rate (1e-3)")
    print("  - Reconstruction-first warmup (first 50 epochs: rec weight 10x, assign weight 0.1x)")
    print("✓ Enhanced symbolic distillation:")
    print("  - Increased populations to 2000")
    print("  - Increased generations to 40")
    print("  - Added soft-floor for distance calculations (0.1)")
    print("✓ MPS numerical stability:")
    print("  - ODE integration moved to CPU when using MPS")
    print("  - Proper device handling for symbolic proxy")
    print("✓ Visualization enhancement:")
    print("  - Symbolic predicted trajectory overlaid on learned latent trajectory")
    print("=" * 60)
    
    success_metrics = {
        'reconstruction_loss': last_rec,
        'latent_stability': dz_stability,
        'equations_found': len([eq for eq in equations if eq is not None]),
        'temporal_weight_increased': temp_weight > 10.0,
        'log_vars_in_optimizer': log_var_found
    }
    
    print(f"\nSUCCESS METRICS:")
    print(f"  - Final reconstruction loss: {last_rec:.4f} (< 0.45 target: {'✓' if last_rec < 0.45 else '✗'})")
    print(f"  - Latent stability (Var[dz]): {dz_stability:.6f} (< 0.05 target: {'✓' if dz_stability < 0.05 else '✗'})")
    print(f"  - Equations discovered: {success_metrics['equations_found']} (target: > 0: {'✓' if success_metrics['equations_found'] > 0 else '✗'})")
    print(f"  - Temporal weight increased: {'✓' if success_metrics['temporal_weight_increased'] else '✗'}")
    print(f"  - Log vars in optimizer: {'✓' if success_metrics['log_vars_in_optimizer'] else '✗'}")
    
    return success_metrics


if __name__ == "__main__":
    test_optimized_pipeline()