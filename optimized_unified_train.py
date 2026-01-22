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

def main():
    parser = argparse.ArgumentParser(description="Unified Emergentia Training Pipeline - Optimized")
    parser.add_argument('--particles', type=int, default=16)
    parser.add_argument('--super_nodes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--sim', type=str, default='spring', choices=['spring', 'lj'])
    parser.add_argument('--hamiltonian', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training steps')
    parser.add_argument('--eval_every', type=int, default=50, help='Evaluate every N epochs')
    parser.add_argument('--quick_symbolic', action='store_true', help='Use quick symbolic distillation')
    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else args.device
    print(f"Using device: {device}")

    # 1. Setup Simulator
    if args.sim == 'spring':
        sim = SpringMassSimulator(n_particles=args.particles, dynamic_radius=2.0)
    else:
        sim = LennardJonesSimulator(n_particles=args.particles, dynamic_radius=2.0)

    # 2. Generate Data
    print("Generating trajectory...")
    pos, vel = sim.generate_trajectory(steps=args.steps)
    dataset, stats = prepare_data(pos, vel, radius=2.0, device=device)

    # 3. Setup Model & Trainer
    model = DiscoveryEngineModel(
        n_particles=args.particles,
        n_super_nodes=args.super_nodes,
        latent_dim=4,
        hamiltonian=args.hamiltonian
    ).to(device)

    # Initialize SparsityScheduler to prevent resolution collapse
    sparsity_scheduler = SparsityScheduler(
        initial_weight=0.0,
        target_weight=0.1,
        warmup_steps=args.epochs // 2,
        max_steps=args.epochs
    )

    trainer = Trainer(
        model,
        lr=args.lr,
        device=device,
        stats=stats,
        warmup_epochs=50, # Stage 1: Train rec and assign
        max_epochs=args.epochs,
        sparsity_scheduler=sparsity_scheduler
    )
    early_stopping = ImprovedEarlyStopping(patience=300)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)

    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    last_rec = 1.0
    
    # Optimized training loop
    for epoch in range(args.epochs):
        # Random batch selection for training
        idx = np.random.randint(0, len(dataset) - args.batch_size)
        batch_data = dataset[idx : idx + args.batch_size]
        
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=args.epochs)
        scheduler.step()
        last_rec = rec

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Rec: {rec:.4f} | Cons: {cons:.4f}")

        if early_stopping(loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 5. Analysis & Symbolic Discovery
    print("Extracting latent data for visualization...")
    z_states, dz_states, t_states = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=args.hamiltonian)

    # Compute stability metric: variance of latent derivatives over the last 20% of the trajectory
    # dz_states: [T, K, D]
    stability_window = max(10, int(len(dz_states) * 0.2))
    recent_dz = dz_states[-stability_window:]
    dz_stability = np.var(recent_dz)
    print(f"Latent Stability (Var[dz]): {dz_stability:.6f}")

    # Adjusted Quality Gate: Relax threshold if latent space is stable
    rec_threshold = 0.45 if dz_stability < 0.01 else 0.25

    if epoch < 100 or (last_rec > rec_threshold and dz_stability > 0.05):
        reason = ""
        if epoch < 100: reason += "Insufficient epochs. "
        if last_rec > rec_threshold: reason += f"Rec Loss too high ({last_rec:.4f} > {rec_threshold}). "
        if dz_stability > 0.05: reason += f"Latent space unstable ({dz_stability:.4f} > 0.05). "
        print(f"Skipping symbolic discovery: {reason}")
        equations = []
        symbolic_transformer = None
    else:
        print("Analyzing latent space...")
        # Sample data for analysis to reduce computation time
        sample_size = min(100, len(dataset))
        sample_indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
        sample_dataset = [dataset[i] for i in sample_indices]
        sample_pos = pos[sample_indices]
        
        corrs = analyze_latent_space(model, sample_dataset, sample_pos, device=device)

        print("Performing symbolic distillation...")
        if args.hamiltonian:
            # Reduce populations and generations for faster execution
            populations = 500 if args.quick_symbolic else 2000
            generations = 10 if args.quick_symbolic else 40
            from hamiltonian_symbolic import HamiltonianSymbolicDistiller
            distiller = HamiltonianSymbolicDistiller(populations=populations, generations=generations)
            equations = distiller.distill(z_states, dz_states, args.super_nodes, 4, model=model)
        else:
            # Reduce populations and generations for faster execution
            populations = 500 if args.quick_symbolic else 2000
            generations = 10 if args.quick_symbolic else 40
            from symbolic import SymbolicDistiller
            distiller = SymbolicDistiller(populations=populations, generations=generations)
            equations = distiller.distill(z_states, dz_states, args.super_nodes, 4)

        print("\nDiscovered Equations:")
        for i, eq in enumerate(equations):
            target = "H" if args.hamiltonian else f"dz_{i}/dt"
            print(f"{target} = {eq}")

        # Create symbolic transformer and update trainer with symbolic proxy
        if hasattr(distiller, 'transformer'):
            symbolic_transformer = distiller.transformer
        else:
            from symbolic import FeatureTransformer
            symbolic_transformer = FeatureTransformer(args.super_nodes, 4)
            symbolic_transformer.fit(z_states, dz_states)

        # Update trainer with symbolic proxy
        trainer.update_symbolic_proxy(equations, symbolic_transformer, weight=0.1, confidence=0.8)

    # 6. Visualization - OPTIMIZED
    print("Visualizing results...")
    # Only compute assignments for a subset to reduce computation
    sample_size = min(50, len(dataset))
    sample_indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
    sample_dataset = [dataset[i] for i in sample_indices]
    
    # Get assignments for sampled data
    model.eval()
    s_list = []
    with torch.no_grad():
        for data in sample_dataset:
            batch = Batch.from_data_list([data]).to(device)
            _, s, _, _ = model.encode(batch.x, batch.edge_index, batch.batch)
            s_list.append(s.cpu())

    s_full = torch.cat(s_list, dim=0) # [sample_size*N, K]
    # Reshape to [sample_size, N, K] and take mean over particles N for the heatmap
    s_heatmap = s_full.view(sample_size, args.particles, args.super_nodes).mean(dim=1)

    # Get final assignments for scatter plot
    batch_0 = Batch.from_data_list([dataset[0]]).to(device)
    _, s_0, _, _ = model.encode(batch_0.x, batch_0.edge_index, batch_0.batch)
    assignments = torch.argmax(s_0, dim=1).cpu().numpy()

    # Use sampled z_states for visualization
    sample_z_states = z_states[sample_indices] if len(z_states) > sample_size else z_states
    z_states_plot = sample_z_states.reshape(-1, args.super_nodes, 4) if len(sample_z_states) > 0 else np.zeros((sample_size, args.super_nodes, 4))

    # NEW: Generate symbolic predictions if symbolic equations exist
    symbolic_predictions = None
    if 'equations' in locals() and equations and trainer.symbolic_proxy is not None:
        print("Generating symbolic predictions for visualization...")
        try:
            # Convert z_states to tensor for symbolic prediction
            z_tensor = torch.tensor(z_states_plot.reshape(-1, args.super_nodes * 4), dtype=torch.float32, device=device)

            # Get symbolic predictions
            with torch.no_grad():
                symbolic_dz_dt = trainer.symbolic_proxy(z_tensor)

            # Reshape to match z_states_plot dimensions [T, K, D]
            symbolic_dz_dt = symbolic_dz_dt.cpu().numpy().reshape(-1, args.super_nodes, 4)

            # Integrate to get trajectories (simple Euler integration)
            dt = 0.01  # Use a small time step for integration
            symbolic_traj = np.zeros_like(z_states_plot)
            symbolic_traj[0] = z_states_plot[0]  # Initialize with first state

            for t in range(1, len(symbolic_traj)):
                # Simple Euler integration: z_new = z_old + dt * dz_dt
                symbolic_traj[t] = symbolic_traj[t-1] + dt * symbolic_dz_dt[t-1]

            symbolic_predictions = symbolic_traj
            print(f"Generated symbolic predictions: shape {symbolic_predictions.shape}")
        except Exception as e:
            print(f"Failed to generate symbolic predictions: {e}")
            symbolic_predictions = None

    plot_discovery_results(model, sample_dataset, pos[sample_indices], s_heatmap, z_states_plot, assignments, symbolic_predictions=symbolic_predictions)
    plot_training_history(trainer.loss_tracker)

if __name__ == "__main__":
    main()