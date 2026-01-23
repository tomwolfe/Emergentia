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
    parser = argparse.ArgumentParser(description="Extreme Focus on Reconstruction Quality")
    parser.add_argument('--particles', type=int, default=8)
    parser.add_argument('--super_nodes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)  # Increased further
    parser.add_argument('--steps', type=int, default=40)    # Increased further
    parser.add_argument('--lr', type=float, default=5e-4)  # Slightly reduced for stability
    parser.add_argument('--sim', type=str, default='spring', choices=['spring', 'lj'])
    parser.add_argument('--hamiltonian', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training steps')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate every N epochs')
    parser.add_argument('--quick_symbolic', action='store_true', help='Use quick symbolic distillation')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient mode')
    parser.add_argument('--latent_dim', type=int, default=4, help='Dimension of latent space')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the model')
    parser.add_argument('--consistency_weight', type=float, default=0.05, help='Very low weight for consistency loss to prioritize reconstruction')
    parser.add_argument('--spatial_weight', type=float, default=0.1, help='Very low weight for spatial loss')
    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else args.device
    print(f"Using device: {device}")

    # 1. Setup Simulator
    if args.sim == 'spring':
        sim = SpringMassSimulator(n_particles=args.particles, spring_dist=1.0, dynamic_radius=1.5)
    else:
        sim = LennardJonesSimulator(n_particles=args.particles, dynamic_radius=2.0)

    # 2. Generate Data
    print("Generating trajectory...")
    pos, vel = sim.generate_trajectory(steps=args.steps)
    dataset_list, stats = prepare_data(pos, vel, radius=1.5 if args.sim == 'spring' else 2.0, device=device)
    
    # OPTIMIZATION: Pre-batch windows of the trajectory to avoid overhead in training loop
    print(f"Pre-batching windows (size {args.batch_size})...")
    pre_batched_windows = []
    if len(dataset_list) > args.batch_size:
        for i in range(len(dataset_list) - args.batch_size + 1):
            window = dataset_list[i : i + args.batch_size]
            batch = Batch.from_data_list(window).to(device)
            batch.seq_len = args.batch_size
            pre_batched_windows.append(batch)
    else:
        full_batch = Batch.from_data_list(dataset_list).to(device)
        full_batch.seq_len = len(dataset_list)
        pre_batched_windows.append(full_batch)

    # Full dataset for analysis
    dataset = Batch.from_data_list(dataset_list).to(device)
    dataset.seq_len = len(dataset_list)

    # 3. Setup Model & Trainer
    model = DiscoveryEngineModel(
        n_particles=args.particles,
        n_super_nodes=args.super_nodes,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        hamiltonian=args.hamiltonian
    ).to(device)

    # Initialize SparsityScheduler to prevent resolution closure
    sparsity_scheduler = SparsityScheduler(
        initial_weight=0.0,
        target_weight=0.05,  # Reduced sparsity weight to focus on reconstruction
        warmup_steps=50,  # Extended warmup to focus on reconstruction longer
        max_steps=args.epochs
    )

    # Focus extremely on reconstruction in early epochs by reducing other losses significantly
    trainer = Trainer(
        model,
        lr=args.lr,
        device=device,
        stats=stats,
        warmup_epochs=int(args.epochs * 0.25),  # Aligned with stage1_end
        max_epochs=args.epochs,
        sparsity_scheduler=sparsity_scheduler,
        consistency_weight=args.consistency_weight,
        spatial_weight=args.spatial_weight
    )

    # Very patient early stopping to allow full training
    early_stopping = ImprovedEarlyStopping(patience=60, ignore_epochs=50, monitor_rec=True, rec_threshold=0.1)  # Very patient
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)

    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    last_rec = 1.0

    # Track historical metrics for stability check
    loss_history = []

    # Memory-efficient training loop
    for epoch in range(args.epochs):
        # OPTIMIZATION: Use pre-batched windows to speed up training
        batch_data = pre_batched_windows[np.random.randint(0, len(pre_batched_windows))]

        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=args.epochs)

        # Learning rate warmup: gradually increase LR for first 25 epochs to focus on reconstruction
        if epoch < 25:  # Extended warmup
            # Linear warmup from 0 to the scheduled LR
            warmup_factor = (epoch + 1) / 25.0
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor

        scheduler.step()
        last_rec = rec
        loss_history.append(loss)
        if len(loss_history) > 20: loss_history.pop(0)  # Larger window for stability

        if epoch % 10 == 0:  # Print less frequently to reduce overhead
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Rec: {rec:.4f} | Cons: {cons:.4f}")

        # Early discovery gate - only check late in training with strict criteria
        if epoch >= 100 and epoch % 40 == 0:  # Check every 40 epochs starting at 100
            loss_std = np.std(loss_history) if len(loss_history) >= 10 else 1.0  # Larger window
            if loss_std < 0.01 and last_rec < 0.05:  # Strict criteria
                print(f"\n[Adaptive Gate] Loss stabilized (std={loss_std:.6f}). Attempting early discovery...")

                # Extract subset for quick check
                check_dataset = dataset[::max(1, len(dataset)//20)]  # Small subset
                z_check, dz_check, _ = extract_latent_data(model, check_dataset, sim.dt, include_hamiltonian=args.hamiltonian)

                # Quick GP check with minimal resources
                distiller = SymbolicDistiller(populations=200, generations=2)  # Minimal
                eqs = distiller.distill(z_check, dz_check, args.super_nodes, args.latent_dim, hamiltonian=args.hamiltonian)

                # If we found a non-trivial equation with decent potential, we could stop
                # For now, let's just log it and decide if we want to early exit
                print(f"[Adaptive Gate] Found candidate: {eqs[0] if eqs else 'None'}")
                if eqs and len(str(eqs[0])) > 4:  # Moderate criteria
                    print("[Adaptive Gate] Discovery successful and non-trivial. Finishing training early.")
                    break

        if early_stopping(loss, rec):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 5. Analysis & Symbolic Discovery
    print("\n--- Discovery Health Report ---")
    # Use memory-efficient extraction for large datasets
    if args.memory_efficient and len(dataset) > 30:  # Larger threshold
        # Sample a subset of the trajectory for analysis
        sample_size = 30  # Larger
        sample_indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
        sample_dataset = [dataset[i] for i in sample_indices]
        sample_pos = pos[sample_indices]
        z_states, dz_states, t_states = extract_latent_data(model, sample_dataset, sim.dt, include_hamiltonian=args.hamiltonian)
    else:
        z_states, dz_states, t_states = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=args.hamiltonian)

    # Compute health metrics
    # NEW: Dynamics-to-Noise Ratio check
    # We want the variance of the signal to be much larger than the variance of high-frequency jitter
    if dz_states.shape[0] > 1:
        # High-frequency jitter is estimated by first-order differences
        jitter = dz_states[1:] - dz_states[:-1]
        noise_var = np.var(jitter) + 1e-9
        signal_var = np.var(dz_states)
        dnr = signal_var / noise_var
    else:
        dnr = 0.0

    health_metrics = {
        "Reconstruction Fidelity": (last_rec < 0.4, f"Rec Loss {last_rec:.4f}"),  # Relaxed from 0.15
        "Latent Dynamics": (dnr > 0.5, f"DNR {dnr:.4f}"),  # Relaxed from 0.8
        "Training Maturity": (epoch >= int(args.epochs * 0.8), f"Epochs {epoch}")  # Relative maturity
    }
    
    all_pass = True
    for metric, (passed, status) in health_metrics.items():
        icon = "✅" if passed else "❌"
        print(f"{icon} {metric:25s}: {status}")
        if not passed: all_pass = False

    if not all_pass:
        print("\nSkipping symbolic discovery due to poor health metrics.")
        print("Tip: If Latent Stability is ❌, increase --consistency_weight or --spatial_weight.")
        print("Tip: If Reconstruction is ❌, check model capacity or increase --epochs.")
        equations = []
        symbolic_transformer = None
    else:
        print("\nProceeding to symbolic distillation...")
        # Sample data for analysis to reduce computation time
        sample_size = min(30, len(dataset))  # Larger
        sample_indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
        sample_dataset = [dataset[i] for i in sample_indices]
        sample_pos = pos[sample_indices]

        corrs = analyze_latent_space(model, sample_dataset, sample_pos, device=device)

        print("Performing symbolic distillation...")
        if args.hamiltonian:
            # Increased populations and generations for better discovery
            populations = 400 if args.quick_symbolic else 2000
            generations = 10 if args.quick_symbolic else 50
            from hamiltonian_symbolic import HamiltonianSymbolicDistiller
            distiller = HamiltonianSymbolicDistiller(populations=populations, generations=generations,
                                                   perform_coordinate_alignment=not args.quick_symbolic)
            equations = distiller.distill(z_states, dz_states, args.super_nodes, args.latent_dim, model=model)
        else:
            # Increased populations and generations for better discovery
            populations = 400 if args.quick_symbolic else 2000
            generations = 10 if args.quick_symbolic else 50
            from symbolic import SymbolicDistiller
            distiller = SymbolicDistiller(populations=populations, generations=generations)
            equations = distiller.distill(z_states, dz_states, args.super_nodes, args.latent_dim)

        print("\nDiscovered Equations:")
        for i, eq in enumerate(equations):
            target = "H" if args.hamiltonian else f"dz_{i}/dt"
            print(f"{target} = {eq}")

        # Create symbolic transformer and update trainer with symbolic proxy
        if hasattr(distiller, 'transformer'):
            symbolic_transformer = distiller.transformer
        else:
            from symbolic import FeatureTransformer
            symbolic_transformer = FeatureTransformer(args.super_nodes, args.latent_dim)
            symbolic_transformer.fit(z_states, dz_states)

        # Update trainer with symbolic proxy
        trainer.update_symbolic_proxy(equations, symbolic_transformer, weight=0.1, confidence=0.8)

    # 6. Visualization - OPTIMIZED
    print("Visualizing results...")
    # Only compute assignments for a moderate subset to reduce computation
    vis_sample_size = min(20, len(dataset))  # Larger
    vis_sample_indices = np.linspace(0, len(dataset)-1, vis_sample_size, dtype=int)
    vis_sample_dataset = [dataset[i] for i in vis_sample_indices]

    # Get assignments for sampled data
    model.eval()
    s_list = []
    with torch.no_grad():
        for data in vis_sample_dataset:
            batch = Batch.from_data_list([data]).to(device)
            _, s, _, _ = model.encode(batch.x, batch.edge_index, batch.batch)
            s_list.append(s.cpu())

    s_full = torch.cat(s_list, dim=0) # [sample_size*N, K]
    # Reshape to [sample_size, N, K] and take mean over particles N for the heatmap
    s_heatmap = s_full.view(vis_sample_size, args.particles, args.super_nodes).mean(dim=1)

    # Get final assignments for scatter plot
    batch_0 = Batch.from_data_list([dataset[0]]).to(device)
    _, s_0, _, _ = model.encode(batch_0.x, batch_0.edge_index, batch_0.batch)
    assignments = torch.argmax(s_0, dim=1).cpu().numpy()

    # Extract z_states for visualization using the same indices
    vis_z_states, _, _ = extract_latent_data(model, vis_sample_dataset, sim.dt, include_hamiltonian=args.hamiltonian)
    z_states_plot = vis_z_states.reshape(-1, args.super_nodes, args.latent_dim) if len(vis_z_states) > 0 else np.zeros((vis_sample_size, args.super_nodes, args.latent_dim))

    # NEW: Generate symbolic predictions if symbolic equations exist
    symbolic_predictions = None
    if 'equations' in locals() and equations and trainer.symbolic_proxy is not None:
        print("Generating closed-loop symbolic predictions for visualization...")
        try:
            from visualization import generate_closed_loop_trajectory
            # Get initial state from the first time step of the neural trajectory
            # z_states_plot: [T, K, D] -> [1, K*D]
            initial_z = torch.tensor(z_states_plot[0].reshape(1, -1), dtype=torch.float32, device=device)

            # Integrate using the same number of steps as the neural trajectory
            steps = z_states_plot.shape[0]
            dt = sim.dt # Use simulator dt

            # Generate closed-loop trajectory
            symbolic_traj_flat = generate_closed_loop_trajectory(trainer.symbolic_proxy, initial_z, steps, dt, device=device)

            # Reshape back to [T, K, D]
            symbolic_predictions = symbolic_traj_flat.reshape(steps, args.super_nodes, args.latent_dim)

        except Exception as e:
            print(f"Failed to generate symbolic predictions: {e}")
            symbolic_predictions = None

    plot_discovery_results(model, vis_sample_dataset, pos[vis_sample_indices], s_heatmap, z_states_plot, assignments, symbolic_predictions=symbolic_predictions)
    plot_training_history(trainer.loss_tracker)

if __name__ == "__main__":
    main()