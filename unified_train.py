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
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sim', type=str, default='spring', choices=['spring', 'lj'])
    parser.add_argument('--hamiltonian', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training steps')
    parser.add_argument('--eval_every', type=int, default=50, help='Evaluate every N epochs')
    parser.add_argument('--quick_symbolic', action='store_true', help='Use quick symbolic distillation')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient mode')
    parser.add_argument('--latent_dim', type=int, default=8, help='Dimension of latent space')
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
        latent_dim=args.latent_dim,
        hamiltonian=args.hamiltonian
    ).to(device)

    # Initialize SparsityScheduler to prevent resolution collapse
    sparsity_scheduler = SparsityScheduler(
        initial_weight=0.0,
        target_weight=0.1,
        warmup_steps=400, # Stay at 0.0 until epoch 400
        max_steps=args.epochs
    )

    trainer = Trainer(
        model,
        lr=args.lr,
        device=device,
        stats=stats,
        warmup_epochs=int(args.epochs * 0.25), # Stage 1: Train rec and assign - 25% of total epochs
        max_epochs=args.epochs,
        sparsity_scheduler=sparsity_scheduler
    )
    early_stopping = ImprovedEarlyStopping(patience=300)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)

    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    last_rec = 1.0

    # Track historical metrics for stability check
    loss_history = []

    # Memory-efficient training loop
    for epoch in range(args.epochs):
        # Random batch selection for training
        idx = np.random.randint(0, len(dataset) - args.batch_size)
        batch_data = dataset[idx : idx + args.batch_size]

        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=args.epochs)

        # Learning rate warmup: gradually increase LR for first 20 epochs
        if epoch < 20:
            # Linear warmup from 0 to the scheduled LR
            warmup_factor = (epoch + 1) / 20.0
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = args.lr * warmup_factor

        scheduler.step()
        last_rec = rec
        loss_history.append(loss)
        if len(loss_history) > 50: loss_history.pop(0)

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Rec: {rec:.4f} | Cons: {cons:.4f}")

        # PARETO OPTIMIZATION: Adaptive Discovery Gate
        # Every 200 epochs (after warmup), check if we can discover equations early
        if epoch >= 400 and epoch % 200 == 0:
            loss_std = np.std(loss_history) if len(loss_history) >= 20 else 1.0
            if loss_std < 0.005 and last_rec < 0.02:
                print(f"\n[Adaptive Gate] Loss stabilized (std={loss_std:.6f}). Attempting early discovery...")
                
                # Extract subset for quick check
                check_dataset = dataset[::max(1, len(dataset)//50)]
                z_check, dz_check, _ = extract_latent_data(model, check_dataset, sim.dt, include_hamiltonian=args.hamiltonian)
                
                # Quick GP check
                distiller = SymbolicDistiller(populations=1000, generations=5)
                eqs = distiller.distill(z_check, dz_check, args.super_nodes, 4, hamiltonian=args.hamiltonian)
                
                # If we found a non-trivial equation with decent potential, we could stop
                # For now, let's just log it and decide if we want to early exit
                print(f"[Adaptive Gate] Found candidate: {eqs[0]}")
                if len(str(eqs[0])) > 5: # Not just "X0" or "0.5"
                    print("[Adaptive Gate] Discovery successful and non-trivial. Finishing training early.")
                    break

        if early_stopping(loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 5. Analysis & Symbolic Discovery
    print("\n--- Discovery Health Report ---")
    # Use memory-efficient extraction for large datasets
    if args.memory_efficient and len(dataset) > 100:
        # Sample a subset of the trajectory for analysis
        sample_size = 100
        sample_indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
        sample_dataset = [dataset[i] for i in sample_indices]
        sample_pos = pos[sample_indices]
        z_states, dz_states, t_states = extract_latent_data(model, sample_dataset, sim.dt, include_hamiltonian=args.hamiltonian)
    else:
        z_states, dz_states, t_states = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=args.hamiltonian)

    # Compute health metrics
    stability_window = max(10, int(len(dz_states) * 0.2))
    recent_dz = dz_states[-stability_window:]
    dz_stability = np.var(recent_dz)
    
    health_metrics = {
        "Reconstruction Fidelity": (last_rec < 0.05, f"Rec Loss {last_rec:.4f}"),
        "Latent Stability": (dz_stability < 0.2, f"Var[dz] {dz_stability:.4f}"),
        "Training Maturity": (epoch >= 200, f"Epochs {epoch}")
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
        sample_size = min(100, len(dataset))
        sample_indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
        sample_dataset = [dataset[i] for i in sample_indices]
        sample_pos = pos[sample_indices]
        
        corrs = analyze_latent_space(model, sample_dataset, sample_pos, device=device)

        print("Performing symbolic distillation...")
        if args.hamiltonian:
            # INCREASED: populations to 5000 and generations to 100 for better convergence
            populations = 1000 if args.quick_symbolic else 5000
            generations = 20 if args.quick_symbolic else 100
            from hamiltonian_symbolic import HamiltonianSymbolicDistiller
            distiller = HamiltonianSymbolicDistiller(populations=populations, generations=generations)
            equations = distiller.distill(z_states, dz_states, args.super_nodes, 4, model=model)
        else:
            # INCREASED: populations to 5000 and generations to 100 for better convergence
            populations = 1000 if args.quick_symbolic else 5000
            generations = 20 if args.quick_symbolic else 100
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
    vis_sample_size = min(50, len(dataset))
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
    z_states_plot = vis_z_states.reshape(-1, args.super_nodes, 4) if len(vis_z_states) > 0 else np.zeros((vis_sample_size, args.super_nodes, 4))

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

            # Reshape to match z_states_plot dimensions [T, K, D], ensuring compatibility
            expected_shape = z_states_plot.shape
            num_elements_needed = expected_shape[0] * expected_shape[1] * expected_shape[2]
            symbolic_dz_dt_flat = symbolic_dz_dt.cpu().numpy().flatten()
            
            # Trim or pad if needed to match expected size
            if len(symbolic_dz_dt_flat) > num_elements_needed:
                symbolic_dz_dt_flat = symbolic_dz_dt_flat[:num_elements_needed]
            elif len(symbolic_dz_dt_flat) < num_elements_needed:
                # Pad with zeros if shorter
                padding = np.zeros(num_elements_needed - len(symbolic_dz_dt_flat))
                symbolic_dz_dt_flat = np.concatenate([symbolic_dz_dt_flat, padding])

            symbolic_dz_dt = symbolic_dz_dt_flat.reshape(expected_shape)

            # Integrate to get trajectories (simple Euler integration)
            dt = 0.01  # Use a small time step for integration
            symbolic_traj = np.zeros_like(z_states_plot)
            symbolic_traj[0] = z_states_plot[0]  # Initialize with first state

            for t in range(1, len(symbolic_traj)):
                # Simple Euler integration: z_new = z_old + dt * dz_dt
                if t-1 < len(symbolic_dz_dt):
                    symbolic_traj[t] = symbolic_traj[t-1] + dt * symbolic_dz_dt[t-1]
                else:
                    symbolic_traj[t] = symbolic_traj[t-1]

            symbolic_predictions = symbolic_traj
        except RuntimeError as e:
            if "size of tensor" in str(e) and "match the size" in str(e):
                print("Note: Symbolic proxy size mismatch. Skipping symbolic trajectory visualization.")
                symbolic_predictions = None
            else:
                raise
        except Exception as e:
            print(f"Failed to generate symbolic predictions: {e}")
            symbolic_predictions = None

    plot_discovery_results(model, vis_sample_dataset, pos[vis_sample_indices], s_heatmap, z_states_plot, assignments, symbolic_predictions=symbolic_predictions)
    plot_training_history(trainer.loss_tracker)

if __name__ == "__main__":
    main()