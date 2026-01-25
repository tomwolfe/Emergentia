import torch
import numpy as np
import argparse
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

from simulator import SpringMassSimulator, LennardJonesSimulator
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data, analyze_latent_space
from symbolic import extract_latent_data, SymbolicDistiller
from symbolic_proxy import SymbolicProxy
from hamiltonian_symbolic import HamiltonianSymbolicDistiller
from train_utils import ImprovedEarlyStopping, robust_energy, get_device
from stable_pooling import SparsityScheduler
from visualization import plot_discovery_results, plot_training_history

def print_diagnostic_dashboard(model, trainer, epoch, rec, cons, cond_proxy):
    """
    Textual Diagnostic Dashboard for the AI Physicist.
    Summarizes phase-space density and manifold health.
    """
    print(f"\n>>> DIAGNOSTIC DASHBOARD | Epoch {epoch} <<<")
    print(f"  [Losses] Rec: {rec:.6f} | Cons: {cons:.4f} | Total: {trainer.loss_tracker.history.get('total', 0):.4f}")
    
    # Phase Space Density Estimate (std of latents)
    lvars = torch.exp(trainer.model.log_vars).detach().cpu().numpy()
    print(f"  [Density] Latent SNR: {cond_proxy:.4f} | Rec Weight: {1.0/lvars[0]:.2f} | Cons Weight: {1.0/lvars[1]:.2f}")
    
    # Manifold Curvature (Log-Var of log_vars)
    print(f"  [Curvature] LogVar Std: {np.std(lvars):.4f} | Align Scale: {torch.exp(trainer.log_align_scale).item():.4f}")
    
    # Active Super-nodes
    active_mask = model.encoder.pooling.active_mask
    n_active = (active_mask > 0.5).sum().item()
    print(f"  [Resolution] Active Super-nodes: {n_active}/{len(active_mask)} | Sparsity: {trainer.model.encoder.pooling.current_sparsity_weight:.4f}")
    print("-------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Extreme Focus on Reconstruction Quality")
    parser.add_argument('--particles', type=int, default=8)
    parser.add_argument('--super_nodes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)  # Increased further
    parser.add_argument('--steps', type=int, default=40)    # Increased further
    parser.add_argument('--lr', type=float, default=5e-4)  # Slightly reduced for stability
    parser.add_argument('--sim', type=str, default='spring', choices=['spring', 'lj'])
    parser.add_argument('--hamiltonian', action='store_true')
    parser.add_argument('--non_separable', action='store_true', help='Use non-separable Hamiltonian H(q, p)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=5, help='Sequence length of each window')
    parser.add_argument('--traj_batch_size', type=int, default=4, help='Number of trajectories to process in parallel')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate every N epochs')
    parser.add_argument('--quick_symbolic', action='store_true', help='Use quick symbolic distillation')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient mode')
    parser.add_argument('--latent_dim', type=int, default=8, help='Dimension of latent space')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension of the model')
    parser.add_argument('--consistency_weight', type=float, default=0.001, help='Weight for consistency loss')
    parser.add_argument('--spatial_weight', type=float, default=0.1, help='Very low weight for spatial loss')
    parser.add_argument('--sym_weight', type=float, default=1.0, help='Weight for symbolic loss')
    parser.add_argument('--min_active', type=int, default=4, help='Minimum active super-nodes')
    parser.add_argument('--pop', type=int, default=5000, help='Symbolic population')
    parser.add_argument('--gen', type=int, default=40, help='Symbolic generations')
    args = parser.parse_args()

    if args.quick_symbolic:
        args.pop = min(args.pop, 1000)
        args.gen = min(args.gen, 15)

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
    
    # OPTIMIZATION: Store raw windows of the trajectory to allow flexible batching
    print(f"Preparing windows (size {args.batch_size})...")
    pre_batched_windows_raw = []
    if len(dataset_list) > args.batch_size:
        for i in range(len(dataset_list) - args.batch_size + 1):
            window = dataset_list[i : i + args.batch_size]
            pre_batched_windows_raw.append(window)
    else:
        pre_batched_windows_raw.append(dataset_list)

    # Full dataset for analysis
    dataset = Batch.from_data_list(dataset_list).to(device)
    dataset.seq_len = len(dataset_list)

    # 3. Setup Model & Trainer
    model = DiscoveryEngineModel(
        n_particles=args.particles,
        n_super_nodes=args.super_nodes,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        hamiltonian=args.hamiltonian,
        min_active_super_nodes=args.min_active,
        dissipative=True # Enabled by default
    ).to(device)

    # If Hamiltonian, set separable flag based on argument
    if args.hamiltonian:
        model.ode_func.separable = not args.non_separable
        print(f"  -> Hamiltonian Dynamics: {'Separable' if not args.non_separable else 'Non-Separable'}")

    # Initialize SparsityScheduler to prevent resolution closure
    # If super_nodes == particles, we disable sparsity to maintain 1-to-1 mapping
    target_sparsity = 0.05 if args.super_nodes < args.particles else 0.0
    sparsity_scheduler = SparsityScheduler(
        initial_weight=0.0,
        target_weight=target_sparsity,
        warmup_steps=50,
        max_steps=args.epochs
    )

    # Focus extremely on reconstruction in early epochs by reducing other losses significantly
    trainer = Trainer(
        model,
        lr=args.lr,
        device=device,
        stats=stats,
        warmup_epochs=int(args.epochs * 0.1),  # Reduced from 0.25
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
    stable_epoch = -1

    # Memory-efficient training loop
    for epoch in range(args.epochs):
        # OPTIMIZATION: Sample multiple windows and batch them together
        if args.traj_batch_size > 1:
            indices = np.random.randint(0, len(pre_batched_windows_raw), size=args.traj_batch_size)
            sampled_data = []
            for i in indices:
                sampled_data.extend(pre_batched_windows_raw[i])
            batch_data = Batch.from_data_list(sampled_data).to(device)
            batch_data.seq_len = args.batch_size
        else:
            window = pre_batched_windows_raw[np.random.randint(0, len(pre_batched_windows_raw))]
            batch_data = Batch.from_data_list(window).to(device)
            batch_data.seq_len = args.batch_size

        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=args.epochs)
        last_rec = rec
        loss_history.append(loss)
        if len(loss_history) > 100:
            loss_history.pop(0)

        # NEW: Distillation Gate - Check for stabilization
        stable_epoch = -1
        if len(loss_history) >= 20:
            recent_losses = loss_history[-20:]
            loss_var = np.var(recent_losses)
            if last_rec < 0.05 and loss_var < 1e-4:
                if stable_epoch == -1:
                    print(f"\n[Distillation Gate] System stabilized at epoch {epoch} (Rec: {last_rec:.4f}, Var: {loss_var:.2e})")
                    stable_epoch = epoch
                
                # If we have been stable for at least 10 epochs AND we are past a minimum training time
                if epoch >= max(50, stable_epoch + 10):
                    print(f"[Distillation Gate] Stabilization confirmed. Proceeding to discovery.")
                    break

        # NEW: Jacobian Condition Number Logging - Only if we are going to print
        if epoch % 50 == 0:
            try:
                model.eval()
                # Use a very small subset for Jacobian to avoid OOM/latency
                subset_data = Batch.from_data_list([pre_batched_windows_raw[0][0]]).to(device)
                x_input = subset_data.x.detach().requires_grad_(True)
                z_out, _, _, _ = model.encode(x_input, subset_data.edge_index, subset_data.batch)
                z0 = z_out[0, 0, 0] # Take one latent dimension
                grad = torch.autograd.grad(z0, x_input, retain_graph=True)[0]
                jacobian_norm = torch.norm(grad)
                # Approximation of condition number using norm ratio if we can't do full SVD
                latent_std = torch.std(z_out).item()
                latent_mean_mag = torch.mean(torch.abs(z_out)).item()
                cond_proxy = latent_std / (latent_mean_mag + 1e-6)
                
                # Adaptive sparsity adjustment
                if sparsity_scheduler is not None:
                    sparsity_scheduler.adjust_to_snr(cond_proxy)
                
                print_diagnostic_dashboard(model, trainer, epoch, rec, cons, cond_proxy)
            except Exception as e:
                pass
            model.train()
        elif epoch % 10 == 0:  # Print less frequently to reduce overhead
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Rec: {rec:.4f} | Cons: {cons:.4f}")

        if early_stopping(loss, rec):
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # Automated Revival Strategy: check if super-nodes are under-utilized
        if epoch > 0 and epoch % 50 == 0:
            active_count = (model.encoder.pooling.active_mask > 0.5).sum().item()
            if active_count < args.super_nodes:
                print(f"\n[Revival Check] Only {active_count}/{args.super_nodes} super-nodes active. Triggering hard revival...")
                model.encoder.pooling.apply_hard_revival()
                # Temporarily reduce sparsity weight to allow exploration
                if trainer.sparsity_scheduler:
                    trainer.sparsity_scheduler.current_step = max(0, trainer.sparsity_scheduler.current_step - 100)

    # 5. Analysis & Symbolic Discovery
    print("\n--- Discovery Health Report ---")
    from symbolic import DiscoveryOrchestrator
    
    orchestrator = DiscoveryOrchestrator(
        args.super_nodes, args.latent_dim, 
        config={'pop': args.pop, 'gen': args.gen}
    )
    
    discovery_results = orchestrator.discover(
        model, dataset, sim.dt, 
        hamiltonian=args.hamiltonian, 
        sim_type=args.sim, 
        quick=args.quick_symbolic,
        stats=stats
    )
    
    equations = discovery_results['equations']
    distiller = discovery_results['distiller']
    z_states = discovery_results['z_states']
    dz_states = discovery_results['dz_states']
    bench_report = discovery_results['bench_report']
    
    # Update trainer with symbolic proxy
    symbolic_transformer = distiller.transformer
    confidence = np.mean(distiller.confidences) if hasattr(distiller, 'confidences') else 0.8
    trainer.update_symbolic_proxy(equations, symbolic_transformer, weight=args.sym_weight, confidence=confidence)

    # Calculate correlation and flicker for the report
    is_quick = args.quick_symbolic or args.epochs < 50
    print("Calculating Correlation Matrix...")
    sample_size = min(50 if args.quick_symbolic else 200, len(dataset))
    sample_indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
    sample_dataset = [dataset_list[i] for i in sample_indices]
    sample_pos = pos[sample_indices]
    corrs = analyze_latent_space(model, sample_dataset, sample_pos, device=device)
    max_corrs = [np.max(corrs[k]) for k in range(args.super_nodes)]
    mean_max_corr = np.mean(max_corrs)

    # flicker rate ...
    print("\nCalculating Flicker Rate...")
    model.eval()
    flicker_sample_size = min(30 if args.quick_symbolic else 100, len(dataset_list))
    flicker_indices = np.linspace(0, len(dataset_list)-1, flicker_sample_size, dtype=int)
    flicker_data = [dataset_list[i] for i in flicker_indices]
    flicker_batch = Batch.from_data_list(flicker_data).to(device)
    with torch.no_grad():
        _, s_all_flicker, _, _ = model.encode(flicker_batch.x, flicker_batch.edge_index, flicker_batch.batch)
        # Reshape to [sample_size, particles, super_nodes] and take mean over particles
        s_all_flicker = s_all_flicker.view(flicker_sample_size, -1, args.super_nodes).mean(dim=1)
    flicker_rate = torch.mean(torch.abs(s_all_flicker[1:] - s_all_flicker[:-1])).item()

    # 5.5 STAGE 3: Neural-Symbolic Consistency Training (Closed-Loop)
    if trainer.symbolic_proxy is not None:
        print("\n--- Starting Stage 3: Neural-Symbolic Consistency Training ---")
        # ADAPTIVE AGGRESSION: Increase epochs and weight if correlation is low
        # Proportional to main training epochs
        stage3_base = max(1, int(args.epochs * 0.15)) if is_quick else 30
        stage3_epochs = stage3_base
        if mean_max_corr < 0.9:
            stage3_epochs = int(stage3_base * 2)
            print(f"  [Adaptive] Low correlation ({mean_max_corr:.4f}). Doubling Stage 3 epochs to {stage3_epochs}.")
        
        trainer.model.train()
        # Increase symbolic weight to force alignment
        trainer.symbolic_weight = 50.0 if mean_max_corr >= 0.9 else 150.0 # Much more aggressive if low corr
        if mean_max_corr < 0.9:
            print(f"  [Adaptive] Setting symbolic_weight to {trainer.symbolic_weight} for stronger alignment.")
        
        # Unfreeze encoder and symbolic proxy for joint optimization
        for p in trainer.model.encoder.parameters(): p.requires_grad = True
        
        for s3_epoch in range(stage3_epochs):
            # Sample window
            indices = np.random.randint(0, len(pre_batched_windows_raw), size=args.traj_batch_size)
            sampled_data = []
            for i in indices: sampled_data.extend(pre_batched_windows_raw[i])
            batch_data = Batch.from_data_list(sampled_data).to(device)
            batch_data.seq_len = args.batch_size
            
            # Step with higher symbolic weight
            # We want to force the encoder to align with the symbolic proxy
            loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=args.epochs + s3_epoch, max_epochs=args.epochs + stage3_epochs)
            
            if s3_epoch % 10 == 0:
                print(f"Stage 3 | Epoch {s3_epoch:2d} | Loss: {loss:.4f} | Rec: {rec:.4f} | Cons: {cons:.4f}")

    # 6. Visualization - OPTIMIZED
    # Visualization - OPTIMIZED
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

    # 7. Final Report
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(results_dir, f"model_{timestamp}.pt")
    torch.save(model.state_dict(), model_save_path)

    def make_serializable(obj):
        if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list): return [make_serializable(v) for v in obj]
        elif hasattr(obj, 'item'): return obj.item()
        elif isinstance(obj, (np.float32, np.float64)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    final_report = make_serializable({
        'discovered_sympy': [str(eq) for eq in equations],
        'health_check': {
            "Symbolic R2": float(bench_report.get('symbolic_r2', 0.0)),
            "Energy Drift": bench_report.get('energy_conservation_error', 1.0),
            "OOD R2": bench_report.get('symbolic_r2_ood', 0.0),
            "Rotational Error": bench_report.get('rotational_invariance_error', 0.0),
            "Active Super-nodes": (model.encoder.pooling.active_mask > 0.5).sum().item(),
            "Flicker Rate": flicker_rate
        },
        'config': {
            'particles': args.particles,
            'super_nodes': args.super_nodes,
            'sim': args.sim,
            'hamiltonian': args.hamiltonian
        }
    })
    
    with open('discovery_report.json', 'w') as f:
        json.dump(final_report, f, indent=4)
    print(f"Final report saved. Model: {model_save_path}")

if __name__ == "__main__":
    main()