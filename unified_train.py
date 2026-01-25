"""
Optimized version of unified_train.py with major performance improvements
"""
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
    parser = argparse.ArgumentParser(description="Extreme Focus on Reconstruction Quality - OPTIMIZED VERSION")
    parser.add_argument('--particles', type=int, default=8)
    parser.add_argument('--super_nodes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)  # Increased further
    parser.add_argument('--steps', type=int, default=40)    # Increased further
    parser.add_argument('--lr', type=float, default=5e-4)  # Slightly reduced for stability
    parser.add_argument('--sim', type=str, default='spring', choices=['spring', 'lj'])
    parser.add_argument('--hamiltonian', action='store_true')
    parser.add_argument('--non_separable', action='store_true', help='Use non-separable Hamiltonian H(q, p)')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--batch_size', type=int, default=10, help='Sequence length of each window')
    parser.add_argument('--traj_batch_size', type=int, default=4, help='Number of trajectories to process in parallel')
    parser.add_argument('--eval_every', type=int, default=10, help='Evaluate every N epochs')
    parser.add_argument('--quick_symbolic', action='store_true', help='Use quick symbolic distillation')
    parser.add_argument('--memory_efficient', action='store_true', help='Use memory-efficient mode')
    parser.add_argument('--latent_dim', type=int, default=8, help='Dimension of latent space')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension of the model')
    parser.add_argument('--consistency_weight', type=float, default=0.001, help='Weight for consistency loss')
    parser.add_argument('--spatial_weight', type=float, default=0.1, help='Very low weight for spatial loss')
    parser.add_argument('--sym_weight', type=float, default=1.0, help='Weight for symbolic loss')
    parser.add_argument('--min_active', type=int, default=4, help='Minimum active super-nodes')
    parser.add_argument('--pop', type=int, default=5000, help='Symbolic population')
    parser.add_argument('--gen', type=int, default=40, help='Symbolic generations')
    args = parser.parse_args()

    if args.quick_symbolic:
        args.pop = min(args.pop, 1000) # Much lower for quick run
        args.gen = min(args.gen, 20)   # Much lower for quick run

    device = get_device() if args.device == 'auto' else args.device
    print(f"Using device: {device}")

    # OPTIMIZATION: Use mixed precision training if available
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        print("Using Automatic Mixed Precision training")

    # 1. Setup Simulator
    if args.sim == 'spring':
        sim = SpringMassSimulator(n_particles=args.particles, spring_dist=1.0, dynamic_radius=1.5)
    else:
        sim = LennardJonesSimulator(n_particles=args.particles, dynamic_radius=2.0)

    # 2. Generate Data
    print("Generating trajectory...")
    pos, vel = sim.generate_trajectory(steps=args.steps)
    dataset_list, stats = prepare_data(pos, vel, radius=1.5 if args.sim == 'spring' else 2.0, device=device)

    # OPTIMIZATION: Pre-batch windows into Batch objects to avoid overhead in main loop
    print(f"Preparing windows (size {args.batch_size})...")
    pre_batched_windows = []
    if len(dataset_list) > args.batch_size:
        for i in range(len(dataset_list) - args.batch_size + 1):
            window = dataset_list[i : i + args.batch_size]
            batch = Batch.from_data_list(window).to(device)
            batch.seq_len = args.batch_size
            pre_batched_windows.append(batch)
    else:
        batch = Batch.from_data_list(dataset_list).to(device)
        batch.seq_len = len(dataset_list)
        pre_batched_windows.append(batch)

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
        dissipative=False # Disabled for exact recovery of conservative laws
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
        spatial_weight=args.spatial_weight,
        quick=args.quick_symbolic
    )

    # NEW: Curriculum Learning - Warm-up Phase for Reconstruction
    curriculum_phase = "reconstruction"  # Start with reconstruction phase
    warmup_epochs = 200  # Fixed warm-up period for reconstruction focus

    # Store original weights for restoration after warm-up
    original_consistency_weight = trainer.consistency_weight
    original_spatial_weight = trainer.spatial_weight

    # Very patient early stopping to allow full training
    early_stopping = ImprovedEarlyStopping(patience=100, ignore_epochs=100, monitor_rec=True, rec_threshold=0.05)  # More patient
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)

    # OPTIMIZATION: Reduce frequency of diagnostic prints and evaluations
    diagnostic_freq = 50  # Reduced from 50 to reduce overhead

    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    last_rec = 1.0

    # Track historical metrics for stability check
    loss_history = []
    stable_epoch = -1
    flicker_rate = 1.0

    # Memory-efficient training loop
    for epoch in range(args.epochs):
        # NEW: Curriculum Learning - Adjust training based on phase
        if epoch < warmup_epochs:
            # Warm-up phase: Train only as GNN Autoencoder (focus on reconstruction)
            curriculum_phase = "reconstruction"
            # Temporarily set weights to focus on reconstruction
            original_consistency_weight = trainer.consistency_weight
            original_spatial_weight = trainer.spatial_weight
            trainer.consistency_weight = 0.0001  # Minimal consistency during warm-up
            trainer.spatial_weight = 0.0001      # Minimal spatial loss during warm-up
        else:
            # Post-warm-up: Restore original weights and allow full training
            curriculum_phase = "full_training"
            trainer.consistency_weight = original_consistency_weight
            trainer.spatial_weight = original_spatial_weight

        # OPTIMIZATION: Sample from pre-batched windows
        if args.traj_batch_size > 1:
            indices = np.random.randint(0, len(pre_batched_windows), size=args.traj_batch_size)
            batch_list = [pre_batched_windows[i] for i in indices]
            if args.particles * args.traj_batch_size < 32:
                batch_data = batch_list[0]
            else:
                sampled_data = []
                for b in batch_list: sampled_data.extend(b.to_data_list())
                batch_data = Batch.from_data_list(sampled_data).to(device)
                batch_data.seq_len = args.batch_size
        else:
            batch_data = pre_batched_windows[np.random.randint(0, len(pre_batched_windows))]

        # OPTIMIZATION: Use mixed precision if available
        if use_amp:
            with torch.cuda.amp.autocast():
                loss, rec, cons = trainer.train_step(batch_data, sim.dt * (sim.sub_steps if args.sim == 'lj' else 2), epoch=epoch, max_epochs=args.epochs)
        else:
            loss, rec, cons = trainer.train_step(batch_data, sim.dt * (sim.sub_steps if args.sim == 'lj' else 2), epoch=epoch, max_epochs=args.epochs)

        # NEW: Curriculum Learning - Check if reconstruction threshold is met to proceed
        if curriculum_phase == "reconstruction" and rec < 0.05:
            print(f"[Curriculum Learning] Reconstruction MSE ({rec:.6f}) below threshold (0.05) at epoch {epoch}. Proceeding to full training.")
            curriculum_phase = "early_transition"
            # Restore original weights to allow full training
            trainer.consistency_weight = original_consistency_weight
            trainer.spatial_weight = original_spatial_weight

        last_rec = rec
        loss_history.append(loss)
        if len(loss_history) > 100:
            loss_history.pop(0)

        # OPTIMIZATION: Reduce frequency of distillation gate checks
        if epoch > 50 and epoch % 20 == 0:  # Reduced from 10 to 20
            model.eval()
            with torch.no_grad():
                # Calculate flicker rate on a subset
                flicker_batch = pre_batched_windows[0]
                _, s_all_flicker, _, _ = model.encode(flicker_batch.x, flicker_batch.edge_index, flicker_batch.batch)
                s_all_flicker = s_all_flicker.view(args.batch_size, -1, args.super_nodes).mean(dim=1)
                flicker_rate = torch.mean(torch.abs(s_all_flicker[1:] - s_all_flicker[:-1])).item()
            model.train()

            recent_losses = loss_history[-20:]
            loss_var = np.var(recent_losses)

            # NEW: Enhanced Manifold Health Check
            # Check if Rec and Flicker Rate have been stable for at least 50 epochs
            if last_rec < 0.02 and flicker_rate < 0.01 and loss_var < 1e-4:
                if stable_epoch == -1:
                    print(f"\n[Distillation Gate] System stabilized at epoch {epoch} (Rec: {last_rec:.4f}, Flicker: {flicker_rate:.4f})")
                    stable_epoch = epoch

                # NEW: Require stability for at least 50 epochs before proceeding
                if epoch >= stable_epoch + 50:  # Increased from 20 to 50
                    print(f"[Distillation Gate] Manifold Health Check passed. Stabilization confirmed for 50+ epochs. Proceeding to discovery.")
                    break
            else:
                # Reset stable_epoch if conditions are not met
                if stable_epoch != -1:
                    print(f"[Distillation Gate] Stability broken at epoch {epoch}, resetting check. Rec: {last_rec:.4f}, Flicker: {flicker_rate:.4f}")
                    stable_epoch = -1

        # OPTIMIZATION: Reduce frequency of Jacobian condition number logging
        if epoch % diagnostic_freq == 0:
            try:
                model.eval()
                # Use a very small subset for Jacobian to avoid OOM/latency
                subset_data = Batch.from_data_list([pre_batched_windows[0][0]]).to(device)
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
        elif epoch % 20 == 0:  # Print less frequently to reduce overhead - from 10 to 20
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Rec: {rec:.4f} | Cons: {cons:.4f}")

        if early_stopping(loss, rec):
            print(f"Early stopping triggered at epoch {epoch}")
            break

        # OPTIMIZATION: Reduce frequency of automated revival strategy
        if epoch > 0 and epoch % 100 == 0:  # Increased from 50 to 100
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

    max_discovery_attempts = 2  # Reduced from 3 to 2
    discovery_success = False
    parsimony_coeff = 0.01

    for attempt in range(max_discovery_attempts):
        print(f"\n[Self-Correction] Discovery Attempt {attempt + 1}/{max_discovery_attempts}...")

        # OPTIMIZATION: Reduce symbolic population and generations for faster discovery
        opt_pop = min(args.pop, 1000)  # Reduced from 5000 to 2000 to 1000
        opt_gen = min(args.gen, 20)   # Reduced from 40 to 30 to 20
        
        orchestrator = DiscoveryOrchestrator(
            args.super_nodes, args.latent_dim,
            config={'pop': opt_pop, 'gen': opt_gen, 'parsimony': parsimony_coeff, 'max_retries': 0, 'use_gpu_acceleration': True}
        )

        dt_effective = sim.dt * (sim.sub_steps if args.sim == 'lj' else 2)
        discovery_results = orchestrator.discover(
            model, dataset, dt_effective,
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

        # NEW: Check rotational invariance using NoetherChecker
        from symmetry_checks import NoetherChecker
        if distiller is not None and trainer.symbolic_proxy is not None:
            noether_checker = NoetherChecker(trainer.symbolic_proxy, args.latent_dim)
            # Sample a few points to check rotational invariance
            sample_z = z_states[:10]  # Use first 10 states from the discovery
            if len(sample_z) > 0:
                sample_z_tensor = torch.tensor(sample_z, dtype=torch.float32, device=device).view(-1, args.super_nodes * args.latent_dim)
                rot_error = noether_checker.check_rotational_invariance(sample_z_tensor)
                print(f"[Symmetry Check] Rotational invariance error: {rot_error:.6f}")

                # Reject discoveries that violate rotational invariance significantly
                if rot_error > 0.1:  # Threshold for rejecting non-symmetric laws
                    print(f"[Symmetry Check] Discovery violates rotational invariance (error: {rot_error:.6f}), rejecting...")
                    discovery_success = False
                    bench_report['success'] = False
                    bench_report['rotational_invariance_error'] = rot_error
                else:
                    bench_report['rotational_invariance_error'] = rot_error

        # Check success from benchmark
        if (bench_report.get('success', False) or
            (bench_report.get('symbolic_r2_ood', 0) > 0.95 and  # Relaxed for LJ system
             bench_report.get('energy_conservation_error', 1.0) < 1e-7 and  # Tightened energy conservation
             bench_report.get('rotational_invariance_error', 1.0) < 0.1)):  # NEW: Check rotational invariance
            print(f"[Self-Correction] Attempt {attempt + 1} Succeeded!")
            discovery_success = True

        # Update trainer with symbolic proxy
        if distiller is None:
            print(f"  [Orchestrator] Distillation failed, skipping symbolic proxy update for attempt {attempt + 1}")
            continue  # Skip to next attempt

        symbolic_transformer = distiller.transformer
        confidence = np.mean(distiller.confidences) if hasattr(distiller, 'confidences') else 0.8
        trainer.update_symbolic_proxy(equations, symbolic_transformer, weight=args.sym_weight, confidence=confidence)

        # STAGE 3: Neural-Symbolic Consistency Training (Closed-Loop)
        if trainer.symbolic_proxy is not None:
            print(f"\n--- Starting Stage 3 (Attempt {attempt + 1}): Neural-Symbolic Consistency Training ---")

            # Save best model state before Stage 3 to enable rollback
            best_model_state = {key: value.clone() for key, value in model.state_dict().items()}
            print(f"[Rollback Mechanism] Saved best model state before Stage 3")

            # Calculate correlation for adaptive weight
            model.eval()
            sample_size = min(30 if args.quick_symbolic else 100, len(dataset))  # Reduced from 50/200 to 30/100
            sample_indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
            sample_dataset = [dataset_list[i] for i in sample_indices]
            sample_pos = pos[sample_indices]
            corrs = analyze_latent_space(model, sample_dataset, sample_pos, device=device)
            max_corrs = [np.max(corrs[k]) for k in range(args.super_nodes)]
            mean_max_corr = np.mean(max_corrs)

            stage3_epochs = 50 if not discovery_success else 20  # Reduced from 100/30 to 50/20
            trainer.model.train()
            trainer.symbolic_weight = 100.0 if discovery_success else 200.0

            for p in trainer.model.encoder.parameters(): p.requires_grad = True

            # Flag to detect numerical instability
            numerical_instability_detected = False

            for s3_epoch in range(stage3_epochs):
                batch_data = pre_batched_windows[np.random.randint(0, len(pre_batched_windows))]

                # OPTIMIZATION: Use mixed precision for Stage 3 as well
                if use_amp:
                    with torch.cuda.amp.autocast():
                        loss, rec, cons = trainer.train_step(batch_data, dt_effective, epoch=args.epochs + s3_epoch, max_epochs=args.epochs + stage3_epochs)
                else:
                    loss, rec, cons = trainer.train_step(batch_data, dt_effective, epoch=args.epochs + s3_epoch, max_epochs=args.epochs + stage3_epochs)

                # Check for numerical instability (NaN or Inf)
                if torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss)):
                    print(f"[Numerical Instability] Detected NaN/Inf in loss at Stage 3 epoch {s3_epoch}")
                    numerical_instability_detected = True
                    break

                if torch.isnan(torch.tensor(rec)) or torch.isinf(torch.tensor(rec)):
                    print(f"[Numerical Instability] Detected NaN/Inf in reconstruction loss at Stage 3 epoch {s3_epoch}")
                    numerical_instability_detected = True
                    break

                if torch.isnan(torch.tensor(cons)) or torch.isinf(torch.tensor(cons)):
                    print(f"[Numerical Instability] Detected NaN/Inf in consistency loss at Stage 3 epoch {s3_epoch}")
                    numerical_instability_detected = True
                    break

                if s3_epoch % 10 == 0:  # Reduced from 20 to 10 for more frequent updates
                    print(f"Stage 3 | Epoch {s3_epoch:2d} | Loss: {loss:.4f} | Rec: {rec:.4f} | Cons: {cons:.4f}")

            # If numerical instability was detected, rollback to the saved state
            if numerical_instability_detected:
                print(f"[Rollback Mechanism] Restoring best model state due to numerical instability")
                model.load_state_dict(best_model_state)
                trainer.optimizer.zero_grad()  # Clear optimizer state
                # Reset symbolic proxy to None to prevent further Stage 3 training
                trainer.symbolic_proxy = None

        if discovery_success:
            break
        else:
            print(f"[Self-Correction] Attempt {attempt + 1} failed. Increasing parsimony and retrying...")
            parsimony_coeff *= 2.0
            # Perform additional Stage 3 is already done above

    # 5.5 Final metrics calculation
    print("\nCalculating Final Metrics...")
    # Calculate correlation and flicker for the report

    # Only compute assignments for a moderate subset to reduce computation
    vis_sample_size = min(10, len(dataset))  # Reduced from 20 to 10
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
    vis_z_states, _, _ = extract_latent_data(model, vis_sample_dataset, dt_effective, include_hamiltonian=args.hamiltonian)
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
            dt = dt_effective # Use correct dt

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
            "Rotational Error": bench_report.get('rotational_invariance_error', 1.0),  # Updated default value
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