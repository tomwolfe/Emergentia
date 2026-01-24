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
    parser.add_argument('--pop', type=int, default=1000, help='Symbolic population')
    parser.add_argument('--gen', type=int, default=20, help='Symbolic generations')
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

        # NEW: Jacobian Condition Number Logging
        if epoch % 20 == 0:
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
                print(f"  [Diagnostic] Latent SNR: {cond_proxy:.4f} | Grad Norm: {jacobian_norm.item():.4f}")
                
                # Adaptive sparsity adjustment
                if sparsity_scheduler is not None:
                    sparsity_scheduler.adjust_to_snr(cond_proxy)
            except Exception as e:
                pass
            model.train()

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

        if epoch % 50 == 0:  # Print dashboard every 50 epochs
            print_diagnostic_dashboard(model, trainer, epoch, rec, cons, cond_proxy if 'cond_proxy' in locals() else 0.0)
        elif epoch % 10 == 0:  # Print less frequently to reduce overhead
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
    
    # Extract data for analysis: h_targets for distillation, dz_states for validation
    z_states, h_targets, t_states = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=args.hamiltonian)
    _, dz_states, _ = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=False)
    
    # 1. Calculate Correlation Matrix between latents and physical CoM
    print("Calculating Correlation Matrix...")
    sample_size = min(200, len(dataset))
    sample_indices = np.linspace(0, len(dataset)-1, sample_size, dtype=int)
    sample_dataset = [dataset[i] for i in sample_indices]
    sample_pos = pos[sample_indices]
    corrs = analyze_latent_space(model, sample_dataset, sample_pos, device=device)
    
    print("\nLatent-Physical Correlation Matrix (max correlation per super-node):")
    for k in range(args.super_nodes):
        max_corr = np.max(corrs[k])
        print(f"  Super-node {k}: {max_corr:.4f}")

    # 2. Calculate Flicker Rate (mean change in assignments S)
    print("\nCalculating Flicker Rate...")
    model.eval()
    s_list = []
    with torch.no_grad():
        for data in dataset_list:
            batch = Batch.from_data_list([data]).to(device)
            _, s, _, _ = model.encode(batch.x, batch.edge_index, batch.batch)
            s_list.append(s.cpu())
    
    s_all = torch.stack(s_list) # [T, N, K]
    flicker_rate = torch.mean(torch.abs(s_all[1:] - s_all[:-1])).item()
    print(f"  Mean Flicker Rate: {flicker_rate:.6f}")

    # 3. Perform Symbolic Distillation using EnhancedSymbolicDistiller
    print("\nPerforming enhanced symbolic distillation...")
    from enhanced_symbolic import EnhancedSymbolicDistiller, PhysicsAwareSymbolicDistiller
    
    if args.hamiltonian:
        distiller = PhysicsAwareSymbolicDistiller(
            populations=args.pop, 
            generations=args.gen,
            secondary_optimization=True
        )
        equations = distiller.distill(z_states, h_targets, args.super_nodes, args.latent_dim, sim_type=args.sim, enforce_separable=not args.non_separable)
    else:
        distiller = EnhancedSymbolicDistiller(
            populations=args.pop, 
            generations=args.gen,
            secondary_optimization=True
        )
        equations = distiller.distill(z_states, h_targets, args.super_nodes, args.latent_dim, sim_type=args.sim)
    
    print("\nDiscovered Equations:")
    for i, eq in enumerate(equations):
        target = "H" if args.hamiltonian else f"dz_{i}/dt"
        print(f"{target} = {eq}")

    # 4. Calculate Symbolic R2
    print("\nCalculating Symbolic R2...")
    symbolic_r2s = []
    # We need to transform z_states to the same feature space used by the distiller
    X_poly = distiller.transformer.transform(z_states)
    X_norm = distiller.transformer.normalize_x(X_poly)
    Y_norm_true = distiller.transformer.normalize_y(dz_states)
    
    for i, eq in enumerate(equations):
        if eq is not None:
            if hasattr(eq, 'compute_derivatives'):
                # Hamiltonian case: compute_derivatives returns full dz/dt
                dz_pred = []
                for j in range(len(z_states)):
                    dz_pred.append(eq.compute_derivatives(z_states[j], distiller.transformer))
                y_pred_full = np.array(dz_pred)
                # Compute R2 for each dimension and average
                r2s = []
                for d in range(dz_states.shape[1]):
                    y_true_d = dz_states[:, d]
                    y_pred_d = y_pred_full[:, d]
                    var_true = np.var(y_true_d)
                    
                    if var_true < 1e-4:
                        # For low variance, use Relative MSE as a proxy for quality
                        # Score = 1 - (MSE / (mean_magnitude^2 + eps))
                        mse = np.mean((y_true_d - y_pred_d)**2)
                        mean_mag = np.mean(np.abs(y_true_d))
                        r2 = 1.0 - mse / (mean_mag**2 + 1e-6)
                    else:
                        r2 = 1 - np.sum((y_true_d - y_pred_d)**2) / (len(y_true_d) * var_true + 1e-9)
                    r2s.append(r2)
                r2 = np.mean(r2s)
                symbolic_r2s.append(r2)
                print(f"  Hamiltonian Symbolic R2 (mean across dims): {r2:.4f}")
                print(f"  Sample dz_true (first 5): {dz_states[:5, 0]}")
                print(f"  Sample dz_pred (first 5): {y_pred_full[:5, 0]}")
            else:
                y_pred = eq.execute(X_norm)
                y_true = Y_norm_true[:, i]
                r2 = 1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - np.mean(y_true))**2) + 1e-9)
                symbolic_r2s.append(r2)
                print(f"  Equation {i} R2: {r2:.4f}")
        else:
            symbolic_r2s.append(0.0)

    # 5. Symbolic Parsimony Score
    def get_complexity(expr):
        if hasattr(expr, 'length_'): return expr.length_
        return len(str(expr))
        
    parsimony_scores = []
    for i, eq in enumerate(equations):
        if eq is not None:
            # Use max(0, r2) for parsimony calculation to avoid negative scores
            score = get_complexity(eq) / (max(0.01, symbolic_r2s[i]) + 1e-6)
            parsimony_scores.append(score)
            print(f"  Equation {i} Parsimony Score: {score:.4f}")

    # Output Health Check JSON (simulated with print for now)
    health_check = {
        "Symbolic Parsimony": float(np.mean(parsimony_scores)) if parsimony_scores else 0.0,
        "Latent Correlation": float(np.mean([np.max(corrs[k]) for k in range(args.super_nodes)])),
        "Flicker Rate": float(flicker_rate),
        "Symbolic R2": float(np.mean(symbolic_r2s)) if symbolic_r2s else 0.0
    }
    print("\n--- Final Health Check ---")
    for k, v in health_check.items():
        print(f"{k}: {v:.4f}")

    # Create symbolic transformer and update trainer with symbolic proxy
    if hasattr(distiller, 'transformer'):
        symbolic_transformer = distiller.transformer
    else:
        from symbolic import FeatureTransformer
        symbolic_transformer = FeatureTransformer(args.super_nodes, args.latent_dim)
        symbolic_transformer.fit(z_states, dz_states)

    # Update trainer with symbolic proxy
    confidence = np.mean(distiller.confidences) if hasattr(distiller, 'confidences') else 0.8
    trainer.update_symbolic_proxy(equations, symbolic_transformer, weight=args.sym_weight, confidence=confidence)

    # 5.5 STAGE 3: Neural-Symbolic Consistency Training (Closed-Loop)
    if trainer.symbolic_proxy is not None and confidence > 0.1:
        print("\n--- Starting Stage 3: Neural-Symbolic Consistency Training ---")
        stage3_epochs = 50
        trainer.model.train()
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

    # 7. Save Results for Validation
    print("\nSaving results...")
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(results_dir, f"model_{timestamp}.pt")
    torch.save(model.state_dict(), model_save_path)

    config_data = {
        'particles': args.particles,
        'super_nodes': args.super_nodes,
        'latent_dim': args.latent_dim,
        'hidden_dim': args.hidden_dim,
        'hamiltonian': args.hamiltonian,
        'min_active': args.min_active,
        'sim': args.sim,
        'dt': sim.dt
    }

    # NEW: Shadow Integration for Stability Score
    print("\nCalculating Stability Score (1000-step shadow integration)...")
    stability_score = 0.0
    if trainer.symbolic_proxy is not None:
        try:
            # Integrate 1000 steps
            initial_z = torch.tensor(z_states[0].reshape(1, -1), dtype=torch.float32, device=device)
            steps_1000 = 1000
            dt = sim.dt
            from visualization import generate_closed_loop_trajectory
            shadow_traj = generate_closed_loop_trajectory(trainer.symbolic_proxy, initial_z, steps_1000, dt, device=device)
            # Check for NaNs and finite values
            # shadow_traj might be a numpy array if generate_closed_loop_trajectory handles conversion
            if isinstance(shadow_traj, torch.Tensor):
                is_fin = torch.isfinite(shadow_traj).all()
                shadow_np = shadow_traj.cpu().numpy()
            else:
                is_fin = np.isfinite(shadow_traj).all()
                shadow_np = shadow_traj
                
            if is_fin:
                # Score based on whether it exploded (drift from mean)
                drift = np.std(np.linalg.norm(shadow_np, axis=1))
                stability_score = 1.0 / (1.0 + drift)
            else:
                stability_score = 0.0
            print(f"  Stability Score: {stability_score:.4f}")
            
            # NEW: Calculate physical drifts
            from check_conservation import calculate_symplectic_drift, calculate_energy_drift
            print("  Calculating Symplectic and Energy Drifts...")
            
            # Move symbolic proxy to CPU for drift calculation to ensure stability and consistency
            trainer.symbolic_proxy.to('cpu')
            initial_z_cpu = initial_z.to('cpu')
            
            symp_drift = calculate_symplectic_drift(trainer.symbolic_proxy, initial_z_cpu, dt, steps=100)
            
            # Ground truth energy function (requires decoding)
            def gt_energy_func(z_in):
                with torch.no_grad():
                    # Move to CPU for consistent numpy integration
                    z_in_cpu = z_in.to('cpu')
                    z_reshaped = z_in_cpu.view(-1, config_data['super_nodes'], config_data['latent_dim'])
                    
                    # Decoder requires s, batch, and stats
                    model_device = next(trainer.model.parameters()).device
                    
                    # For a clean drift calculation, we assume 1-to-1 mapping if possible
                    # or just use the first particle
                    n_particles_eval = config_data['particles']
                    n_super_eval = config_data['super_nodes']
                    
                    # Create a dummy batch and hard assignment
                    # For 4 particles and 2 super-nodes, we assign 2 particles per super-node
                    eval_batch = torch.zeros(n_particles_eval, dtype=torch.long, device=model_device)
                    eval_s = torch.zeros(n_particles_eval, n_super_eval, device=model_device)
                    for i in range(n_particles_eval):
                        eval_s[i, i % n_super_eval] = 1.0
                    
                    # Move inputs to model device
                    z_in_dev = z_reshaped.to(model_device)
                    
                    # Use the model's stats for denormalization
                    model_stats = trainer.stats
                    
                    recon = trainer.model.decoder(z_in_dev, eval_s, eval_batch, stats=model_stats)
                    pos_pred = recon[:, :2]
                    vel_pred = recon[:, 2:]
                    
                    p_np = pos_pred.cpu().numpy()
                    v_np = vel_pred.cpu().numpy()
                    return torch.tensor(sim.energy(p_np, v_np))
            
            eng_drift = calculate_energy_drift(trainer.symbolic_proxy, gt_energy_func, initial_z_cpu, dt, steps=100)
            
            # Ensure they are scalar floats
            if hasattr(symp_drift, 'item'): symp_drift = symp_drift.item()
            if hasattr(eng_drift, 'item'): eng_drift = eng_drift.item()
            
            print(f"  Symplectic Drift: {symp_drift:.6f}")
            print(f"  Energy Drift: {eng_drift:.6f}")
            
        except Exception as e:
            print(f"  Drift calculation failed: {e}")
            stability_score = 0.0
            symp_drift = 1.0
            eng_drift = 1.0

    # NEW: discovery_report.json as requested
    report_data = {
        'discovered_sympy': [str(eq) for eq in equations],
        'recovered_constants': getattr(distiller, 'recovered_constants', {}),
        'stability_score': float(stability_score),
        'symplectic_drift': float(symp_drift),
        'energy_drift': float(eng_drift),
        'health_check': {k: float(v) if hasattr(v, 'item') else v for k, v in health_check.items()},
        'config': config_data
    }
    
    with open('discovery_report.json', 'w') as f:
        json.dump(report_data, f, indent=4)
    print("Final report saved to discovery_report.json")
    
    discovery_data = {
        'config': config_data,
        'equations': [str(eq) for eq in equations],
        'health_check': health_check,
        'model_path': model_save_path
    }
    
    results_json_path = os.path.join(results_dir, f"discovery_{timestamp}.json")
    with open(results_json_path, 'w') as f:
        json.dump(discovery_data, f, indent=4)
    
    print(f"Results saved to {results_json_path}")
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()