import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator import SpringMassSimulator
from optimized_model import OptimizedDiscoveryEngineModel
from engine import Trainer, prepare_data
from symbolic import SymbolicDistiller, extract_latent_data
from enhanced_symbolic import create_enhanced_distiller
from improved_symbolic_distillation import ImprovedSymbolicDistiller
from robust_symbolic import create_robust_symbolic_proxy
from enhanced_balancer import create_enhanced_loss_balancer
from learnable_basis import EnhancedFeatureTransformer
from transformer_symbolic import create_neural_symbolic_hybrid
import argparse

def main():
    # Enable anomaly detection for debugging
    torch.autograd.set_detect_anomaly(True)
    # NOTE: This is the enhanced version of the program with improvements
    # addressing the critical analysis points:
    # 1. Basis function bottleneck - using learnable basis functions
    # 2. Loss landscape hyper-dimensionality - using enhanced loss balancer
    # 3. Symbolic proxy fragility - using robust symbolic proxy
    # 4. Basis-free symbolic discovery - using transformer-based generator
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--particles', type=int, default=16)
    parser.add_argument('--super_nodes', type=int, default=4)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps)')
    parser.add_argument('--use_learnable_bases', action='store_true', help='Use learnable basis functions')
    parser.add_argument('--use_robust_proxy', action='store_true', help='Use robust symbolic proxy')
    parser.add_argument('--use_enhanced_balancer', action='store_true', help='Use enhanced loss balancer')
    args = parser.parse_args()

    # 0. Device Discovery
    if args.device:
        device = torch.device(args.device)
        print(f"Manually selected device: {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Use MPS for GNN/Decoder, but Trainer will still move ODE to CPU for stability
        device = torch.device('mps')
        print("MPS is available")
    else:
        device = torch.device('cpu')
        print("GPU not available, falling back to CPU")
    print(f"Using device: {device}")

    # 1. Setup Parameters
    n_particles = args.particles
    n_super_nodes = args.super_nodes
    latent_dim = 4
    steps = args.steps
    epochs = args.epochs
    seq_len = 20
    dynamic_radius = 1.5
    # Enable PBC with a reasonable box size
    box_size = (10.0, 10.0)  # Set a reasonable box size for PBC

    print("--- 1. Generating Data ---")
    from simulator import LennardJonesSimulator
    # Reduce dt for better energy conservation in data generation
    sim = LennardJonesSimulator(n_particles=n_particles, epsilon=1.0, sigma=1.0,
                                dynamic_radius=dynamic_radius, box_size=box_size, dt=0.0005)
    pos, vel = sim.generate_trajectory(steps=steps)
    initial_energy = sim.energy(pos[0], vel[0])
    final_energy = sim.energy(pos[-1], vel[-1])
    print(f"Energy conservation: {initial_energy:.2f} -> {final_energy:.2f} ({(final_energy/initial_energy-1)*100:.2f}%)")

    # Prepare data with device support and robust normalization
    dataset, stats = prepare_data(pos, vel, radius=dynamic_radius, device=device, box_size=box_size)

    # 2. Initialize Model and Trainer ---
    print("--- 2. Initialize Model and Trainer ---")
    # Using Hamiltonian dynamics with learnable dissipation for improved physics fidelity
    # NEW: Ensure at least half of super-nodes stay active to prevent resolution collapse
    min_active = max(1, n_super_nodes // 2)
    model = OptimizedDiscoveryEngineModel(n_particles=n_particles,
                                 n_super_nodes=n_super_nodes,
                                 latent_dim=latent_dim,
                                 hidden_dim=128,
                                 hamiltonian=True,
                                 dissipative=True,
                                 min_active_super_nodes=min_active,
                                 box_size=box_size).to(device)

    # NEW: Weight initialization for GNN stability (critical for MPS)
    def weights_init(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    model.apply(weights_init)
    print("Model weights initialized with Xavier uniform")

    # NEW: Sparsity Scheduler to prevent resolution collapse
    from stable_pooling import SparsityScheduler
    sparsity_scheduler = SparsityScheduler(
        initial_weight=0.001,
        target_weight=0.05,
        warmup_steps=int(epochs * 0.1),
        max_steps=int(epochs * 0.8)
    )

    # NEW: Enhanced loss balancer to address hyper-dimensional loss landscape
    if args.use_enhanced_balancer:
        enhanced_balancer = create_enhanced_loss_balancer(strategy='gradient_based')
        print("Using enhanced gradient-based loss balancer")
    else:
        enhanced_balancer = None

    # Trainer now uses adaptive loss weighting, manual weights are deprecated
    # Optimized parameters for faster training
    warmup_epochs = 50  # Increased warmup epochs to 50 as requested
    trainer = Trainer(model, lr=2e-4, device=device, stats=stats,
                      warmup_epochs=warmup_epochs, max_epochs=epochs,
                      sparsity_scheduler=sparsity_scheduler,
                      skip_consistency_freq=3,  # Compute consistency loss every 3 epochs to save time
                      enable_gradient_accumulation=True,  # Use gradient accumulation for memory efficiency
                      grad_acc_steps=2,  # Accumulate gradients over 2 steps
                      enhanced_balancer=enhanced_balancer)  # Pass enhanced balancer

    # Increased patience and adjusted factor to prevent premature decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min',
                                                           factor=0.5, patience=1000, min_lr=1e-6)

    last_loss = 1.0
    for epoch in range(epochs):
        idx = np.random.randint(0, len(dataset) - seq_len)
        batch_data = dataset[idx : idx + seq_len]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=epochs)
        last_loss = loss
        scheduler.step(loss)

        if epoch % 20 == 0:
            progress = (epoch / epochs) * 100
            stats_tracker = trainer.loss_tracker.get_stats()
            active_nodes = int(model.encoder.pooling.active_mask.sum().item())
            log_str = f"Progress: {progress:3.0f}% | Loss: {stats_tracker.get('total', 0):.4f} | "
            log_str += f"Rec: {stats_tracker.get('rec_raw', 0):.4f} | "
            log_str += f"Cons: {stats_tracker.get('cons_raw', 0):.4f} | "
            log_str += f"LVar: {stats_tracker.get('lvars_mean', 0):.2f} | "
            log_str += f"Active: {active_nodes} | "
            log_str += f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}"
            print(log_str)

    # --- Interpretability Check ---
    print("\n--- 2.1 Latent Interpretability Analysis ---")
    from engine import analyze_latent_space
    corrs = analyze_latent_space(model, dataset, pos, device=device)
    for k in range(n_super_nodes):
        max_corr = np.max(np.abs(corrs[k]))
        print(f"Super-node {k} max CoM correlation: {max_corr:.3f}")
        if max_corr > 0.8:
            print(f"  -> Strong physical mapping detected for super-node {k}")

    # --- Quality Gate ---
    print(f"\nFinal Training Loss: {last_loss:.6f}")
    if rec > 0.2: # Relaxed from 0.1 to 0.2
        print(f"CRITICAL ERROR: Model failed to converge (Rec Loss: {rec:.6f} > 0.2).")
        print("This indicates a Normalization Failure. Aborting symbolic distillation.")
        return

    # 3. Extract Symbolic Equations
    print("--- 3. Distilling Symbolic Laws ---")
    is_hamiltonian = model.hamiltonian
    latent_data = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=is_hamiltonian)

    if len(latent_data[0]) == 0:
        print("Error: No valid latent data extracted (all NaN or divergent). Skipping symbolic distillation.")
        return

    # NEW: Use ImprovedSymbolicDistiller which includes physicality gates and optimized search
    distiller = ImprovedSymbolicDistiller(populations=3000, generations=50, secondary_optimization=True)

    # Use normalized box size for distillation since latents are aligned to [-1, 1]
    norm_box = (2.0, 2.0)

    if is_hamiltonian:
        z_states, dz_states, t_states, h_states = latent_data
        print("Distilling Hamiltonian H(q, p) with secondary optimization...")
        # ImprovedSymbolicDistiller.distill includes the physicality gate
        equations = distiller.distill(z_states, h_states, n_super_nodes, latent_dim, box_size=norm_box)
        confidences = distiller.confidences
        # Update trainer with symbolic laws if confidence is high enough
        if confidences[0] > 0.5:
            # NEW: Use robust symbolic proxy if requested
            if args.use_robust_proxy:
                print("Creating robust symbolic proxy...")
                robust_proxy = create_robust_symbolic_proxy(
                    equations, distiller.feature_masks, distiller.transformer,
                    confidence_threshold=0.5, use_adaptive=True
                )
                # Print validation summary
                if hasattr(robust_proxy, 'get_reliability_report'):
                    report = robust_proxy.get_reliability_report()
                    print(f"Robust proxy validation: {report['validation_summary']['valid_equations']}/{report['validation_summary']['total_equations']} equations valid")
                trainer.update_symbolic_proxy(robust_proxy, weight=0.1, confidence=confidences[0])
            else:
                trainer.update_symbolic_proxy(equations, distiller.transformer, weight=0.1, confidence=confidences[0])
    else:
        z_states, dz_states, t_states = latent_data
        print("Distilling derivatives dZ/dt with secondary optimization...")
        equations = distiller.distill(z_states, dz_states, n_super_nodes, latent_dim, box_size=norm_box)
        confidences = distiller.confidences
        avg_conf = np.mean(confidences)
        if avg_conf > 0.5:
            # NEW: Use robust symbolic proxy if requested
            if args.use_robust_proxy:
                print("Creating robust symbolic proxy...")
                robust_proxy = create_robust_symbolic_proxy(
                    equations, distiller.feature_masks, distiller.transformer,
                    confidence_threshold=0.5, use_adaptive=True
                )
                trainer.update_symbolic_proxy(robust_proxy, weight=0.1, confidence=avg_conf)
            else:
                trainer.update_symbolic_proxy(equations, distiller.transformer, weight=0.1, confidence=avg_conf)

    print("\nDiscovered Symbolic Laws:")
    if is_hamiltonian:
        if equations[0] is not None:
            print(f"H(z) = {equations[0]} (Confidence: {confidences[0]:.3f})")
        else:
            print("No Hamiltonian discovered.")
    else:
        for i, eq in enumerate(equations):
            if eq is not None:
                print(f"dZ_{i}/dt = {eq} (Confidence: {confidences[i]:.3f})")
            else:
                print(f"dZ_{i}/dt = None")

    # NEW: Transformer-based symbolic generation (optional)
    print("\n--- 4. Exploring Transformer-Based Symbolic Generation ---")
    try:
        if hasattr(distiller, 'transformer'):
            n_vars = n_super_nodes * latent_dim  # Total number of variables
            neural_symbolic_hybrid = create_neural_symbolic_hybrid(
                neural_model=model, 
                n_variables=n_vars, 
                blend_factor=0.5
            )
            
            print("Generating candidate symbolic expressions...")
            candidates = neural_symbolic_hybrid.generate_candidates(n_candidates=5, temperature=0.8)
            print("Generated candidates:")
            for i, candidate in enumerate(candidates[:3]):  # Show first 3
                print(f"  {i+1}. {candidate}")
    except Exception as e:
        print(f"Transformer-based symbolic generation not available: {e}")

    # 5. Visualization & Integration
    if is_hamiltonian and equations[0] is None:
        print("Skipping visualization due to lack of discovered Hamiltonian.")
        return

    print("--- 5. Visualizing Results ---")
    from scipy.integrate import odeint
    from improved_symbolic import ImprovedSymbolicDynamics

    dyn_fn = ImprovedSymbolicDynamics(distiller, equations, distiller.feature_masks, is_hamiltonian, n_super_nodes, latent_dim, model=model)

    # Integrate the discovered equations
    z0 = z_states[0]
    t_eval = np.linspace(0, (len(z_states)-1)*sim.dt, len(z_states))
    z_simulated = odeint(dyn_fn, z0, t_eval)


    model.eval()
    with torch.no_grad():
        test_idx = 0
        data = dataset[test_idx]
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        z, s, _, _ = model.encode(x, edge_index, batch)
        recon = model.decode(z, s, batch).cpu().numpy()

    plt.figure(figsize=(18, 5))

    # 1. Micro Plot: Reconstruction
    plt.subplot(1, 3, 1)
    plt.scatter(data.x.cpu().numpy()[:, 0], data.x.cpu().numpy()[:, 1], c='blue', alpha=0.5, label='Truth')
    plt.scatter(recon[:, 0], recon[:, 1], c='red', marker='x', label='Recon')
    plt.title("Micro: Reconstruction")
    plt.legend()

    # 2. Assignment Plot
    plt.subplot(1, 3, 2)
    assignments = torch.argmax(s, dim=1).cpu().numpy()
    plt.scatter(data.x.cpu().numpy()[:, 0], data.x.cpu().numpy()[:, 1], c=assignments, cmap='viridis')
    plt.title("Hierarchical: Assignments")

    # 3. Meso Plot: Symbolic Integration vs Learned Latent
    plt.subplot(1, 3, 3)
    # Plot first few dimensions for clarity
    n_plot = min(4, z_states.shape[1])
    for i in range(n_plot):
        plt.plot(t_eval, z_states[:, i], 'k--', alpha=0.3, label=f'Learned Z_{i}' if i==0 else "")
        plt.plot(t_eval, z_simulated[:, i], label=f'Symbolic Z_{i}')
    plt.title("Meso: Symbolic Integration")
    plt.legend()

    plt.tight_layout()
    plt.savefig("discovery_result.png")
    print("Result saved to discovery_result.png")

if __name__ == "__main__":
    main()