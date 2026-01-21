"""
Improved Optimized Fast Training Script

This script implements performance optimizations to speed up training while maintaining
the quality of the neural-symbolic discovery pipeline.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator import SpringMassSimulator
from optimized_model import OptimizedDiscoveryEngineModel
from engine import Trainer, prepare_data
from symbolic import extract_latent_data
import argparse

class ImprovedEarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)  # Increased epochs for better convergence
    parser.add_argument('--steps', type=int, default=100)   # Increased steps for better data
    parser.add_argument('--particles', type=int, default=8) # Increased particles for richer dynamics
    parser.add_argument('--super_nodes', type=int, default=4)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps)')
    args = parser.parse_args()

    # 0. Device Discovery
    if args.device:
        device = torch.device(args.device)
        print(f"Manually selected device: {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS is available")
    else:
        device = torch.device('cpu')
        print("GPU not available, falling back to CPU")
    print(f"Using device: {device}")

    # 1. Setup Parameters
    n_particles = args.particles
    n_super_nodes = args.super_nodes
    latent_dim = 8  # Increased latent dimension for better representation
    steps = args.steps
    epochs = args.epochs
    seq_len = 5  # Increased sequence length for better temporal modeling
    dynamic_radius = 2.0  # Increased radius for better connectivity
    box_size = (15.0, 15.0)  # Larger box size

    print("--- 1. Generating Data ---")
    from simulator import LennardJonesSimulator
    # Use smaller dt for better simulation accuracy
    sim = LennardJonesSimulator(n_particles=n_particles, epsilon=1.0, sigma=1.0,
                                dynamic_radius=dynamic_radius, box_size=box_size, dt=0.005)  # Smaller dt
    pos, vel = sim.generate_trajectory(steps=steps)
    
    # Track energy conservation throughout the trajectory
    energies = [sim.energy(pos[i], vel[i]) for i in range(len(pos))]
    initial_energy = energies[0]
    final_energy = energies[-1]
    print(f"Initial energy: {initial_energy:.4f}")
    print(f"Final energy: {final_energy:.4f}")
    print(f"Energy conservation: {initial_energy:.4f} -> {final_energy:.4f} ({(abs(final_energy-initial_energy)/abs(initial_energy))*100:.2f}%)")

    # Prepare data with device support and robust normalization
    dataset, stats = prepare_data(pos, vel, radius=dynamic_radius, device=device, cache_edges=True)

    # 2. Initialize Model and Trainer
    print("--- 2. Initialize Improved Optimized Model and Trainer ---")
    # Using Hamiltonian dynamics with learnable dissipation for improved physics fidelity
    # Ensure at least half of super-nodes stay active to prevent resolution collapse
    min_active = max(1, n_super_nodes // 2)

    # Use the optimized model with reduced hidden dimensions
    model = OptimizedDiscoveryEngineModel(n_particles=n_particles,
                                          n_super_nodes=n_super_nodes,
                                          latent_dim=latent_dim,
                                          hidden_dim=32,  # Increased hidden dim for better capacity
                                          hamiltonian=True,
                                          dissipative=True,
                                          min_active_super_nodes=min_active).to(device)

    # NEW: Sparsity Scheduler to prevent resolution collapse
    from stable_pooling import SparsityScheduler
    sparsity_scheduler = SparsityScheduler(
        initial_weight=0.01,
        target_weight=0.1,  # Increased target weight for better regularization
        warmup_steps=max(10, int(epochs * 0.1)),
        max_steps=max(50, int(epochs * 0.4))  # Increased max steps
    )

    # Optimized trainer parameters for better training
    trainer = Trainer(model, lr=5e-4,  # Lower learning rate for stability
                      device=device, stats=stats, sparsity_scheduler=sparsity_scheduler,
                      skip_consistency_freq=2,  # Compute consistency loss more frequently
                      enable_gradient_accumulation=True,  # Use gradient accumulation for memory efficiency
                      grad_acc_steps=4)  # Accumulate gradients over more steps

    # Better scheduler parameters for improved convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min',
                                                           factor=0.5, patience=30, min_lr=1e-7)  # Slower decay

    # Early stopping setup with better parameters
    early_stopping = ImprovedEarlyStopping(patience=50, min_delta=1e-4)  # Better patience and delta

    last_loss = 1.0
    for epoch in range(epochs):
        idx = np.random.randint(0, len(dataset) - seq_len)
        batch_data = dataset[idx : idx + seq_len]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=epochs)
        last_loss = loss
        scheduler.step(loss)

        if epoch % 50 == 0:  # Adjusted logging frequency for better monitoring
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

        # Check for early stopping
        if early_stopping(last_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

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
    if rec > 0.05: # Stricter convergence check
        print(f"WARNING: Model may not have converged fully (Rec Loss: {rec:.6f}).")

    # 3. Extract Symbolic Equations with improved parameters
    print("--- 3. Distilling Symbolic Laws ---")
    is_hamiltonian = model.hamiltonian

    # Extract latent data with reduced computation
    latent_data = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=is_hamiltonian)

    if len(latent_data[0]) == 0:
        print("Error: No valid latent data extracted (all NaN or divergent). Skipping symbolic distillation.")
        return

    # Use improved symbolic distiller with better parameters
    from optimized_symbolic_distillation import OptimizedSymbolicDistiller
    distiller = OptimizedSymbolicDistiller(
        populations=1200,  # Increased from 1000
        generations=30,    # Increased from 20
        stopping_criteria=0.005,  # Tighter from 0.01
        max_features=12,    # Increased from 8
        secondary_optimization=True,
        opt_method='L-BFGS-B',
        opt_iterations=100,  # Increased from 50
        use_sindy_pruning=True,
        sindy_threshold=0.05  # Decreased from 0.1 for more features
    )

    if is_hamiltonian:
        z_states, dz_states, t_states, h_states = latent_data
        print("Distilling Hamiltonian H(q, p) with secondary optimization...")
        equations = distiller.distill(z_states, h_states, n_super_nodes, latent_dim, box_size=box_size)
        confidences = distiller.confidences
        # Update trainer with symbolic laws if confidence is high enough
        if confidences[0] > 0.5:  # Higher threshold for better quality
            trainer.update_symbolic_proxy(equations, distiller.transformer, weight=0.1, confidence=confidences[0])
    else:
        z_states, dz_states, t_states = latent_data
        print("Distilling derivatives dZ/dt with secondary optimization...")
        equations = distiller.distill(z_states, dz_states, n_super_nodes, latent_dim, box_size=box_size)
        confidences = distiller.confidences
        avg_conf = np.mean(confidences)
        if avg_conf > 0.5:  # Higher threshold for better quality
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

    # 4. Visualization & Integration
    if is_hamiltonian and equations[0] is None:
        print("Skipping visualization due to lack of discovered Hamiltonian.")
        return

    print("--- 4. Visualizing Results ---")
    from scipy.integrate import odeint
    from optimized_symbolic import OptimizedSymbolicDynamics

    dyn_fn = OptimizedSymbolicDynamics(distiller, equations, distiller.feature_masks, is_hamiltonian, n_super_nodes, latent_dim)

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
    plt.savefig("improved_optimized_discovery_result.png")
    print("Result saved to improved_optimized_discovery_result.png")

if __name__ == "__main__":
    main()