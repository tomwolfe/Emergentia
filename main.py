import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data
from symbolic import SymbolicDistiller, extract_latent_data
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--particles', type=int, default=16)
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
        print("MPS (Apple Silicon) is available")
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
    box_size = None # Disable PBC for stability
    
    print("--- 1. Generating Data ---")
    from simulator import LennardJonesSimulator
    sim = LennardJonesSimulator(n_particles=n_particles, epsilon=1.0, sigma=1.0, 
                                dynamic_radius=dynamic_radius, box_size=box_size)
    pos, vel = sim.generate_trajectory(steps=steps)
    initial_energy = sim.energy(pos[0], vel[0])
    final_energy = sim.energy(pos[-1], vel[-1])
    print(f"Energy conservation: {initial_energy:.2f} -> {final_energy:.2f} ({(final_energy/initial_energy-1)*100:.2f}%)")
    
    # Prepare data with device support and robust normalization
    dataset, stats = prepare_data(pos, vel, radius=dynamic_radius, device=device)
    
    # 2. Initialize Model and Trainer
    print("--- 2. Training Discovery Engine ---")
    # Using Hamiltonian dynamics with learnable dissipation for improved physics fidelity
    model = DiscoveryEngineModel(n_particles=n_particles, 
                                 n_super_nodes=n_super_nodes, 
                                 latent_dim=latent_dim,
                                 hidden_dim=128,
                                 hamiltonian=True,
                                 dissipative=True).to(device)
    
    # Trainer now uses adaptive loss weighting, manual weights are deprecated
    trainer = Trainer(model, lr=5e-4, device=device, stats=stats)
    
    # Increased patience and adjusted factor to prevent premature decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min', 
                                                           factor=0.8, patience=800, min_lr=1e-6)
    
    last_loss = 1.0
    for epoch in range(epochs):
        idx = np.random.randint(0, len(dataset) - seq_len)
        batch_data = dataset[idx : idx + seq_len]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=epochs)
        last_loss = loss
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            progress = (epoch / epochs) * 100
            stats_tracker = trainer.loss_tracker.get_stats()
            active_nodes = int(model.encoder.pooling.active_mask.sum().item())
            log_str = f"Progress: {progress:3.0f}% | Loss: {loss:.6f} | "
            log_str += f"Rec: {stats_tracker.get('rec_raw', 0):.4f} | "
            log_str += f"Active Nodes: {active_nodes} | "
            log_str += f"W_Rec: {stats_tracker.get('w_rec', 0):.2e} | "
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
    if rec > 0.1: # Check unweighted reconstruction loss
        print(f"WARNING: Model may not have converged fully (Rec Loss: {rec:.6f}).")

    # 3. Extract Symbolic Equations
    print("--- 3. Distilling Symbolic Laws ---")
    is_hamiltonian = model.hamiltonian
    latent_data = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=is_hamiltonian)
    
    distiller = SymbolicDistiller(populations=2000, generations=50) 
    
    if is_hamiltonian:
        z_states, dz_states, t_states, h_states = latent_data
        print("Distilling Hamiltonian H(q, p)...")
        equations = distiller.distill(z_states, h_states, n_super_nodes, latent_dim)
        confidences = distiller.confidences
    else:
        z_states, dz_states, t_states = latent_data
        print("Distilling derivatives dZ/dt...")
        equations = distiller.distill(z_states, dz_states, n_super_nodes, latent_dim)
        confidences = distiller.confidences
    
    print("\nDiscovered Symbolic Laws:")
    if is_hamiltonian:
        print(f"H(z) = {equations[0]} (Confidence: {confidences[0]:.3f})")
    else:
        for i, eq in enumerate(equations):
            print(f"dZ_{i}/dt = {eq} (Confidence: {confidences[i]:.3f})")

    # 4. Visualization & Integration
    print("--- 4. Visualizing Results ---")
    from scipy.integrate import odeint
    
    class SymbolicDynamics:
        def __init__(self, distiller, equations, feature_masks, is_hamiltonian, n_super_nodes, latent_dim):
            self.distiller = distiller
            self.equations = equations
            self.feature_masks = feature_masks
            self.is_hamiltonian = is_hamiltonian
            self.n_super_nodes = n_super_nodes
            self.latent_dim = latent_dim
            
            # Cache transformer for speed
            self.transformer = distiller.transformer

        def __call__(self, z, t):
            # z: [n_super_nodes * latent_dim]
            if self.is_hamiltonian:
                # Optimized vectorized numerical gradient
                eps = 1e-4
                n_dims = len(z)
                # Create a batch of perturbed inputs: [2 * n_dims, n_dims]
                z_batch = np.tile(z, (2 * n_dims, 1))
                for i in range(n_dims):
                    z_batch[2*i, i] += eps
                    z_batch[2*i+1, i] -= eps
                
                # Transform all at once
                X_poly = self.transformer.transform(z_batch)
                X_norm = self.transformer.normalize_x(X_poly)
                
                # Execute symbolic H on the batch
                h_norm = self.equations[0].execute(X_norm[:, self.feature_masks[0]])
                h_val = self.transformer.denormalize_y(h_norm)
                
                # Compute gradients: (H(z+eps) - H(z-eps)) / 2eps
                grad = (h_val[0::2] - h_val[1::2]) / (2 * eps)
                
                dzdt = np.zeros_like(z)
                d_sub = self.latent_dim // 2
                for k in range(self.n_super_nodes):
                    dq_idx = k * self.latent_dim + np.arange(d_sub)
                    dp_idx = k * self.latent_dim + d_sub + np.arange(d_sub)
                    dzdt[dq_idx] = grad[dp_idx]
                    dzdt[dp_idx] = -grad[dq_idx]
                return dzdt
            else:
                z_reshaped = z.reshape(1, -1)
                X_poly = self.transformer.transform(z_reshaped)
                X_norm = self.transformer.normalize_x(X_poly)
                
                dzdt_norm = []
                for i, (eq, mask) in enumerate(zip(self.equations, self.feature_masks)):
                    X_selected = X_norm[:, mask]
                    dzdt_norm.append(eq.execute(X_selected)[0])
                
                return self.transformer.denormalize_y(np.array(dzdt_norm))

    dyn_fn = SymbolicDynamics(distiller, equations, distiller.feature_masks, is_hamiltonian, n_super_nodes, latent_dim)

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
