import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data
from symbolic import SymbolicDistiller, extract_latent_data

def main():
    # 0. Device Discovery
    # Check for CUDA first, then MPS (for Apple Silicon), then fall back to CPU
    if torch.cuda.is_available():
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
    n_particles = 16
    n_super_nodes = 4
    latent_dim = 4 
    steps = 800
    epochs = 5000
    seq_len = 20
    dynamic_radius = 1.5 
    box_size = None # Disable PBC for stability
    
    print("--- 1. Generating Data ---")
    sim = SpringMassSimulator(n_particles=n_particles, k=15.0, 
                             dynamic_radius=dynamic_radius, box_size=box_size)
    pos, vel = sim.generate_trajectory(steps=steps)
    
    # Prepare data with device support and robust normalization
    dataset, stats = prepare_data(pos, vel, radius=dynamic_radius, device=device)
    
    # 2. Initialize Model and Trainer
    print("--- 2. Training Discovery Engine ---")
    # Using Hamiltonian dynamics for improved physics fidelity
    model = DiscoveryEngineModel(n_particles=n_particles, 
                                 n_super_nodes=n_super_nodes, 
                                 latent_dim=latent_dim,
                                 hidden_dim=128,
                                 hamiltonian=True).to(device)
    
    # Trainer now uses adaptive loss weighting, manual weights are deprecated
    trainer = Trainer(model, lr=5e-4, device=device, stats=stats)
    
    # Increased patience and adjusted factor to prevent premature decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min', 
                                                           factor=0.8, patience=800, min_lr=1e-6)
    
    last_loss = 1.0
    for epoch in range(epochs):
        idx = np.random.randint(0, len(dataset) - seq_len)
        batch_data = dataset[idx : idx + seq_len]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch)
        last_loss = loss
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            progress = (epoch / epochs) * 100
            stats = trainer.loss_tracker.get_stats()
            log_str = f"Progress: {progress:3.0f}% | Loss: {loss:.6f} | "
            log_str += f"Rec: {stats.get('rec_raw', 0):.4f} | Cons: {stats.get('cons_raw', 0):.4f} | "
            log_str += f"Assign: {stats.get('assign_raw', 0):.4f} | Ortho: {stats.get('ortho_raw', 0):.4f} | "
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
    z_states, dz_states, t_states = extract_latent_data(model, dataset, sim.dt, stats=stats)
    
    distiller = SymbolicDistiller(populations=2000, generations=50) 
    equations, z_stats = distiller.distill(z_states, dz_states)
    z_mean, z_std, dz_mean, dz_std = z_stats
    
    print("\nDiscovered Meso-scale Laws (dZ/dt = ...):")
    for i, eq in enumerate(equations):
        print(f"dZ_{i}/dt = {eq}")

    # 4. Visualization & Integration
    print("--- 4. Visualizing Results ---")
    from scipy.integrate import odeint
    
    def symbolic_dynamics(z, t):
        # z: [18] (16 latent vars + 2 physical vars)
        z_latent = z[:16].reshape(n_super_nodes, latent_dim)
        dists = []
        for i in range(n_super_nodes):
            for j in range(i + 1, n_super_nodes):
                dists.append(np.linalg.norm(z_latent[i] - z_latent[j]))
        
        z_full = np.concatenate([z[:16], np.array(dists), z[16:]])
        
        # Normalize input
        z_norm = (z_full - z_mean) / z_std
        X = z_norm.reshape(1, -1)
        
        # Feature Engineering for integration
        n_features = X.shape[1]
        features = [X]
        for i in range(n_features):
            features.append((X[:, i]**2).reshape(-1, 1))
            for j in range(i + 1, n_features):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        X_poly = np.hstack(features)
        
        # Execute symbolic equations with feature masks
        dzdt_norm = []
        for i, (eq, mask) in enumerate(zip(equations, distiller.feature_masks)):
            X_selected = X_poly[:, mask]
            dzdt_norm.append(eq.execute(X_selected)[0])
        
        dzdt_norm = np.array(dzdt_norm)
        
        # Denormalize output
        dzdt = dzdt_norm * dz_std + dz_mean
        return dzdt

    # Integrate the discovered equations (only independent variables)
    # z_states: [16 latent, 6 dists, 2 physical] -> extract 16+2
    z0 = np.concatenate([z_states[0, :16], z_states[0, 22:]])
    t_eval = np.linspace(0, (len(z_states)-1)*sim.dt, len(z_states))
    z_simulated = odeint(symbolic_dynamics, z0, t_eval)

    model.eval()
    with torch.no_grad():
        test_idx = 0
        data = dataset[test_idx]
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        z, s, _ = model.encode(x, edge_index, batch)
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
    z_states_independent = np.concatenate([z_states[:, :16], z_states[:, 22:]], axis=1)
    for i in range(z_states_independent.shape[1]):
        if i < 4: # Plot first 4 for clarity
            plt.plot(t_eval, z_states_independent[:, i], 'k--', alpha=0.3, label=f'Learned Z_{i}' if i==0 else "")
            plt.plot(t_eval, z_simulated[:, i], label=f'Symbolic Z_{i}')
    plt.title("Meso: Symbolic Integration")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("discovery_result.png")
    print("Result saved to discovery_result.png")

if __name__ == "__main__":
    main()
