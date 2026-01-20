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
    n_super_nodes = 2
    latent_dim = 2 
    steps = 800
    epochs = 5000
    seq_len = 12
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
    model = DiscoveryEngineModel(n_particles=n_particles, 
                                 n_super_nodes=n_super_nodes, 
                                 latent_dim=latent_dim,
                                 hidden_dim=128).to(device)
    
    # Adjusted weights: Significantly more emphasis on reconstruction and consistency
    loss_weights = {
        'rec': 100.0,     
        'cons': 50.0,    
        'assign': 10.0,   
        'latent_l2': 0.1 
    }
    
    trainer = Trainer(model, lr=8e-4, device=device, 
                      loss_weights=loss_weights, stats=stats)
    
    # Increased patience and adjusted factor
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min', 
                                                           factor=0.5, patience=500, min_lr=1e-6)
    
    last_loss = 1.0
    for epoch in range(epochs):
        idx = np.random.randint(0, len(dataset) - seq_len)
        batch_data = dataset[idx : idx + seq_len]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch)
        last_loss = loss
        scheduler.step(loss)
        
        if epoch % 100 == 0:
            progress = (epoch / epochs) * 100
            print(f"Progress: {progress:3.0f}% | Loss: {loss:.6f} | Rec: {rec:.6f} | Cons: {cons:.6f} | LR: {trainer.optimizer.param_groups[0]['lr']:.2e}")

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
        # Normalize input
        z_norm = (z - z_mean) / z_std
        z_input = z_norm.reshape(1, -1)
        
        # Execute symbolic equations
        dzdt_norm = np.array([eq.execute(z_input)[0] for eq in equations])
        
        # Denormalize output
        dzdt = dzdt_norm * dz_std + dz_mean
        return dzdt

    # Integrate the discovered equations
    z0 = z_states[0]
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
    for i in range(z_states.shape[1]):
        plt.plot(t_eval, z_states[:, i], 'k--', alpha=0.3, label=f'Learned Z_{i}' if i==0 else "")
        plt.plot(t_eval, z_simulated[:, i], label=f'Symbolic Z_{i}')
    plt.title("Meso: Symbolic Integration")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("discovery_result.png")
    print("Result saved to discovery_result.png")

if __name__ == "__main__":
    main()
