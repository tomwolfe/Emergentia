import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data
from symbolic import SymbolicDistiller, extract_latent_data

def main():
    # 1. Setup Parameters
    n_particles = 16
    n_super_nodes = 2
    latent_dim = 2 
    steps = 500
    epochs = 500
    seq_len = 5
    dynamic_radius = 1.5 # Radius for dynamic topology
    
    print("--- 1. Generating Data ---")
    # Initialize simulator with dynamic topology
    sim = SpringMassSimulator(n_particles=n_particles, k=15.0, dynamic_radius=dynamic_radius)
    pos, vel = sim.generate_trajectory(steps=steps)
    # Prepare data with dynamic edge_index
    dataset = prepare_data(pos, vel, radius=dynamic_radius)
    
    # 2. Initialize Model and Trainer
    print("--- 2. Training Discovery Engine ---")
    model = DiscoveryEngineModel(n_particles=n_particles, 
                                 n_super_nodes=n_super_nodes, 
                                 latent_dim=latent_dim)
    trainer = Trainer(model, lr=1e-3)
    
    last_loss = 1.0
    for epoch in range(epochs):
        # Sample a short sequence for training
        idx = np.random.randint(0, len(dataset) - seq_len)
        batch_data = dataset[idx : idx + seq_len]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt)
        last_loss = loss
        
        if epoch % 50 == 0:
            progress = (epoch / epochs) * 100
            print(f"Progress: {progress:3.0f}% | Loss: {loss:.6f} | Rec: {rec:.6f} | Cons: {cons:.6f}")

    # --- Quality Gate ---
    print(f"\nFinal Training Loss: {last_loss:.6f}")
    if last_loss > 0.05: # Threshold for 'reasonable' convergence
        print("WARNING: Model may not have converged fully (Loss > 0.05).")
        print("Distilled equations might be approximate.")

    # 3. Extract Symbolic Equations
    print("--- 3. Distilling Symbolic Laws ---")
    # Extract states and their derivatives from the learned Latent ODE
    z_states, dz_states, t_states = extract_latent_data(model, dataset, sim.dt)
    
    # Use the enhanced distiller with expanded function set
    # Perform autonomous distillation (no time input)
    distiller = SymbolicDistiller(populations=2000, generations=40) 
    equations = distiller.distill(z_states, dz_states)
    
    print("\nDiscovered Meso-scale Laws (dZ/dt = ...):")
    for i, eq in enumerate(equations):
        print(f"dZ_{i}/dt = {eq}")

    # 4. Visualization
    print("--- 4. Visualizing Results ---")
    model.eval()
    with torch.no_grad():
        test_idx = 0
        data = dataset[test_idx]
        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        z, s = model.encode(data.x, data.edge_index, batch)
        recon = model.decode(z, s, batch).numpy()
        
    plt.figure(figsize=(12, 5))
    
    # Micro Plot
    plt.subplot(1, 2, 1)
    plt.scatter(pos[test_idx, :, 0], pos[test_idx, :, 1], c='blue', label='Ground Truth')
    plt.scatter(recon[:, 0], recon[:, 1], c='red', marker='x', label='Reconstruction')
    plt.title("Micro-scale: Particles")
    plt.legend()
    
    # Meso Plot (Latent states over time)
    plt.subplot(1, 2, 2)
    plt.plot(z_states[:, 0], label='Z_0')
    plt.plot(z_states[:, 1], label='Z_1')
    plt.title("Meso-scale: Latent Dynamics")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("discovery_result.png")
    print("Result saved to discovery_result.png")

if __name__ == "__main__":
    main()
