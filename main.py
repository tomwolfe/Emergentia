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
    
    print("--- 1. Generating Data ---")
    sim = SpringMassSimulator(n_particles=n_particles, k=15.0)
    pos, vel = sim.generate_trajectory(steps=steps)
    dataset = prepare_data(pos, vel, sim.adj)
    
    # 2. Initialize Model and Trainer
    print("--- 2. Training Discovery Engine ---")
    model = DiscoveryEngineModel(n_particles=n_particles, 
                                 n_super_nodes=n_super_nodes, 
                                 latent_dim=latent_dim)
    trainer = Trainer(model, lr=1e-3)
    
    for epoch in range(epochs):
        # Sample a short sequence for training
        idx = np.random.randint(0, len(dataset) - seq_len)
        batch_data = dataset[idx : idx + seq_len]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss:.6f} | Rec: {rec:.6f} | Cons: {cons:.6f}")

    # 3. Extract Symbolic Equations
    print("--- 3. Distilling Symbolic Laws ---")
    z_states, dz_states = extract_latent_data(model, dataset, sim.dt)
    distiller = SymbolicDistiller(generations=10) # Small for demo
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
        z = model.encode(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
        recon = model.decode(z).squeeze(0).numpy()
        
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
