import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data
from symbolic import SymbolicDistiller, extract_latent_data

def main():
    # 0. Device Discovery
    # Fallback to CPU if only MPS is available due to torchdiffeq float64 issues
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # 1. Setup Parameters
    n_particles = 16
    n_super_nodes = 2
    latent_dim = 2 
    steps = 500
    epochs = 500
    seq_len = 5
    dynamic_radius = 1.5 
    box_size = (10.0, 10.0) # Periodic Boundary
    
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
                                 latent_dim=latent_dim).to(device)
    trainer = Trainer(model, lr=1e-3, device=device)
    
    last_loss = 1.0
    for epoch in range(epochs):
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
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        z, s = model.encode(x, edge_index, batch)
        recon = model.decode(z, s, batch).cpu().numpy()
        
    plt.figure(figsize=(12, 5))
    
    # Micro Plot
    plt.subplot(1, 2, 1)
    # Note: 'pos' contains raw coordinates, we should compare recon to normalized input or denormalize recon
    # For simplicity, we compare recon features to data.x
    plt.scatter(data.x.cpu().numpy()[:, 0], data.x.cpu().numpy()[:, 1], c='blue', label='Ground Truth (Norm)')
    plt.scatter(recon[:, 0], recon[:, 1], c='red', marker='x', label='Reconstruction')
    plt.title("Micro-scale: Particles (Normalized)")
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
