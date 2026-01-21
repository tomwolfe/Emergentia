
import torch
import numpy as np
from torch_geometric.data import Batch
from engine import Trainer, prepare_data, compute_stats
from model import DiscoveryEngineModel
from simulator import SpringMassSimulator
from enhanced_symbolic import EnhancedSymbolicDistiller

def test_symbolic_in_the_loop():
    print("Testing Symbolic-in-the-loop training...")
    n_particles = 8
    n_super_nodes = 2
    latent_dim = 4
    
    sim = SpringMassSimulator(n_particles=n_particles)
    pos, vel = sim.generate_trajectory(steps=50)
    stats = compute_stats(pos, vel)
    dataset, _ = prepare_data(pos, vel, stats=stats)
    
    model = DiscoveryEngineModel(n_particles=n_particles, n_super_nodes=n_super_nodes, latent_dim=latent_dim)
    trainer = Trainer(model, stats=stats)
    
    # 1. Short initial training
    print("Phase 1: Initial GNN training...")
    for epoch in range(20):
        idx = np.random.randint(0, len(dataset) - 2)
        batch_data = dataset[idx:idx+2]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=100)
    
    print(f"Initial loss: {loss:.6f}")
    
    # 2. Extract latents for symbolic distillation
    print("Phase 2: Symbolic Distillation...")
    model.eval()
    latents = []
    targets = []
    
    with torch.no_grad():
        for i in range(len(dataset)-1):
            batch_t = Batch.from_data_list([dataset[i]])
            batch_next = Batch.from_data_list([dataset[i+1]])
            z, _, _, _ = model.encode(batch_t.x, batch_t.edge_index, batch_t.batch)
            z_next, _, _, _ = model.encode(batch_next.x, batch_next.edge_index, batch_next.batch)
            
            z_flat = z.view(-1).cpu().numpy()
            z_next_flat = z_next.view(-1).cpu().numpy()
            dz = (z_next_flat - z_flat) / sim.dt
            
            latents.append(z_flat)
            targets.append(dz)
            
    latents = np.array(latents)
    targets = np.array(targets)
    
    distiller = EnhancedSymbolicDistiller(populations=100, generations=5) # Small for testing
    equations = distiller.distill_with_secondary_optimization(latents, targets, n_super_nodes, latent_dim)
    
    # 3. Update symbolic proxy
    print("Phase 3: Updating symbolic proxy...")
    trainer.update_symbolic_proxy(equations, distiller.transformer, weight=0.1)
    
    # 4. Resume training with symbolic guidance
    print("Phase 4: Resuming training with Symbolic-in-the-loop...")
    for epoch in range(21, 40):
        idx = np.random.randint(0, len(dataset) - 2)
        batch_data = dataset[idx:idx+2]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=100)
        
    print(f"Final loss: {loss:.6f}")
    print("Symbolic-in-the-loop test completed successfully!")

if __name__ == "__main__":
    test_symbolic_in_the_loop()
