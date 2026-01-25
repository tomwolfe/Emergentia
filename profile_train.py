import torch
import time
import numpy as np
from torch_geometric.data import Batch
from simulator import LennardJonesSimulator
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data, compute_stats

def profile_train():
    device = 'cpu' 
    if torch.backends.mps.is_available():
        device = 'mps'
    
    print(f"Profiling on {device}...")
    
    # 1. Setup Simulator
    sim = LennardJonesSimulator(n_particles=8, dynamic_radius=2.0)
    pos, vel = sim.generate_trajectory(steps=100)
    dataset_list, stats = prepare_data(pos, vel, radius=2.0, device=device)
    dataset = Batch.from_data_list(dataset_list).to(device)
    dataset.seq_len = len(dataset_list)

    # 2. Setup Model & Trainer
    model = DiscoveryEngineModel(
        n_particles=8,
        n_super_nodes=4,
        latent_dim=6,
        hidden_dim=256,
        hamiltonian=True
    ).to(device)

    trainer = Trainer(
        model,
        lr=3e-4,
        device=device,
        stats=stats,
        warmup_epochs=20,
        max_epochs=400
    )

    # 3. Profile train_step
    print("Starting warmup...")
    # OPTIMIZATION: Use pre-batched data for profiling
    batch_data = Batch.from_data_list(dataset_list[:20]).to(device)
    batch_data.seq_len = 20
    
    for _ in range(5):
        trainer.train_step(batch_data, sim.dt, epoch=20, max_epochs=400) # Stage 1
    
    print("Starting profile (Stage 2)...")
    start_time = time.time()
    for i in range(10):
        t0 = time.time()
        trainer.train_step(batch_data, sim.dt, epoch=101 + i, max_epochs=400)
        print(f"Step {i}: {time.time() - t0:.4f}s")
    
    avg_time = (time.time() - start_time) / 10
    print(f"Average train_step time (Stage 2): {avg_time:.4f}s")

if __name__ == "__main__":
    profile_train()
