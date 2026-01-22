import torch
import numpy as np
import argparse
import os
from datetime import datetime
import matplotlib.pyplot as plt
from torch_geometric.data import Batch

from simulator import SpringMassSimulator, LennardJonesSimulator
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data, analyze_latent_space
from symbolic import extract_latent_data, SymbolicDistiller
from train_utils import ImprovedEarlyStopping, robust_energy, get_device, SparsityScheduler
from visualization import plot_discovery_results, plot_training_history

def main():
    parser = argparse.ArgumentParser(description="Unified Emergentia Training Pipeline")
    parser.add_argument('--particles', type=int, default=16)
    parser.add_argument('--super_nodes', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sim', type=str, default='spring', choices=['spring', 'lj'])
    parser.add_argument('--hamiltonian', action='store_true')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    device = get_device() if args.device == 'auto' else args.device
    print(f"Using device: {device}")

    # 1. Setup Simulator
    if args.sim == 'spring':
        sim = SpringMassSimulator(n_particles=args.particles, dynamic_radius=2.0)
    else:
        sim = LennardJonesSimulator(n_particles=args.particles, dynamic_radius=2.0)

    # 2. Generate Data
    print("Generating trajectory...")
    pos, vel = sim.generate_trajectory(steps=args.steps)
    dataset, stats = prepare_data(pos, vel, radius=2.0, device=device)

    # 3. Setup Model & Trainer
    model = DiscoveryEngineModel(
        n_particles=args.particles,
        n_super_nodes=args.super_nodes,
        latent_dim=4,
        hamiltonian=args.hamiltonian
    ).to(device)

    trainer = Trainer(model, lr=args.lr, device=device, stats=stats)
    early_stopping = ImprovedEarlyStopping(patience=100)

    # 4. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        idx = np.random.randint(0, len(dataset) - 10)
        batch_data = dataset[idx : idx + 10]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=args.epochs)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Rec: {rec:.4f} | Cons: {cons:.4f}")
        
        if early_stopping(loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # 5. Analysis & Symbolic Discovery
    print("Analyzing latent space...")
    corrs = analyze_latent_space(model, dataset, pos, device=device)
    
    print("Extracting latent data for symbolic distillation...")
    z_states, dz_states, t_states = extract_latent_data(model, dataset, sim.dt)
    
    distiller = SymbolicDistiller()
    equations = distiller.distill(z_states, dz_states, args.super_nodes, 4)

    print("\nDiscovered Equations:")
    for i, eq in enumerate(equations):
        print(f"dz_{i}/dt = {eq}")

    # 6. Visualization
    print("Visualizing results...")
    # Get final assignments for plotting
    batch = Batch.from_data_list([dataset[0]]).to(device)
    z, s, _, _ = model.encode(batch.x, batch.edge_index, batch.batch)
    assignments = torch.argmax(s, dim=1).cpu().numpy()
    
    plot_discovery_results(model, dataset, pos, s, z_states.reshape(-1, args.super_nodes, 4), assignments)
    plot_training_history(trainer.loss_tracker)

if __name__ == "__main__":
    main()
