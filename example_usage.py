"""
Comprehensive example demonstrating the Neural-Symbolic Discovery Pipeline
with all enhanced features: secondary optimization, coordinate alignment,
and collapse prevention.
"""

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from model import DiscoveryEngineModel
from enhanced_symbolic import EnhancedSymbolicDistiller, PhysicsAwareSymbolicDistiller
from coordinate_mapping import AlignedHamiltonianSymbolicDistiller, create_enhanced_coord_mapper
from stable_pooling import StableHierarchicalPooling, DynamicLossBalancer
from hamiltonian_symbolic import HamiltonianSymbolicDistiller
from balanced_features import BalancedFeatureTransformer
from simulator import LennardJonesSimulator
from symbolic import extract_latent_data
import matplotlib.pyplot as plt


def create_sample_dataset(n_trajectories=10, trajectory_length=50, n_particles=20):
    """
    Create a sample dataset of particle trajectories for demonstration.
    In practice, you would load your actual particle dynamics data.
    """
    print("Creating sample dataset...")
    
    dataset = []
    simulator = LennardJonesSimulator(n_particles=n_particles, box_size=10.0)
    
    for i in range(n_trajectories):
        # Generate random initial conditions
        pos = torch.randn(n_particles, 2) * 2.0
        vel = torch.randn(n_particles, 2) * 0.5
        masses = torch.ones(n_particles) * 1.0
        
        # Simulate trajectory
        trajectory = simulator.simulate(pos, vel, masses, n_steps=trajectory_length, dt=0.01)
        
        # Create graph data for each time step
        for t in range(len(trajectory)):
            pos_t, vel_t = trajectory[t]
            
            # Node features: [pos_x, pos_y, vel_x, vel_y]
            x = torch.cat([pos_t, vel_t], dim=1)
            
            # Create edges (all-to-all for this example)
            n_nodes = x.size(0)
            row = torch.arange(n_nodes).repeat_interleave(n_nodes)
            col = torch.arange(n_nodes).repeat(n_nodes)
            edge_index = torch.stack([row, col], dim=0)
            
            # Remove self-loops
            mask = edge_index[0] != edge_index[1]
            edge_index = edge_index[:, mask]
            
            data = Data(x=x, edge_index=edge_index)
            dataset.append(data)
    
    return dataset


def train_model_with_enhancements():
    """
    Train the model with all enhancements: enhanced pooling, coordinate alignment,
    and symbolic distillation with secondary optimization.
    """
    print("Starting training with enhanced features...")
    
    # Create dataset
    dataset = create_sample_dataset(n_trajectories=5, trajectory_length=30, n_particles=15)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize model with enhanced pooling
    model = DiscoveryEngineModel(
        n_particles=15,
        n_super_nodes=8,  # Reduced for demo
        node_features=4,  # [pos_x, pos_y, vel_x, vel_y]
        latent_dim=4,     # [q_x, q_y, p_x, p_y] for Hamiltonian system
        hidden_dim=64,
        hamiltonian=True,
        dissipative=True
    )
    
    # Use enhanced pooling (though it's integrated in the model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    model.train()
    for epoch in range(10):  # Few epochs for demo
        total_loss = 0
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            z, s, assign_losses, mu = model.encode(data.x, data.edge_index, data.batch)
            
            # Simple reconstruction loss for demo
            x_recon = model.decode(z, s, data.batch)
            recon_loss = nn.MSELoss()(x_recon, data.x)
            
            # Combine losses
            total_batch_loss = recon_loss
            for loss_val in assign_losses.values():
                total_batch_loss += 0.01 * loss_val  # Small weight for assignment losses
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
        
        print(f"Epoch {epoch+1}/10, Loss: {total_loss/len(dataloader):.4f}")
    
    return model


def demonstrate_symbolic_distillation(model, dataset):
    """
    Demonstrate enhanced symbolic distillation with secondary optimization
    and coordinate alignment.
    """
    print("\nDemonstrating enhanced symbolic distillation...")
    
    # Extract latent dynamics data
    dt = 0.01
    latent_states, latent_derivs, times = extract_latent_data(model, dataset[:50], dt)
    
    print(f"Extracted {len(latent_states)} data points for distillation")
    print(f"Latent state shape: {latent_states.shape}")
    print(f"Latent derivative shape: {latent_derivs.shape}")
    
    # Method 1: Enhanced symbolic distillation with secondary optimization
    print("\n1. Enhanced symbolic distillation with secondary optimization:")
    enhanced_distiller = EnhancedSymbolicDistiller(
        populations=1000,  # Reduced for demo
        generations=20,    # Reduced for demo
        secondary_optimization=True,
        opt_method='L-BFGS-B',
        opt_iterations=50
    )
    
    equations_enhanced = enhanced_distiller.distill_with_secondary_optimization(
        latent_states=latent_states,
        targets=latent_derivs,
        n_super_nodes=model.encoder.n_super_nodes,
        latent_dim=model.encoder.latent_dim
    )
    
    print(f"Discovered {len(equations_enhanced)} equations with enhanced distillation")
    
    # Method 2: Physics-aware symbolic distillation
    print("\n2. Physics-aware symbolic distillation:")
    physics_distiller = PhysicsAwareSymbolicDistiller(
        populations=1000,
        generations=20,
        secondary_optimization=True
    )
    
    equations_physics = physics_distiller.distill(
        latent_states=latent_states,
        targets=latent_derivs,
        n_super_nodes=model.encoder.n_super_nodes,
        latent_dim=model.encoder.latent_dim
    )
    
    print(f"Discovered {len(equations_physics)} equations with physics awareness")
    
    # Method 3: Hamiltonian distillation with coordinate alignment
    print("\n3. Hamiltonian distillation with coordinate alignment:")
    hamiltonian_distiller = AlignedHamiltonianSymbolicDistiller(
        populations=1000,
        generations=20,
        enforce_hamiltonian_structure=True
    )
    
    # For demonstration, we'll use the same data (in practice, you might have physical coordinates)
    equations_hamiltonian = hamiltonian_distiller.distill_with_alignment(
        neural_latents=latent_states.reshape(-1, model.encoder.n_super_nodes, model.encoder.latent_dim),
        targets=latent_derivs,
        n_super_nodes=model.encoder.n_super_nodes,
        latent_dim=model.encoder.latent_dim
    )
    
    print(f"Discovered {len(equations_hamiltonian)} Hamiltonian equations with alignment")
    
    return equations_enhanced, equations_physics, equations_hamiltonian


def evaluate_discovered_equations(equations_enhanced, equations_physics, equations_hamiltonian, model, dataset):
    """
    Evaluate the discovered equations qualitatively.
    """
    print("\nEvaluating discovered equations...")
    
    # Extract latent dynamics data for evaluation
    dt = 0.01
    latent_states, latent_derivs, times = extract_latent_data(model, dataset[:20], dt)
    
    # Evaluate enhanced equations
    if equations_enhanced and equations_enhanced[0] is not None:
        try:
            X_poly = enhanced_distiller.transformer.transform(latent_states)
            X_norm = enhanced_distiller.transformer.normalize_x(X_poly)
            
            if hasattr(equations_enhanced[0], 'execute'):
                y_pred = equations_enhanced[0].execute(X_norm[:, enhanced_distiller.feature_masks[0]])
                y_true = enhanced_distiller.transformer.normalize_y(latent_derivs)[:, 0]
                
                mse = np.mean((y_true - y_pred.flatten())**2)
                print(f"Enhanced distillation MSE (first equation): {mse:.6f}")
        except Exception as e:
            print(f"Could not evaluate enhanced equations: {e}")
    
    # Print summary of discoveries
    print(f"\nSummary:")
    print(f"- Enhanced symbolic distillation: {len([eq for eq in equations_enhanced if eq is not None])}/{len(equations_enhanced)} equations discovered")
    print(f"- Physics-aware distillation: {len([eq for eq in equations_physics if eq is not None])}/{len(equations_physics)} equations discovered")  
    print(f"- Hamiltonian alignment: {len([eq for eq in equations_hamiltonian if eq is not None])}/{len(equations_hamiltonian)} equations discovered")


def main():
    """
    Main function demonstrating the complete pipeline.
    """
    print("="*60)
    print("NEURAL-SYMBOLIC DISCOVERY PIPELINE DEMONSTRATION")
    print("="*60)
    
    print("\nStep 1: Creating sample dataset...")
    dataset = create_sample_dataset(n_trajectories=5, trajectory_length=30, n_particles=15)
    print(f"Created dataset with {len(dataset)} samples")
    
    print("\nStep 2: Training model with enhanced features...")
    model = train_model_with_enhancements()
    print("Model training completed")
    
    print("\nStep 3: Performing enhanced symbolic distillation...")
    equations_enhanced, equations_physics, equations_hamiltonian = demonstrate_symbolic_distillation(model, dataset)
    
    print("\nStep 4: Evaluating discovered equations...")
    evaluate_discovered_equations(equations_enhanced, equations_physics, equations_hamiltonian, model, dataset)
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED")
    print("Enhanced features implemented:")
    print("- Secondary optimization for GP convergence")
    print("- Coordinate alignment for neural-symbolic handshake")
    print("- Collapse prevention in hierarchical pooling")
    print("- Physics-aware symbolic distillation")
    print("="*60)


if __name__ == "__main__":
    main()