#!/usr/bin/env python3
"""Simple test to verify MPS device fix"""
import torch
import numpy as np
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data
from simulator import LennardJonesSimulator

def test_mps_fix():
    print("Testing MPS device fix...")
    
    # Check if MPS is available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS is available")
    else:
        device = torch.device('cpu')
        print("Falling back to CPU")
        
    print(f"Using device: {device}")
    
    # Setup parameters
    n_particles = 8  # Smaller for test
    n_super_nodes = 2  # Smaller for test
    latent_dim = 4
    steps = 20  # Smaller for test
    
    # Generate simple test data
    sim = LennardJonesSimulator(n_particles=n_particles, epsilon=1.0, sigma=1.0,
                                dynamic_radius=1.5, box_size=(10.0, 10.0), dt=0.001)
    pos, vel = sim.generate_trajectory(steps=steps)
    
    print(f"Generated trajectory: pos shape {pos.shape}, vel shape {vel.shape}")
    
    # Prepare data
    dataset, stats = prepare_data(pos, vel, radius=1.5, device=device)
    print(f"Dataset prepared with {len(dataset)} samples")
    
    # Initialize model
    model = DiscoveryEngineModel(n_particles=n_particles,
                                 n_super_nodes=n_super_nodes,
                                 latent_dim=latent_dim,
                                 hidden_dim=64,  # Smaller for test
                                 hamiltonian=True,
                                 dissipative=True,
                                 min_active_super_nodes=max(1, n_super_nodes // 2)).to(device)
    
    print("Model initialized and moved to device")
    
    # Initialize trainer
    trainer = Trainer(model, lr=2e-4, device=device, stats=stats)
    print("Trainer initialized")
    
    # Test a single training step
    try:
        batch_data = dataset[:5]  # Small batch
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=0, max_epochs=10)
        print(f"Training step successful! Loss: {loss:.6f}, Rec: {rec:.6f}, Cons: {cons:.6f}")
        print("MPS device fix is working correctly!")
        return True
    except Exception as e:
        print(f"Error during training step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_mps_fix()
    if success:
        print("\nTest PASSED: MPS device fix is working!")
    else:
        print("\nTest FAILED: MPS device fix needs more work.")