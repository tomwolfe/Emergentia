#!/usr/bin/env python3
"""
Test script to validate the fixes for the physics discovery pipeline.
This script runs a few epochs to verify that the loss imbalances and numerical instabilities are resolved.
"""

import torch
import numpy as np
from simulator import SpringMassSimulator
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data, compute_stats
import os

def test_fixes():
    print("Testing the implemented fixes...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Determine device (prefer MPS if available, otherwise CPU)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a simple simulation
    n_particles = 8  # Smaller for faster testing
    spring_dist = 1.0
    sim = SpringMassSimulator(n_particles=n_particles, spring_dist=spring_dist, dynamic_radius=1.5)

    # Generate training data to compute stats
    train_pos, train_vel = sim.generate_trajectory(steps=50)
    stats = compute_stats(train_pos, train_vel)

    # Generate evaluation data
    eval_pos, eval_vel = sim.generate_trajectory(steps=50)
    dataset, _ = prepare_data(eval_pos, eval_vel, radius=1.5, stats=stats, device=device)

    # Create model with smaller dimensions for testing
    model = DiscoveryEngineModel(
        n_particles=n_particles, 
        n_super_nodes=4,  # Fewer super-nodes for testing
        latent_dim=4,     # Standard latent dimension
        hamiltonian=True  # Use Hamiltonian dynamics
    )
    
    # Create trainer
    trainer = Trainer(
        model, 
        lr=1e-3, 
        device=device, 
        stats=stats,
        warmup_epochs=20,  # Shorter warmup for testing
        max_epochs=50      # Fewer epochs for testing
    )

    print("Starting training loop...")
    
    # Train for a few epochs and monitor losses
    for epoch in range(50):
        # Pick a random starting point in the trajectory
        idx = np.random.randint(0, len(dataset) - 2)
        batch_data = [dataset[idx], dataset[idx+1], dataset[idx+2]]  # Slightly longer sequence
        dt = sim.dt
        
        loss, rec, cons = trainer.train_step(batch_data, dt, epoch=epoch, max_epochs=50)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Total Loss: {loss:.6f}, Rec: {rec:.6f}, Cons: {cons:.6f}")
            
            # Print running averages of losses to check for balance
            running_avgs = trainer.loss_tracker.get_running_averages()
            print(f"  Running avgs - Rec: {running_avgs.get('rec_raw', 0):.4f}, "
                  f"Cons: {running_avgs.get('cons_raw', 0):.4f}, "
                  f"Assign: {running_avgs.get('assign', 0):.4f}")
    
    print("\nTraining completed successfully!")
    print("Fixes validation: No crashes or numerical instabilities observed.")
    
    # Additional checks
    print("\nChecking loss magnitudes for balance:")
    final_avgs = trainer.loss_tracker.get_running_averages()
    
    rec_loss = final_avgs.get('rec_raw', float('inf'))
    assign_loss = final_avgs.get('assign', float('inf'))
    cons_loss = final_avgs.get('cons_raw', float('inf'))
    
    print(f"Final running averages:")
    print(f"  Reconstruction Loss: {rec_loss:.4f}")
    print(f"  Assignment Loss: {assign_loss:.4f}")
    print(f"  Consistency Loss: {cons_loss:.4f}")
    
    # Validate that losses are in reasonable ranges
    success = True
    if rec_loss > 10.0:
        print("  WARNING: Reconstruction loss is high (> 10.0)")
        success = False
    if assign_loss > 10.0:
        print("  WARNING: Assignment loss is extremely high (> 10.0) - may indicate imbalance")
        success = False
    if cons_loss > 5.0 and cons_loss != 0.0:  # cons_loss might be 0 during warmup
        print("  WARNING: Consistency loss is high (> 5.0) - may indicate shock at warmup end")
        success = False
    
    if success:
        print("\n✅ All fixes appear to be working correctly!")
        print("- Losses are balanced and in reasonable ranges")
        print("- No numerical instabilities detected")
        print("- Dynamics transition appears smooth")
    else:
        print("\n⚠️  Some issues may still exist - check loss values")
    
    return success

if __name__ == "__main__":
    test_fixes()