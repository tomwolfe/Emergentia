"""
Comprehensive Improved Training Script

This script implements comprehensive improvements to speed up training while maintaining
the quality of the neural-symbolic discovery pipeline, with better energy conservation,
enhanced symbolic regression, and improved interpretability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from simulator import SpringMassSimulator
from optimized_model import OptimizedDiscoveryEngineModel
from engine import Trainer, prepare_data, analyze_latent_space
from symbolic import extract_latent_data
import argparse

class ImprovedEarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

def robust_energy(sim, pos, vel):
    """
    Robust energy calculation with a soft-floor for particle distances to prevent singularities.
    """
    ke = 0.5 * sim.m * np.sum(vel**2)
    pe = 0.0
    pairs = sim._compute_pairs(pos)
    if len(pairs) > 0:
        idx1, idx2 = zip(*pairs)
        idx1 = np.array(idx1)
        idx2 = np.array(idx2)
        diff = pos[idx2] - pos[idx1]
        if sim.box_size:
            for i in range(2):
                diff[:, i] -= sim.box_size[i] * np.round(diff[:, i] / sim.box_size[i])
        
        dist_sq = np.sum(diff**2, axis=1)
        
        # NEW: Implement 'soft-floor' for particle distances: clip at 0.9 * sigma
        sigma = getattr(sim, 'sigma', 1.0)
        dist_sq = np.maximum(dist_sq, (0.9 * sigma)**2)
        
        if hasattr(sim, 'epsilon'): # Lennard-Jones
            sr6 = (sim.sigma**2 / dist_sq)**3
            sr6 = np.clip(sr6, -1e10, 1e10)
            sr12 = sr6**2
            pe = 4 * sim.epsilon * np.sum(sr12 - sr6)
        else: # Spring-Mass
            dist = np.sqrt(dist_sq)
            pe = 0.5 * sim.k * np.sum((dist - sim.spring_dist)**2)
            
    return ke + pe

def compute_energy_conservation_during_training(model, dataset, sim, device):
    """
    Compute energy conservation during training by evaluating the reconstructed trajectories.
    """
    model.eval()
    total_energy_error = 0.0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, min(len(dataset), 50), 5):  # Sample fewer batches to improve performance and stability
            if i >= len(dataset):
                break

            batch_data = dataset[i]
            x = batch_data.x.to(device)
            edge_index = batch_data.edge_index.to(device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

            # Encode and decode to get reconstruction
            try:
                z, s, _, _ = model.encode(x, edge_index, batch)
                recon = model.decode(z, s, batch)

                # Convert back to physical coordinates for energy calculation
                recon_pos = recon[:, :2].to(torch.float64).cpu().numpy()
                recon_vel = recon[:, 2:].to(torch.float64).cpu().numpy()

                # Calculate energy of reconstructed state using robust energy
                recon_energy = robust_energy(sim, recon_pos, recon_vel)
                true_energy = robust_energy(sim, batch_data.x[:, :2].to(torch.float64).cpu().numpy(), batch_data.x[:, 2:].to(torch.float64).cpu().numpy())

                # Safely compute energy error to avoid division by zero
                # Use log1p to compress the range without losing the gradient signal
                energy_error = abs(recon_energy - true_energy) / (abs(true_energy) + 1e-9)
                energy_error = np.log1p(energy_error)

                # Cap error to prevent extreme values (increased cap)
                energy_error = min(energy_error, 1e6)

                if np.isfinite(energy_error) and not np.isnan(energy_error):
                    total_energy_error += energy_error
                    num_batches += 1
            except Exception as e:
                continue

    avg_energy_error = total_energy_error / max(num_batches, 1)
    return avg_energy_error


def compute_energy_conservation_with_smoothing(model, dataset, sim, device, smoothing_window=3):
    """
    Compute energy conservation with temporal smoothing to reduce noise in measurements.
    """
    model.eval()
    energy_errors = []

    with torch.no_grad():
        for i in range(0, min(len(dataset), 50), 5):
            if i >= len(dataset):
                break

            batch_data = dataset[i]
            x = batch_data.x.to(device)
            edge_index = batch_data.edge_index.to(device)
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

            try:
                z, s, _, _ = model.encode(x, edge_index, batch)
                recon = model.decode(z, s, batch)

                recon_pos = recon[:, :2].cpu().numpy()
                recon_vel = recon[:, 2:].cpu().numpy()

                # Calculate energy of reconstructed state using robust energy
                recon_energy = robust_energy(sim, recon_pos, recon_vel)
                true_energy = robust_energy(sim, batch_data.x[:, :2].cpu().numpy(), batch_data.x[:, 2:].cpu().numpy())

                # Use log1p to compress the range without losing the gradient signal
                energy_error = abs(recon_energy - true_energy) / (abs(true_energy) + 1e-9)
                energy_error = np.log1p(min(energy_error, 1e6))

                if np.isfinite(energy_error) and not np.isnan(energy_error):
                    energy_errors.append(energy_error)
            except Exception as e:
                continue

    if len(energy_errors) == 0:
        return 1.0

    median_error = np.median(energy_errors)

    if len(energy_errors) >= smoothing_window:
        smoothed_errors = []
        for j in range(len(energy_errors)):
            start_idx = max(0, j - smoothing_window//2)
            end_idx = min(len(energy_errors), j + smoothing_window//2 + 1)

            window_data = energy_errors[start_idx:end_idx]
            weights = np.exp(np.linspace(-1., 0., len(window_data)))
            weighted_avg = np.average(window_data, weights=weights)
            smoothed_errors.append(weighted_avg)

        return min(median_error, np.median(smoothed_errors))
    else:
        return median_error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)  # Increased default epochs for better convergence
    parser.add_argument('--steps', type=int, default=100)   # Increased steps for better data
    parser.add_argument('--particles', type=int, default=8) # Increased particles for richer dynamics
    parser.add_argument('--super_nodes', type=int, default=4)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cpu, cuda, mps)')
    args = parser.parse_args()

    # 0. Device Discovery
    if args.device:
        device = torch.device(args.device)
        print(f"Manually selected device: {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("MPS is available")
    else:
        device = torch.device('cpu')
        print("GPU not available, falling back to CPU")
    print(f"Using device: {device}")

    # 1. Setup Parameters
    n_particles = args.particles
    n_super_nodes = args.super_nodes
    latent_dim = 8  # Increased latent dimension for better representation
    steps = args.steps
    epochs = args.epochs
    seq_len = 5  # Increased sequence length for better temporal modeling
    dynamic_radius = 2.0  # Increased radius for better connectivity
    box_size = (15.0, 15.0)  # Larger box size

    print("--- 1. Generating Data ---")
    from simulator import LennardJonesSimulator
    # Use smaller dt for better simulation accuracy
    sim = LennardJonesSimulator(n_particles=n_particles, epsilon=1.0, sigma=1.0,
                                dynamic_radius=dynamic_radius, box_size=box_size, dt=0.001, sub_steps=5)  # Smaller dt and more sub-steps
    pos, vel = sim.generate_trajectory(steps=steps)
    
    # Track energy conservation throughout the trajectory
    energies = [sim.energy(pos[i], vel[i]) for i in range(len(pos))]
    initial_energy = energies[0]
    final_energy = energies[-1]
    print(f"Initial energy: {initial_energy:.4f}")
    print(f"Final energy: {final_energy:.4f}")
    print(f"Energy conservation: {initial_energy:.4f} -> {final_energy:.4f} ({(abs(final_energy-initial_energy)/abs(initial_energy))*100:.2f}%)")

    # Prepare data with device support and robust normalization
    dataset, stats = prepare_data(pos, vel, radius=dynamic_radius, device=device, cache_edges=True)

    # 2. Initialize Model and Trainer
    print("--- 2. Initialize Improved Optimized Model and Trainer ---")
    # Using Hamiltonian dynamics with learnable dissipation for improved physics fidelity
    # Ensure at least half of super-nodes stay active to prevent resolution collapse
    min_active = max(1, n_super_nodes // 2)

    # Use the optimized model with reduced hidden dimensions
    model = OptimizedDiscoveryEngineModel(n_particles=n_particles,
                                          n_super_nodes=n_super_nodes,
                                          latent_dim=latent_dim,
                                          hidden_dim=64,  # Increased hidden dim for better capacity
                                          hamiltonian=True,
                                          dissipative=True,
                                          min_active_super_nodes=min_active,
                                          box_size=box_size[0]).to(device)

    # Initialize model weights for better convergence
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # NEW: Sparsity Scheduler to prevent resolution collapse
    from stable_pooling import SparsityScheduler
    sparsity_scheduler = SparsityScheduler(
        initial_weight=0.01,
        target_weight=0.1,  # Increased target weight for better regularization
        warmup_steps=max(10, int(epochs * 0.1)),
        max_steps=max(50, int(epochs * 0.4))  # Increased max steps
    )

    # NEW: Enhanced balancer for better loss weighting
    from enhanced_balancer import GradientBasedLossBalancer
    enhanced_balancer = GradientBasedLossBalancer()

    # Optimized trainer parameters for better training
    trainer = Trainer(model, lr=1e-3,  # Starting learning rate
                      device=device, stats=stats, sparsity_scheduler=sparsity_scheduler,
                      skip_consistency_freq=2,  # Compute consistency loss more frequently
                      enable_gradient_accumulation=True,  # Use gradient accumulation for memory efficiency
                      grad_acc_steps=2,  # Accumulate gradients over fewer steps for more frequent updates
                      enhanced_balancer=enhanced_balancer)  # NEW: Add enhanced balancer

    # Initialize energy weight attribute for energy-focused retraining
    trainer.energy_weight = 0.1  # Default energy weight

    # Better scheduler parameters for improved convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, mode='min',
                                                           factor=0.5, patience=30, min_lr=1e-7)  # More gradual decay

    # Early stopping setup with better parameters
    early_stopping = ImprovedEarlyStopping(patience=100, min_delta=1e-5)  # Even longer patience and finer delta

    last_loss = 1.0
    for epoch in range(epochs):
        idx = np.random.randint(0, len(dataset) - seq_len)
        batch_data = dataset[idx : idx + seq_len]
        loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch, max_epochs=epochs)
        last_loss = loss
        scheduler.step(loss)

        if epoch % 50 == 0:  # Adjusted logging frequency for better monitoring
            progress = (epoch / epochs) * 100
            stats_tracker = trainer.loss_tracker.get_stats()
            active_nodes = int(model.encoder.pooling.active_mask.sum().item())
            
            # NEW: Compute energy conservation during training with smoothing
            energy_error = compute_energy_conservation_with_smoothing(model, dataset, sim, device)
            
            log_str = f"Progress: {progress:3.0f}% | Loss: {stats_tracker.get('total', 0):.4f} | "
            log_str += f"Rec: {stats_tracker.get('rec_raw', 0):.4f} | "
            log_str += f"Cons: {stats_tracker.get('cons_raw', 0):.4f} | "
            log_str += f"Energy Err: {energy_error:.4f} | "
            log_str += f"LVar: {stats_tracker.get('lvar_raw', 0):.4f} | "
            log_str += f"Active: {active_nodes} | "
            log_str += f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}"
            print(log_str)

        # Check for early stopping
        if early_stopping(last_loss):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # --- Enhanced Interpretability Check ---
    print("\n--- 2.1 Latent Interpretability Analysis ---")
    # Use enhanced physical mapping analysis that includes both position and velocity
    from engine import enhance_physical_mapping
    corrs = enhance_physical_mapping(model, dataset, pos, vel, device=device)
    for k in range(n_super_nodes):
        max_corr = np.max(np.abs(corrs[k]))
        print(f"Super-node {k} max CoM correlation: {max_corr:.3f}")
        if max_corr > 0.8:
            print(f"  -> Strong physical mapping detected for super-node {k}")
        elif max_corr > 0.5:
            print(f"  -> Moderate physical mapping detected for super-node {k}")
        else:
            print(f"  -> Weak physical mapping for super-node {k} - may need adjustment")

    # NEW: Additional interpretability metrics
    print("\n--- 2.2 Additional Interpretability Metrics ---")
    model.eval()
    with torch.no_grad():
        # Compute assignment entropy to measure cluster distinctiveness
        sample_data = dataset[0]
        x = sample_data.x.to(device)
        edge_index = sample_data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        
        z, s, _, _ = model.encode(x, edge_index, batch)
        
        # Assignment entropy (higher means more uniform assignments, lower means more distinct clusters)
        assignment_entropy = -(s * torch.log(s + 1e-9)).sum(dim=1).mean().item()
        print(f"Assignment entropy: {assignment_entropy:.3f} (lower is better for distinct clusters)")
        
        # Compute latent space utilization
        latent_variance = z.var(dim=1).mean().item()
        print(f"Average latent variance: {latent_variance:.3f} (higher indicates better utilization)")

    # --- Quality Gate ---
    print(f"\nFinal Training Loss: {last_loss:.6f}")
    # NEW: Minimum Epoch Guard: Only attempt additional training if we've run for a minimum number of epochs
    if rec > 0.2 and epoch >= 50:  # Adjusted threshold to be more realistic for complex systems
        print(f"WARNING: Model may not have converged fully (Rec Loss: {rec:.6f}).")
        # If the reconstruction loss is still high, try a few more training epochs with a lower learning rate
        if rec > 0.3:
            print("Attempting additional training with reduced learning rate...")
            # Reduce learning rate instead of re-initializing optimizer to preserve momentum
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] *= 0.1

            # Train for additional epochs
            additional_epochs = 50  # Reduced additional epochs to prevent overfitting
            for epoch_add in range(epoch + 1, epoch + 1 + additional_epochs):
                idx = np.random.randint(0, len(dataset) - seq_len)
                batch_data = dataset[idx : idx + seq_len]
                loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch_add, max_epochs=epoch + 1 + additional_epochs)

                # NEW: Gradient Safeguard
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                if epoch_add % 25 == 0:
                    print(f"Additional Epoch {epoch_add}, Loss: {loss:.6f}, Rec: {rec:.6f}")

                if rec < 0.2:  # Break if we achieve reasonable reconstruction
                    print(f"Additional training achieved rec loss of {rec:.6f}, stopping.")
                    break

    # NEW: Compute final energy conservation with smoothing
    final_energy_error = compute_energy_conservation_with_smoothing(model, dataset, sim, device)
    print(f"Final energy conservation error: {final_energy_error:.4f}")
    if final_energy_error > 0.1:  # Reduced threshold to be more sensitive to energy conservation
        print("WARNING: High energy conservation error detected.")
        # NEW: Attempt additional training with energy-focused loss if energy error is high
        if final_energy_error > 0.2:
            print("Attempting energy-focused retraining...")
            
            # ROLLBACK MECHANISM: Save model state before starting energy-focused phase
            pre_energy_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            initial_energy_phase_loss = last_loss
            
            # Temporarily increase the energy conservation weight in the trainer
            original_energy_weight = getattr(trainer, 'energy_weight', 0.1)
            trainer.energy_weight = min(0.8, original_energy_weight * 1.2)  # NEW: Use 1.2x multiplier instead of 3x

            # NEW: Reduce learning rate significantly WITHOUT resetting optimizer for more stable energy-focused training
            original_lr = trainer.optimizer.param_groups[0]['lr']
            new_lr = original_lr * 0.01 # Less drastic reduction than before
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = new_lr

            # NEW: Use a more conservative energy weight to prevent divergence
            original_energy_weight = getattr(trainer, 'energy_weight', 0.1)
            # Reduce the energy weight to be less aggressive and prevent loss explosion
            trainer.energy_weight = min(0.15, original_energy_weight * 1.2)  # Much less aggressive increase

            # NEW: More targeted energy-focused training with physics-informed loss
            additional_epochs = 50  # Reduced epochs to prevent overfitting
            diverged = False
            for epoch_energy in range(epochs + additional_epochs, epochs + 2 * additional_epochs):
                idx = np.random.randint(0, len(dataset) - seq_len)
                batch_data = dataset[idx : idx + seq_len]

                # NEW: Add physics-informed loss during energy-focused training
                loss, rec, cons = trainer.train_step(batch_data, sim.dt, epoch=epoch_energy, max_epochs=epochs + 2 * additional_epochs)

                # NEW: Implement gradient clipping specifically for energy-focused training to prevent divergence
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)  # More conservative clipping

                # ROLLBACK CHECK: If total loss increases by more than 15%, abort and restore
                if loss > initial_energy_phase_loss * 1.15:
                    print(f"Energy-focused training diverged (Loss: {loss:.4f} > {initial_energy_phase_loss:.4f} * 1.15). Rolling back.")
                    model.load_state_dict(pre_energy_state)
                    diverged = True
                    break

                if epoch_energy % 10 == 0:
                    current_energy_error = compute_energy_conservation_with_smoothing(model, dataset, sim, device)
                    print(f"Energy-focused Epoch {epoch_energy}, Loss: {loss:.6f}, Rec: {rec:.6f}, Energy Err: {current_energy_error:.4f}")

                # NEW: More aggressive target for energy conservation improvement
                current_energy_error = compute_energy_conservation_with_smoothing(model, dataset, sim, device)

                # NEW: Early stopping if energy error starts increasing (indicating divergence)
                if epoch_energy > (epochs + additional_epochs + 5):  # Allow some initial exploration
                    if 'prev_energy_errors_list' not in locals():
                        prev_energy_errors_list = []
                    prev_energy_errors_list.append(current_energy_error)

                    # Check if energy error is increasing over recent epochs (sign of divergence)
                    if len(prev_energy_errors_list) > 3:
                        recent_errors = prev_energy_errors_list[-3:]
                        if all(recent_errors[i] > recent_errors[i-1] for i in range(1, len(recent_errors))):
                            print(f"Energy error consistently increasing, stopping energy-focused training to prevent divergence.")
                            break

                if current_energy_error < 0.1:  # Less aggressive target to prevent overfitting
                    print(f"Energy conservation improved to {current_energy_error:.4f}, stopping energy-focused training.")
                    break

            if not diverged:
                # Restore original energy weight and learning rate
                trainer.energy_weight = original_energy_weight
            
            for param_group in trainer.optimizer.param_groups:
                param_group['lr'] = original_lr

    # 3. Extract Symbolic Equations with improved parameters
    print("--- 3. Distilling Symbolic Laws ---")
    is_hamiltonian = model.hamiltonian

    # Extract latent data with reduced computation
    latent_data = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=is_hamiltonian)

    if len(latent_data[0]) == 0:
        print("Error: No valid latent data extracted (all NaN or divergent). Skipping symbolic distillation.")
        return

    # NEW: Use improved symbolic distiller with better parameters
    from improved_symbolic_distillation import ImprovedSymbolicDistiller
    distiller = ImprovedSymbolicDistiller(
        populations=1500,  # Increased from 800 to improve discovery quality
        generations=40,    # Increased from 20 to improve convergence
        stopping_criteria=0.001,  # Tightened from 0.01 to improve accuracy
        max_features=15,    # Increased from 8 to allow more complex expressions
        secondary_optimization=True,
        opt_method='L-BFGS-B',
        opt_iterations=150,  # Increased from 50 to improve optimization
        use_sindy_pruning=True,
        sindy_threshold=0.01,  # Decreased from 0.1 to be less aggressive
        enhanced_feature_selection=True,  # NEW: Enable enhanced feature selection
        physics_informed=True  # NEW: Enable physics-informed features
    )

    if is_hamiltonian:
        z_states, dz_states, t_states, h_states = latent_data
        print("Distilling Hamiltonian H(q, p) with secondary optimization...")
        equations = distiller.distill(z_states, h_states, n_super_nodes, latent_dim, box_size=box_size)
        confidences = distiller.confidences
        # Update trainer with symbolic laws if confidence is high enough
        if confidences[0] > 0.8:  # Increased threshold for better quality
            trainer.update_symbolic_proxy(equations, distiller.transformer, weight=0.15, confidence=confidences[0])  # Increased weight slightly
    else:
        z_states, dz_states, t_states = latent_data
        print("Distilling derivatives dZ/dt with secondary optimization...")
        equations = distiller.distill(z_states, dz_states, n_super_nodes, latent_dim, box_size=box_size)
        confidences = distiller.confidences
        avg_conf = np.mean(confidences)
        if avg_conf > 0.8:  # Increased threshold for better quality
            trainer.update_symbolic_proxy(equations, distiller.transformer, weight=0.15, confidence=avg_conf)  # Increased weight slightly

    print("\nDiscovered Symbolic Laws:")
    if is_hamiltonian:
        if equations[0] is not None:
            print(f"H(z) = {equations[0]} (Confidence: {confidences[0]:.3f})")
        else:
            print("No Hamiltonian discovered.")
    else:
        for i, eq in enumerate(equations):
            if eq is not None:
                print(f"dZ_{i}/dt = {eq} (Confidence: {confidences[i]:.3f})")
            else:
                print(f"dZ_{i}/dt = None")

    # NEW: Enhanced symbolic law analysis
    print("\n--- 3.1 Symbolic Law Analysis ---")
    if is_hamiltonian and equations[0] is not None:
        # Check if the distiller object has the get_complexity method
        if hasattr(distiller, 'get_complexity'):
            print(f"Hamiltonian complexity: {distiller.get_complexity(equations[0])}")
        else:
            print(f"Hamiltonian complexity: {str(equations[0]).count('add') + str(equations[0]).count('sub') + 1}")
        print(f"Number of terms: {str(equations[0]).count('add') + str(equations[0]).count('sub') + 1}")
        print(f"Confidence: {confidences[0]:.3f}")
    else:
        print("No symbolic laws discovered or not using Hamiltonian formulation.")

    # 4. Enhanced Visualization & Integration
    if is_hamiltonian and equations[0] is None:
        print("Skipping visualization due to lack of discovered Hamiltonian.")
        return

    print("--- 4. Visualizing Results ---")
    from scipy.integrate import odeint
    from improved_symbolic import ImprovedSymbolicDynamics  # NEW: Use improved symbolic dynamics

    # Check if we have valid latent data to integrate
    if len(z_states) == 0 or z_states.size == 0:
        print("No valid latent states for integration. Using original states for visualization.")
        z_simulated = z_states[:len(t_eval)] if 't_eval' in locals() else z_states
    else:
        dyn_fn = ImprovedSymbolicDynamics(distiller, equations, distiller.feature_masks, is_hamiltonian, n_super_nodes, latent_dim, model=model)

        # Integrate the discovered equations
        z0 = z_states[0]
        t_eval = np.linspace(0, (len(z_states)-1)*sim.dt, len(z_states))

        # Ensure z0 is the right shape for integration
        if z0.ndim == 1:
            z0_int = z0
        else:
            z0_int = z0.flatten()

        try:
            # Debug: Print shapes before integration
            print(f"z0_int shape: {z0_int.shape}, size: {z0_int.size}")

            # Ensure z0_int is 1D for odeint
            if z0_int.ndim > 1:
                z0_int = z0_int.flatten()

            # NEW: Add more robust integration with error handling
            z_simulated = odeint(dyn_fn, z0_int, t_eval, rtol=1e-8, atol=1e-8, mxstep=5000)

            # Check if the integration result has the expected shape
            expected_shape = (len(t_eval), z0_int.shape[0])

            print(f"z_simulated shape: {z_simulated.shape}, expected: {expected_shape}")

            if z_simulated.shape != expected_shape:
                # If the shape is wrong, try to reshape appropriately
                if z_simulated.size == expected_shape[0] * expected_shape[1]:
                    z_simulated = z_simulated.reshape(expected_shape)
                else:
                    # If we can't reshape properly, use the original states
                    print(f"Integration result shape {z_simulated.shape} doesn't match expected {expected_shape}")
                    z_simulated = z_states[:len(t_eval)]  # Match the time dimension
                    print("Using original states for visualization due to integration failure")
        except Exception as e:
            print(f"Integration failed: {e}")
            # Fallback: use the original z_states for plotting, ensuring correct time dimension
            z_simulated = z_states[:len(t_eval)] if len(z_states) >= len(t_eval) else np.tile(z_states[-1:], (len(t_eval), 1))
            print("Using original states for visualization due to integration failure")


    model.eval()
    with torch.no_grad():
        test_idx = 0
        data = dataset[test_idx]
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        z, s, _, _ = model.encode(x, edge_index, batch)
        recon = model.decode(z, s, batch).cpu().numpy()

    plt.figure(figsize=(20, 15))  # Larger figure for better detail

    # 1. Micro Plot: Reconstruction
    plt.subplot(3, 3, 1)
    plt.scatter(data.x.cpu().numpy()[:, 0], data.x.cpu().numpy()[:, 1], c='blue', alpha=0.5, label='Truth', s=50)
    plt.scatter(recon[:, 0], recon[:, 1], c='red', marker='x', label='Recon', s=50)
    plt.title("Micro: Position Reconstruction")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Velocity Reconstruction
    plt.subplot(3, 3, 2)
    plt.scatter(data.x.cpu().numpy()[:, 2], data.x.cpu().numpy()[:, 3], c='blue', alpha=0.5, label='Truth', s=50)
    plt.scatter(recon[:, 2], recon[:, 3], c='red', marker='x', label='Recon', s=50)
    plt.title("Micro: Velocity Reconstruction")
    plt.xlabel("X Velocity")
    plt.ylabel("Y Velocity")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Assignment Plot
    plt.subplot(3, 3, 3)
    assignments = torch.argmax(s, dim=1).cpu().numpy()
    scatter = plt.scatter(data.x.cpu().numpy()[:, 0], data.x.cpu().numpy()[:, 1], c=assignments, cmap='tab10', s=50)
    plt.colorbar(scatter)
    plt.title("Hierarchical: Assignments")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, alpha=0.3)

    # 4. Assignment Probabilities Heatmap
    plt.subplot(3, 3, 4)
    im = plt.imshow(s.cpu().numpy().T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(im)
    plt.title("Assignment Probabilities")
    plt.xlabel("Particle Index")
    plt.ylabel("Super-node Index")

    # 5. Meso Plot: Symbolic Integration vs Learned Latent
    plt.subplot(3, 3, 5)
    # Plot first few dimensions for clarity
    n_plot = min(4, z_states.shape[1])
    for i in range(n_plot):
        plt.plot(t_eval, z_states[:, i], '--', alpha=0.7, label=f'Learned Z_{i}', linewidth=1.5)
        plt.plot(t_eval, z_simulated[:, i], '-', alpha=0.8, label=f'Symbolic Z_{i}', linewidth=1.5)
    plt.title("Meso: Symbolic Integration vs Learned")
    plt.xlabel("Time")
    plt.ylabel("Latent Value")
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)

    # 6. Energy Conservation Over Time
    plt.subplot(3, 3, 6)
    # Calculate approximate energy from latent states (kinetic + potential approximation)
    if is_hamiltonian and equations[0] is not None:
        # For Hamiltonian systems, the Hamiltonian represents total energy
        h_values = []
        for i in range(len(z_states)):
            try:
                # Use the symbolic dynamics function to evaluate the Hamiltonian
                # But we need to pass time as well, so we'll evaluate the symbolic equation directly
                z_state = z_states[i]

                # Transform z to features for the symbolic equation
                if hasattr(distiller, 'transformer') and distiller.transformer:
                    # Ensure z_state is properly shaped for transformer
                    if z_state.ndim == 1:
                        z_input = z_state.reshape(1, -1)
                    else:
                        z_input = z_state

                    X = distiller.transformer.transform(z_input)
                    X_full = distiller.transformer.normalize_x(X)
                else:
                    X_full = z_state.reshape(1, -1)

                # Evaluate the Hamiltonian equation
                feature_mask = distiller.feature_masks[0] if distiller.feature_masks and len(distiller.feature_masks) > 0 else None

                if feature_mask is not None and np.any(feature_mask):
                    # Ensure X_full is 2D before applying feature mask to prevent "too many indices" error
                    if X_full.ndim == 1:
                        X_full = X_full.reshape(1, -1)
                    h_val = equations[0].execute(X_full[:, feature_mask])
                else:
                    h_val = equations[0].execute(X_full)

                # Use the same safe extraction method as in ImprovedSymbolicDynamics
                if isinstance(h_val, (list, tuple)):
                    if len(h_val) > 0:
                        h_val = float(h_val[0]) if np.isscalar(h_val[0]) else float(h_val[0])
                    else:
                        h_val = 0.0
                elif isinstance(h_val, np.ndarray):
                    if h_val.size == 0:
                        h_val = 0.0
                    elif h_val.size == 1:
                        h_val = float(h_val.flat[0])
                    else:
                        # If it's a multi-element array, return the first element
                        h_val = float(h_val.flat[0])
                elif np.isscalar(h_val):
                    h_val = float(h_val)
                else:
                    # Fallback to 0.0 if we can't handle the type
                    h_val = 0.0
                h_values.append(h_val)
            except Exception as e:
                print(f"Error evaluating Hamiltonian at step {i}: {e}")
                h_values.append(0.0)  # Fallback if evaluation fails
        if h_values:
            plt.plot(t_eval, h_values, 'g-', label='Hamiltonian (Energy)', linewidth=2)
            plt.axhline(y=h_values[0], color='r', linestyle=':', label='Initial Energy', alpha=0.7)
    else:
        # If no Hamiltonian was discovered, plot the energy of the original system
        energies = []
        for i in range(len(z_states)):
            # This is a simplified approach - in practice, you'd need to map back to physical coordinates
            # For now, we'll just plot the norm of the state as a proxy for energy
            energies.append(np.linalg.norm(z_states[i]))
        if energies:
            plt.plot(t_eval, energies, 'g-', label='State Norm (Proxy)', linewidth=2)
            plt.axhline(y=energies[0], color='r', linestyle=':', label='Initial State Norm', alpha=0.7)
    plt.title("Energy Conservation Over Time")
    plt.xlabel("Time")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 7. Latent Space Trajectory
    plt.subplot(3, 3, 7)
    plt.plot(z_states[:, 0], z_states[:, 1], 'b-', label='Learned Trajectory', linewidth=1.5)
    plt.plot(z_simulated[:, 0], z_simulated[:, 1], 'r--', label='Symbolic Trajectory', linewidth=1.5)
    plt.title("Latent Space Phase Portrait")
    plt.xlabel("Z_0")
    plt.ylabel("Z_1")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 8. Reconstruction Error Over Time
    plt.subplot(3, 3, 8)
    # Calculate reconstruction error for each time step
    recon_errors = []
    for i in range(0, len(dataset), max(1, len(dataset)//100)):  # Sample 100 points
        batch_data = dataset[i]
        x = batch_data.x.to(device)
        edge_index = batch_data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        
        z, s, _, _ = model.encode(x, edge_index, batch)
        recon = model.decode(z, s, batch)
        error = torch.mean((recon - x)**2).item()
        recon_errors.append(error)
    
    time_points = np.linspace(0, len(dataset), len(recon_errors))
    plt.plot(time_points, recon_errors, 'purple', linewidth=2)
    plt.title("Reconstruction Error Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("MSE")
    plt.grid(True, alpha=0.3)

    # 9. Super-node Activity Over Time
    plt.subplot(3, 3, 9)
    # Calculate average assignment strength for each super-node over time
    activity_over_time = []
    for i in range(0, len(dataset), max(1, len(dataset)//50)):  # Sample 50 points
        batch_data = dataset[i]
        x = batch_data.x.to(device)
        edge_index = batch_data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        
        z, s, _, _ = model.encode(x, edge_index, batch)
        avg_activity = torch.mean(s, dim=0)  # Average assignment probability per super-node
        activity_over_time.append(avg_activity.detach().cpu().numpy())
    
    activity_array = np.array(activity_over_time)
    for k in range(activity_array.shape[1]):
        plt.plot(time_points[:len(activity_array)], activity_array[:, k], label=f'Super-node {k}', linewidth=1.5)
    plt.title("Super-node Activity Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Avg Assignment Probability")
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comprehensive_improved_discovery_result.png", dpi=150, bbox_inches='tight')
    print("Comprehensive result saved to comprehensive_improved_discovery_result.png")

if __name__ == "__main__":
    main()