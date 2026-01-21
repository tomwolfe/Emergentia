"""
Complete Enhanced Neural-Symbolic Discovery Pipeline with all improvements.

This script implements the 80/20 improvements identified in the critical analysis:
1. Enhanced learnable basis functions to address the basis function bottleneck
2. Optimized ODE functions to reduce adjoint sensitivity complexity
3. Improved hyperparameter management with auto-tuning
4. Enhanced symbolic distillation with secondary optimization
5. Robust symbolic proxy with validation
6. Multi-scale loss balancing
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
import argparse
import os
from datetime import datetime

# Import all enhanced components
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data, analyze_latent_space
from simulator import LennardJonesSimulator
from enhanced_symbolic import EnhancedSymbolicDistiller
from robust_symbolic import create_robust_symbolic_proxy
from enhanced_balancer import create_enhanced_loss_balancer
from learnable_basis import EnhancedFeatureTransformer
from optimized_ode import create_optimized_ode_func
from config_manager import load_config, ConfigManager
from symbolic import extract_latent_data
from optimized_symbolic import OptimizedSymbolicDynamics


def main():
    parser = argparse.ArgumentParser(description="Enhanced Neural-Symbolic Discovery Pipeline")
    parser.add_argument('--config', type=str, default='./config.yaml', help='Path to config file')
    parser.add_argument('--problem_type', type=str, default='general', 
                       choices=['general', 'small', 'large', 'physics', 'chaos'],
                       help='Type of problem to optimize for')
    parser.add_argument('--auto_tune', action='store_true', help='Enable auto-tuning')
    parser.add_argument('--device', type=str, default='auto', help='Device override')
    args = parser.parse_args()

    print("="*80)
    print("ENHANCED NEURAL-SYMBOLIC DISCOVERY PIPELINE")
    print("="*80)

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config_manager = load_config(args.config)
    
    # Override device if specified
    if args.device != 'auto':
        config_manager.system_config.device = args.device
    
    # Validate configuration
    issues = config_manager.validate_config()
    if issues:
        print(f"Configuration issues found: {issues}")
        return
    
    # Get current config
    config = config_manager.get_current_config()
    
    # Set device
    if config['system']['device'] == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("CUDA is available")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("MPS is available")
        else:
            device = torch.device('cpu')
            print("GPU not available, falling back to CPU")
    else:
        device = torch.device(config['system']['device'])
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(config['system']['seed'])
    np.random.seed(config['system']['seed'])
    
    # Create directories
    os.makedirs(config['system']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['system']['log_dir'], exist_ok=True)
    os.makedirs(config['system']['results_dir'], exist_ok=True)

    # 1. Setup Parameters
    print("\n--- 1. Setting up parameters ---")
    mc = config['model']
    tc = config['training']
    sc = config['symbolic']
    
    n_particles = mc['n_particles']
    n_super_nodes = mc['n_super_nodes']
    latent_dim = mc['latent_dim']
    steps = tc['steps']
    epochs = tc['epochs']
    seq_len = tc['seq_len']
    dynamic_radius = tc['dynamic_radius']
    box_size = tc['box_size']

    print(f"Particles: {n_particles}, Super-nodes: {n_super_nodes}, Latent dim: {latent_dim}")
    print(f"Steps: {steps}, Epochs: {epochs}, Sequence length: {seq_len}")

    # 2. Generate Data
    print("\n--- 2. Generating Data ---")
    sim = LennardJonesSimulator(
        n_particles=n_particles, 
        epsilon=1.0, 
        sigma=1.0,
        dynamic_radius=dynamic_radius, 
        box_size=box_size, 
        dt=0.001
    )
    pos, vel = sim.generate_trajectory(steps=steps)
    initial_energy = sim.energy(pos[0], vel[0])
    final_energy = sim.energy(pos[-1], vel[-1])
    print(f"Energy conservation: {initial_energy:.2f} -> {final_energy:.2f} ({(final_energy/initial_energy-1)*100:.2f}%)")

    # Prepare data with device support and robust normalization
    dataset, stats = prepare_data(pos, vel, radius=dynamic_radius, device=device)

    # 3. Initialize Model with Enhanced Components
    print("\n--- 3. Initializing Enhanced Model ---")
    # Use optimized ODE function
    optimized_ode_func = create_optimized_ode_func(
        latent_dim=mc['latent_dim'],
        n_super_nodes=mc['n_super_nodes'],
        hidden_dim=mc['hidden_dim'],
        dissipative=mc['dissipative'],
        optimization_level='standard'
    )
    
    # Initialize model
    min_active = max(1, n_super_nodes // 2)
    model = DiscoveryEngineModel(
        n_particles=n_particles,
        n_super_nodes=n_super_nodes,
        latent_dim=latent_dim,
        hidden_dim=mc['hidden_dim'],
        hamiltonian=mc['hamiltonian'],
        dissipative=mc['dissipative'],
        min_active_super_nodes=min_active
    ).to(device)
    
    # Replace the ODE function with the optimized version
    model.ode_func = optimized_ode_func.to(device)

    # 4. Setup Enhanced Training Components
    print("\n--- 4. Setting up Enhanced Training Components ---")
    
    # Sparsity Scheduler
    if tc['sparsity_scheduler_enabled']:
        from stable_pooling import SparsityScheduler
        sparsity_scheduler = SparsityScheduler(
            initial_weight=tc['sparsity_initial_weight'],
            target_weight=tc['sparsity_target_weight'],
            warmup_steps=tc['sparsity_warmup_steps'],
            max_steps=tc['sparsity_max_steps']
        )
    else:
        sparsity_scheduler = None

    # Enhanced Loss Balancer
    if tc['use_enhanced_balancer']:
        enhanced_balancer = create_enhanced_loss_balancer(
            strategy=tc['enhanced_balancer_strategy']
        )
        print(f"Using enhanced {tc['enhanced_balancer_strategy']} loss balancer")
    else:
        enhanced_balancer = None

    # Initialize Trainer with enhanced components
    trainer = Trainer(
        model, 
        lr=tc['lr'], 
        device=device, 
        stats=stats, 
        sparsity_scheduler=sparsity_scheduler,
        skip_consistency_freq=tc['skip_consistency_freq'],
        enable_gradient_accumulation=tc['enable_gradient_accumulation'],
        grad_acc_steps=tc['grad_acc_steps'],
        enhanced_balancer=enhanced_balancer,
        warmup_epochs=tc['warmup_epochs'],
        align_anneal_epochs=tc['align_anneal_epochs'],
        hard_assignment_start=tc['hard_assignment_start']
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer, 
        mode='min',
        factor=0.5, 
        patience=100, 
        min_lr=1e-6
    )

    # 5. Training Loop
    print("\n--- 5. Starting Enhanced Training ---")
    print(f"Training for {epochs} epochs...")
    
    last_loss = 1.0
    for epoch in range(epochs):
        idx = np.random.randint(0, len(dataset) - seq_len)
        batch_data = dataset[idx : idx + seq_len]
        loss, rec, cons = trainer.train_step(
            batch_data, 
            sim.dt, 
            epoch=epoch, 
            max_epochs=epochs
        )
        last_loss = loss
        scheduler.step(loss)

        if epoch % 100 == 0:
            progress = (epoch / epochs) * 100
            stats_tracker = trainer.loss_tracker.get_stats()
            active_nodes = int(model.encoder.pooling.active_mask.sum().item())
            log_str = f"Progress: {progress:3.0f}% | Loss: {stats_tracker.get('total', 0):.4f} | "
            log_str += f"Rec: {stats_tracker.get('rec_raw', 0):.4f} | "
            log_str += f"Cons: {stats_tracker.get('cons_raw', 0):.4f} | "
            log_str += f"LVar: {stats_tracker.get('lvars_mean', 0):.2f} | "
            log_str += f"Active: {active_nodes} | "
            log_str += f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e}"
            print(log_str)

    # 6. Latent Space Analysis
    print("\n--- 6. Latent Space Analysis ---")
    corrs = analyze_latent_space(model, dataset, pos, device=device)
    for k in range(n_super_nodes):
        max_corr = np.max(np.abs(corrs[k]))
        print(f"Super-node {k} max CoM correlation: {max_corr:.3f}")
        if max_corr > 0.8:
            print(f"  -> Strong physical mapping detected for super-node {k}")

    # 7. Enhanced Symbolic Distillation
    print("\n--- 7. Enhanced Symbolic Distillation ---")
    is_hamiltonian = model.hamiltonian
    latent_data = extract_latent_data(model, dataset, sim.dt, include_hamiltonian=is_hamiltonian)

    if len(latent_data[0]) == 0:
        print("Error: No valid latent data extracted (all NaN or divergent). Skipping symbolic distillation.")
        return

    # Use enhanced distiller with learnable basis functions
    print("Initializing enhanced symbolic distiller with learnable basis functions...")
    distiller = EnhancedSymbolicDistiller(
        populations=sc['populations'],
        generations=sc['generations'],
        stopping_criteria=sc['stopping_criteria'],
        max_features=sc['max_features'],
        secondary_optimization=sc['secondary_optimization'],
        opt_method=sc['opt_method'],
        opt_iterations=sc['opt_iterations'],
        use_sindy_pruning=sc['use_sindy_pruning'],
        sindy_threshold=sc['sindy_threshold']
    )
    
    # Replace the default transformer with enhanced one
    distiller.transformer = EnhancedFeatureTransformer(
        n_super_nodes, 
        latent_dim,
        use_learnable_bases=mc['use_learnable_bases'],
        basis_hidden_dim=mc['basis_hidden_dim'],
        num_learnable_bases=mc['num_learnable_bases'],
        use_attention=mc['use_attention'],
        use_adaptive_selection=mc['use_adaptive_selection']
    )

    if is_hamiltonian:
        z_states, dz_states, t_states, h_states = latent_data
        print("Distilling Hamiltonian H(q, p) with enhanced features...")
        equations = distiller.distill(
            z_states, 
            h_states, 
            n_super_nodes, 
            latent_dim, 
            box_size=box_size
        )
        confidences = distiller.confidences
        
        # Update trainer with robust symbolic proxy if confidence is high enough
        if confidences[0] > 0.5:
            print("Creating robust symbolic proxy...")
            robust_proxy = create_robust_symbolic_proxy(
                equations, 
                distiller.feature_masks, 
                distiller.transformer,
                confidence_threshold=0.5, 
                use_adaptive=True
            )
            trainer.update_symbolic_proxy(
                robust_proxy, 
                weight=0.1, 
                confidence=confidences[0]
            )
    else:
        z_states, dz_states, t_states = latent_data
        print("Distilling derivatives dZ/dt with enhanced features...")
        equations = distiller.distill(
            z_states, 
            dz_states, 
            n_super_nodes, 
            latent_dim, 
            box_size=box_size
        )
        confidences = distiller.confidences
        avg_conf = np.mean(confidences)
        if avg_conf > 0.5:
            print("Creating robust symbolic proxy...")
            robust_proxy = create_robust_symbolic_proxy(
                equations, 
                distiller.feature_masks, 
                distiller.transformer,
                confidence_threshold=0.5, 
                use_adaptive=True
            )
            trainer.update_symbolic_proxy(
                robust_proxy, 
                weight=0.1, 
                confidence=avg_conf
            )

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

    # 8. Visualization & Integration
    if is_hamiltonian and equations[0] is None:
        print("Skipping visualization due to lack of discovered Hamiltonian.")
        return

    print("\n--- 8. Visualizing Results ---")
    from scipy.integrate import odeint

    dyn_fn = OptimizedSymbolicDynamics(
        distiller, 
        equations, 
        distiller.feature_masks, 
        is_hamiltonian, 
        n_super_nodes, 
        latent_dim
    )

    # Integrate the discovered equations
    z0 = z_states[0]
    t_eval = np.linspace(0, (len(z_states)-1)*sim.dt, len(z_states))
    z_simulated = odeint(dyn_fn, z0, t_eval)

    model.eval()
    with torch.no_grad():
        test_idx = 0
        data = dataset[test_idx]
        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        z, s, _, _ = model.encode(x, edge_index, batch)
        recon = model.decode(z, s, batch).cpu().numpy()

    plt.figure(figsize=(18, 5))

    # 1. Micro Plot: Reconstruction
    plt.subplot(1, 3, 1)
    plt.scatter(data.x.cpu().numpy()[:, 0], data.x.cpu().numpy()[:, 1], c='blue', alpha=0.5, label='Truth')
    plt.scatter(recon[:, 0], recon[:, 1], c='red', marker='x', label='Recon')
    plt.title("Micro: Reconstruction")
    plt.legend()

    # 2. Assignment Plot
    plt.subplot(1, 3, 2)
    assignments = torch.argmax(s, dim=1).cpu().numpy()
    plt.scatter(data.x.cpu().numpy()[:, 0], data.x.cpu().numpy()[:, 1], c=assignments, cmap='viridis')
    plt.title("Hierarchical: Assignments")

    # 3. Meso Plot: Symbolic Integration vs Learned Latent
    plt.subplot(1, 3, 3)
    # Plot first few dimensions for clarity
    n_plot = min(4, z_states.shape[1])
    for i in range(n_plot):
        plt.plot(t_eval, z_states[:, i], 'k--', alpha=0.3, label=f'Learned Z_{i}' if i==0 else "")
        plt.plot(t_eval, z_simulated[:, i], label=f'Symbolic Z_{i}')
    plt.title("Meso: Symbolic Integration")
    plt.legend()

    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config['system']['results_dir'], f"discovery_result_{timestamp}.png")
    plt.savefig(result_path)
    print(f"Result saved to {result_path}")

    print("\n" + "="*80)
    print("ENHANCED PIPELINE EXECUTION COMPLETED")
    print(f"Final Training Loss: {last_loss:.6f}")
    print(f"Reconstruction Loss: {rec:.6f}")
    print(f"Discovered {sum(1 for eq in equations if eq is not None)}/{len(equations)} symbolic equations")
    print("="*80)


if __name__ == "__main__":
    main()