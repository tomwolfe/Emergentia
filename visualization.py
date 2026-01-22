import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

def plot_discovery_results(model, dataset, pos_raw, s, z_states, assignments, output_path='discovery_result.png', symbolic_predictions=None):
    """
    Comprehensive visualization of the discovery results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Spatial Assignments
    scatter = axes[0, 0].scatter(pos_raw[0, :, 0], pos_raw[0, :, 1], c=assignments, cmap='tab10', s=50)
    axes[0, 0].set_title("Initial Particle Assignments (Meso-nodes)")
    axes[0, 0].set_aspect('equal')
    plt.colorbar(scatter, ax=axes[0, 0], label='Super-node Index')

    # 2. Assignment Confidence (Heatmap)
    im = axes[0, 1].imshow(s.detach().cpu().numpy().T, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title("Assignment Weights (S) over Time")
    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].set_ylabel("Super-node index")
    plt.colorbar(im, ax=axes[0, 1])

    # 3. Latent Trajectories with symbolic predictions overlay
    n_plot = min(4, z_states.shape[1])

    # Fix tensor size mismatch by checking dimensions
    if symbolic_predictions is not None:
        # Ensure symbolic_predictions has compatible dimensions
        if z_states.shape[0] != symbolic_predictions.shape[0]:
            # Interpolate or truncate to match time dimension
            min_time = min(z_states.shape[0], symbolic_predictions.shape[0])
            z_states = z_states[:min_time]
            symbolic_predictions = symbolic_predictions[:min_time]

    for k in range(n_plot):
        axes[1, 0].plot(z_states[:, k, 0], label=f'Node {k} - q1 (Learned)', alpha=0.7)
        axes[1, 0].plot(z_states[:, k, 1], '--', label=f'Node {k} - q2 (Learned)', alpha=0.7)

        # Overlay symbolic predictions if available and compatible
        if symbolic_predictions is not None and k < symbolic_predictions.shape[1]:
            # Ensure we don't exceed the time dimension
            time_dim = min(z_states.shape[0], symbolic_predictions.shape[0])
            axes[1, 0].plot(symbolic_predictions[:time_dim, k, 0],
                           label=f'Node {k} - q1 (Symbolic)', linestyle='-.', alpha=0.8)
            axes[1, 0].plot(symbolic_predictions[:time_dim, k, 1],
                           label=f'Node {k} - q2 (Symbolic)', linestyle=':', alpha=0.8)

    axes[1, 0].set_title(f"Latent Trajectories (First {n_plot} nodes)")
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].legend(fontsize='small', ncol=2)

    # 4. Energy Conservation or Phase Space
    if z_states.shape[2] >= 4: # Hamiltonian (q, p)
        for k in range(n_plot):
            axes[1, 1].plot(z_states[:, k, 0], z_states[:, k, 2], label=f'Node {k} (Learned)', alpha=0.7)

            # Overlay symbolic predictions if available and compatible
            if symbolic_predictions is not None and k < symbolic_predictions.shape[1]:
                time_dim = min(z_states.shape[0], symbolic_predictions.shape[0])
                axes[1, 1].plot(symbolic_predictions[:time_dim, k, 0],
                               symbolic_predictions[:time_dim, k, 2],
                               label=f'Node {k} (Symbolic)', linestyle='--', alpha=0.8)

        axes[1, 1].set_title("Latent Phase Space (q vs p)")
        axes[1, 1].set_xlabel("q")
        axes[1, 1].set_ylabel("p")
    else:
        for k in range(n_plot):
            axes[1, 1].plot(z_states[:, k, 0], z_states[:, k, 1], label=f'Node {k} (Learned)', alpha=0.7)

            # Overlay symbolic predictions if available and compatible
            if symbolic_predictions is not None and k < symbolic_predictions.shape[1]:
                time_dim = min(z_states.shape[0], symbolic_predictions.shape[0])
                axes[1, 1].plot(symbolic_predictions[:time_dim, k, 0],
                               symbolic_predictions[:time_dim, k, 1],
                               label=f'Node {k} (Symbolic)', linestyle='--', alpha=0.8)

        axes[1, 1].set_title("Latent Trajectories (q1 vs q2)")
        axes[1, 1].set_xlabel("q1")
        axes[1, 1].set_ylabel("q2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Results visualization saved to {output_path}")

def plot_training_history(loss_tracker, output_path='training_history.png'):
    stats = loss_tracker.get_stats()
    plt.figure(figsize=(12, 8))
    
    # Plot raw losses
    plt.subplot(2, 1, 1)
    for k, v in stats.items():
        if not k.startswith('w_') and k != 'total' and not k.endswith('_raw'):
            if len(v) > 0:
                plt.plot(v, label=k, alpha=0.7)
    
    # Also plot some key _raw losses if they exist
    for k in ['rec_raw', 'cons_raw', 'hinge_raw']:
        if k in stats and len(stats[k]) > 0:
            plt.plot(stats[k], label=k, linestyle='--', alpha=0.5)

    plt.title("Training Loss Components (History)")
    plt.yscale('log')
    plt.xlabel("Update Step")
    plt.ylabel("Loss Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Plot weights
    plt.subplot(2, 1, 2)
    for k, v in stats.items():
        if k.startswith('w_'):
            if len(v) > 0:
                plt.plot(v, label=k)
    plt.title("Loss Balancing Weights")
    plt.yscale('log')
    plt.xlabel("Update Step")
    plt.ylabel("Weight Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {output_path}")
