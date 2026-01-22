import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

def plot_discovery_results(model, dataset, pos_raw, s, z_states, assignments, output_path='discovery_result.png'):
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

    # 3. Latent Trajectories
    n_plot = min(4, z_states.shape[1])
    for k in range(n_plot):
        axes[1, 0].plot(z_states[:, k, 0], label=f'Node {k} - q1')
        axes[1, 0].plot(z_states[:, k, 1], '--', label=f'Node {k} - q2')
    axes[1, 0].set_title(f"Latent Trajectories (First {n_plot} nodes)")
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].legend(fontsize='small', ncol=2)

    # 4. Energy Conservation or Phase Space
    if z_states.shape[2] >= 4: # Hamiltonian (q, p)
        for k in range(n_plot):
            axes[1, 1].plot(z_states[:, k, 0], z_states[:, k, 2], label=f'Node {k}')
        axes[1, 1].set_title("Latent Phase Space (q vs p)")
        axes[1, 1].set_xlabel("q")
        axes[1, 1].set_ylabel("p")
    else:
        for k in range(n_plot):
            axes[1, 1].plot(z_states[:, k, 0], z_states[:, k, 1], label=f'Node {k}')
        axes[1, 1].set_title("Latent Trajectories (q1 vs q2)")
        axes[1, 1].set_xlabel("q1")
        axes[1, 1].set_ylabel("q2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Results visualization saved to {output_path}")

def plot_training_history(loss_tracker, output_path='training_history.png'):
    stats = loss_tracker.get_stats()
    plt.figure(figsize=(10, 6))
    for k, v in stats.items():
        if not k.startswith('w_') and k != 'total':
            plt.plot(v, label=k)
    plt.title("Training Loss Components")
    plt.yscale('log')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
