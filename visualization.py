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
            try:
                # Ensure we don't exceed the time dimension
                time_dim = min(z_states.shape[0], symbolic_predictions.shape[0])
                axes[1, 0].plot(symbolic_predictions[:time_dim, k, 0],
                               label=f'Node {k} - q1 (Symbolic)', linestyle='-.', alpha=0.8)
                axes[1, 0].plot(symbolic_predictions[:time_dim, k, 1],
                               label=f'Node {k} - q2 (Symbolic)', linestyle=':', alpha=0.8)
            except Exception as e:
                print(f"Warning: Failed to plot symbolic prediction (traj) for node {k}: {e}")

    axes[1, 0].set_title(f"Latent Trajectories (First {n_plot} nodes)")
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].legend(fontsize='small', ncol=2)

    # 4. Energy Conservation or Phase Space
    if z_states.shape[2] >= 4: # Hamiltonian (q, p)
        for k in range(n_plot):
            axes[1, 1].plot(z_states[:, k, 0], z_states[:, k, 2], label=f'Node {k} (Learned)', alpha=0.7)

            # Overlay symbolic predictions if available and compatible
            if symbolic_predictions is not None and k < symbolic_predictions.shape[1]:
                try:
                    time_dim = min(z_states.shape[0], symbolic_predictions.shape[0])
                    axes[1, 1].plot(symbolic_predictions[:time_dim, k, 0],
                                   symbolic_predictions[:time_dim, k, 2],
                                   label=f'Node {k} (Symbolic)', linestyle='--', alpha=0.8)
                except Exception as e:
                    print(f"Warning: Failed to plot symbolic prediction (phase q-p) for node {k}: {e}")

        axes[1, 1].set_title("Latent Phase Space (q vs p)")
        axes[1, 1].set_xlabel("q")
        axes[1, 1].set_ylabel("p")
    else:
        for k in range(n_plot):
            axes[1, 1].plot(z_states[:, k, 0], z_states[:, k, 1], label=f'Node {k} (Learned)', alpha=0.7)

            # Overlay symbolic predictions if available and compatible
            if symbolic_predictions is not None and k < symbolic_predictions.shape[1]:
                try:
                    time_dim = min(z_states.shape[0], symbolic_predictions.shape[0])
                    axes[1, 1].plot(symbolic_predictions[:time_dim, k, 0],
                                   symbolic_predictions[:time_dim, k, 1],
                                   label=f'Node {k} (Symbolic)', linestyle='--', alpha=0.8)
                except Exception as e:
                    print(f"Warning: Failed to plot symbolic prediction (phase q1-q2) for node {k}: {e}")

        axes[1, 1].set_title("Latent Trajectories (q1 vs q2)")
        axes[1, 1].set_xlabel("q1")
        axes[1, 1].set_ylabel("q2")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Results visualization saved to {output_path}")

def generate_closed_loop_trajectory(symbolic_proxy, z0, steps, dt, device='cpu'):
    """
    Generate a trajectory by iteratively applying the symbolic proxy (closed-loop).
    Uses high-quality RK45 integration from scipy for stability.
    """
    from scipy.integrate import solve_ivp
    
    symbolic_proxy.eval()
    # z0: [1, K*D]
    z0_np = z0.detach().cpu().numpy().flatten()
    
    def ode_func(t, y):
        # y: [K*D]
        y_torch = torch.from_numpy(y).float().to(device).unsqueeze(0)
        with torch.no_grad():
            dz_dt = symbolic_proxy(y_torch)
        return dz_dt.cpu().numpy().flatten()
    
    t_span = (0, (steps - 1) * dt)
    t_eval = np.linspace(0, (steps - 1) * dt, steps)
    
    try:
        # Use RK45 for robust integration of symbolic discovery results
        sol = solve_ivp(ode_func, t_span, z0_np, t_eval=t_eval, method='RK45', rtol=1e-3, atol=1e-6)
        
        # If integration fails or returns fewer steps than requested, pad or fallback
        if sol.status < 0 or sol.y.shape[1] < steps:
            print(f"Warning: solve_ivp failed or truncated (status={sol.status}). Falling back to Euler.")
            raise ValueError("solve_ivp failed")
            
        # sol.y: [K*D, T] -> [T, K*D]
        return sol.y.T
        
    except Exception as e:
        # Fallback to simple Euler if RK45 fails (common if symbolic laws are unstable)
        traj = [z0_np.reshape(1, -1)]
        curr_z = z0_np
        for _ in range(steps - 1):
            dz_dt = ode_func(0, curr_z)
            curr_z = curr_z + dz_dt * dt
            # Clip to prevent divergence in visualization
            curr_z = np.clip(curr_z, -1e3, 1e3)
            traj.append(curr_z.reshape(1, -1))
        return np.concatenate(traj, axis=0)

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

def plot_symbolic_pareto(candidates, output_path='symbolic_pareto.png'):
    """
    Visualize the Pareto front of symbolic expressions (Complexity vs. Accuracy).
    """
    if not candidates:
        return

    plt.figure(figsize=(10, 6))
    
    # Extract data
    complexities = [c['complexity'] for c in candidates]
    scores = [c['score'] for c in candidates]
    target_indices = [c.get('target_idx', 0) for c in candidates]
    
    # Color by target index if multiple targets
    unique_targets = sorted(list(set(target_indices)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_targets)))
    
    for idx, target in enumerate(unique_targets):
        t_complexities = [c['complexity'] for c in candidates if c.get('target_idx', 0) == target]
        t_scores = [c['score'] for c in candidates if c.get('target_idx', 0) == target]
        
        plt.scatter(t_complexities, t_scores, label=f'Target {target}', alpha=0.6, color=colors[idx])
        
        # Compute Pareto front for this target
        if len(t_complexities) > 1:
            points = sorted(zip(t_complexities, t_scores))
            pareto_front = [points[0]]
            for p in points[1:]:
                if p[1] > pareto_front[-1][1]:
                    pareto_front.append(p)
            
            px, py = zip(*pareto_front)
            plt.step(px, py, where='post', color=colors[idx], linestyle='--', alpha=0.4)

    plt.title("Symbolic Discovery Pareto Front")
    plt.xlabel("Expression Complexity (Nodes)")
    plt.ylabel("R^2 Score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Pareto front plot saved to {output_path}")
