"""
Physics Benchmark Suite - The "Blind Physicist"
Performs automated numerical validation of discovered symbolic physical laws.
"""

import torch
import numpy as np
import json
import os
from model import DiscoveryEngineModel
from engine import SymbolicProxy, prepare_data
from simulator import SpringMassSimulator
from enhanced_symbolic import gp_to_sympy
import sympy as sp

def calculate_energy_drift(proxy, z0, steps=1000, dt=0.01):
    """Calculates max relative energy drift over a long shadow integration."""
    device = torch.device('cpu') # Use CPU for long integration stability
    proxy = proxy.to(device)
    z0 = z0.to(device)
    
    # We need a way to calculate H from proxy if it's Hamiltonian
    # If not Hamiltonian, we might not have a well-defined energy
    # But for this task, we assume the discovered system should conserve something
    
    z = z0
    energies = []
    
    # If proxy has hamiltonian method, use it
    has_h = hasattr(proxy, 'is_hamiltonian') and proxy.is_hamiltonian
    
    def get_h(zt):
        if has_h:
            # For Hamiltonian proxy, the first sym_module is the Hamiltonian
            X_norm = proxy.torch_transformer(zt)
            H_norm = proxy.sym_modules[0](X_norm)
            H = proxy.torch_transformer.denormalize_y(H_norm)
            return H.item()
        else:
            # Fallback: sum of squares as a pseudo-energy
            return torch.sum(zt**2).item()

    curr_h = get_h(z)
    energies.append(curr_h)
    
    for _ in range(steps):
        # Simple Euler-Cromer or RK4 for shadow integration
        # Here we use the proxy's forward which computes dz/dt
        dz = proxy(z)
        z = z + dz * dt
        energies.append(get_h(z))
        
    energies = np.array(energies)
    initial_e = energies[0]
    if abs(initial_e) < 1e-9:
        drift = np.max(np.abs(energies - initial_e))
    else:
        drift = np.max(np.abs(energies - initial_e) / abs(initial_e))
        
    return float(drift)

def calculate_forecast_horizon(proxy, z_gt, dt=0.01, threshold=0.05):
    """Steps until MSE > threshold * variance of signal."""
    # z_gt: [T, B, K, D]
    T = z_gt.shape[0]
    variance = torch.var(z_gt).item()
    mse_threshold = threshold * variance
    
    z = z_gt[0].unsqueeze(0)
    horizon = T
    
    for t in range(1, T):
        dz = proxy(z)
        z = z + dz * dt
        mse = torch.mean((z - z_gt[t].unsqueeze(0))**2).item()
        if mse > mse_threshold:
            horizon = t
            break
            
    return int(horizon)

def calculate_mass_consistency(model, dataset):
    """Pearson correlation between learned super-node mass and assignment sums."""
    model.eval()
    device = next(model.parameters()).device
    
    # Get learned masses from HamiltonianODEFunc
    if hasattr(model.ode_func, 'get_masses'):
        learned_masses = model.ode_func.get_masses().detach().cpu().numpy().flatten()
    else:
        return 0.0
        
    # Get assignment sums S_k = sum_i s_ik
    all_s_sums = []
    with torch.no_grad():
        for data in dataset[:10]: # Sample 10 steps
            data = data.to(device)
            _, s, _, _ = model.encode(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long, device=device))
            all_s_sums.append(s.sum(dim=0).cpu().numpy())
            
    avg_s_sums = np.mean(all_s_sums, axis=0)
    
    if len(learned_masses) != len(avg_s_sums):
        return 0.0
        
    corr = np.corrcoef(learned_masses, avg_s_sums)[0, 1]
    return float(np.nan_to_num(corr))

def calculate_parsimony_index(equations, r2_scores):
    """Parsimony index = Mean(R^2 / Number of Symbolic Nodes)."""
    indices = []
    for eq, r2 in zip(equations, r2_scores):
        if eq is None: continue
        # Count nodes in sympy expression
        try:
            expr = gp_to_sympy(str(eq))
            n_nodes = len(expr.args) + 1 # Rough estimate
            # More accurate node count
            n_nodes = sum(1 for _ in sp.preorder_traversal(expr))
            indices.append(r2 / max(1, n_nodes))
        except:
            indices.append(r2 / 10.0) # Penalty for unparseable
            
    return float(np.mean(indices)) if indices else 0.0

def calculate_lyapunov_exponent(proxy, z0, steps=1000, dt=0.01, epsilon=1e-6):
    """
    Calculates the largest Lyapunov exponent of the discovered symbolic law.
    A positive exponent indicates chaos/instability.
    Near-zero or negative indicates stability/conservative behavior.
    """
    device = torch.device('cpu')
    proxy = proxy.to(device)
    z0 = z0.to(device).view(1, -1)
    
    z1 = z0.clone()
    # Perturb z0 in a random direction
    v = torch.randn_like(z0)
    v = v / torch.norm(v)
    z2 = z1 + epsilon * v
    
    lyapunov_sum = 0.0
    
    for _ in range(steps):
        # Evolve both trajectories
        with torch.no_grad():
            dz1 = proxy(z1)
            z1_next = z1 + dz1 * dt
            
            dz2 = proxy(z2)
            z2_next = z2 + dz2 * dt
            
            # Distance after evolution
            dist_next = torch.norm(z1_next - z2_next).item()
            
            # Accumulate growth rate
            if dist_next > 0:
                lyapunov_sum += np.log(dist_next / epsilon)
                
                # Re-normalize z2 to maintain epsilon distance from z1 in the same direction
                z2 = z1_next + epsilon * (z2_next - z1_next) / dist_next
                z1 = z1_next
            else:
                # If they collapse, it's very stable
                lyapunov_sum += -10.0 # Arbitrary negative penalty
                z1 = z1_next
                z2 = z1 + epsilon * v
                
    return float(lyapunov_sum / (steps * dt))

def run_benchmark(model, equations, r2_scores, transformer, test_data_path=None):
    """Runs the full benchmark suite and saves validation_report.json."""
    print("Running Physics Benchmark...")
    
    # 1. Setup Proxy
    proxy = SymbolicProxy(
        model.encoder.n_super_nodes,
        model.encoder.latent_dim,
        equations,
        transformer,
        hamiltonian=model.hamiltonian
    )
    
    # 2. Generate OOD test trajectory
    # Use same number of particles as model was trained on
    n_particles = getattr(model.encoder, 'n_particles', 16)
    sim = SpringMassSimulator(n_particles=n_particles)
    pos, vel = sim.generate_trajectory(steps=1000)
    # Scale velocities to increase energy
    vel_ood = vel * 1.414 # sqrt(2) for 2x kinetic energy
    pos_ood, vel_ood = sim.generate_trajectory(steps=1000, init_pos=pos[0], init_vel=vel_ood[0])
    
    test_dataset, _ = prepare_data(pos_ood, vel_ood, stats=None) # We'll use model's internal normalization if needed
    
    # 3. Perform evaluations
    # Get ground truth latent trajectory
    model.eval()
    device = next(model.parameters()).device
    z_gt_list = []
    with torch.no_grad():
        for data in test_dataset:
            data = data.to(device)
            z, _, _, _ = model.encode(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long, device=device))
            # z: [1, K, D] -> append [K*D]
            z_gt_list.append(z.view(-1).detach())
    
    z_gt = torch.stack(z_gt_list).cpu() # [T, K*D] on CPU for stability
    
    # Ensure z0 has batch dimension [1, K*D]
    z0 = z_gt[0].unsqueeze(0)
    
    # proxy is moved to CPU inside these functions
    energy_drift = calculate_energy_drift(proxy, z0, steps=5000, dt=sim.dt)
    forecast_horizon = calculate_forecast_horizon(proxy, z_gt, dt=sim.dt)
    mass_consistency = calculate_mass_consistency(model, test_dataset)
    parsimony = calculate_parsimony_index(equations, r2_scores)
    lyapunov = calculate_lyapunov_exponent(proxy, z0, steps=200, dt=sim.dt)
    
    # 4. Symbolic R2 (OOD)
    # We evaluate how well the proxy predicts dz/dt on OOD data
    # Ensure proxy is on CPU
    proxy = proxy.to(torch.device('cpu'))
    z_flat = z_gt.view(z_gt.size(0), -1)
    dz_gt = (z_flat[1:] - z_flat[:-1]) / sim.dt
    dz_pred = proxy(z_flat[:-1])
    
    from sklearn.metrics import r2_score
    r2_ood = r2_score(dz_gt.cpu().numpy(), dz_pred.detach().cpu().numpy())
    
    report = {
        "energy_conservation_error": energy_drift,
        "forecast_horizon": forecast_horizon,
        "mass_consistency": mass_consistency,
        "parsimony_index": parsimony,
        "symbolic_r2_ood": float(r2_ood),
        "lyapunov_exponent": lyapunov,
        "success": bool(r2_ood > 0.98 and energy_drift < 1e-6 and lyapunov < 0.1)
    }
    
    with open('validation_report.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    print(f"Benchmark complete. Success: {report['success']}")
    print(f"  R2 (OOD): {r2_ood:.4f}")
    print(f"  Energy Drift: {energy_drift:.2e}")
    print(f"  Mass Consistency: {mass_consistency:.4f}")
    
    return report

if __name__ == "__main__":
    # Example usage if called as script
    # This would require a trained model and equations
    pass
