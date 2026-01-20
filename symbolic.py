from gplearn.genetic import SymbolicRegressor
import numpy as np
import torch

class SymbolicDistiller:
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001):
        self.max_pop = populations
        self.max_gen = generations
        self.stopping_criteria = stopping_criteria

    def _get_regressor(self, pop, gen):
        return SymbolicRegressor(population_size=pop,
                                 generations=gen, 
                                 stopping_criteria=self.stopping_criteria,
                                 function_set=('add', 'sub', 'mul', 'div', 'neg', 'sin', 'cos', 'sqrt', 'log', 'abs'),
                                 p_crossover=0.7, p_subtree_mutation=0.1, # Shifted towards crossover
                                 p_hoist_mutation=0.05, p_point_mutation=0.1,
                                 max_samples=0.9, verbose=0, 
                                 parsimony_coefficient=0.1, # Increased to force simpler, physics-like laws
                                 random_state=42)

    def distill(self, latent_states, latent_derivs, times=None):
        # Standardization (Z-score normalization)
        z_mean = latent_states.mean(axis=0)
        z_std = latent_states.std(axis=0) + 1e-6
        dz_mean = latent_derivs.mean(axis=0)
        dz_std = latent_derivs.std(axis=0) + 1e-6

        X = (latent_states - z_mean) / z_std
        Y = (latent_derivs - dz_mean) / dz_std

        # Feature Engineering: Add squared terms and interaction terms
        n_features = X.shape[1]
        features = [X]
        for i in range(n_features):
            features.append((X[:, i]**2).reshape(-1, 1))
            for j in range(i + 1, n_features):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))
        
        X_poly = np.hstack(features)

        if times is not None:
            # Normalize times as well if provided
            t_mean = times.mean()
            t_std = times.std() + 1e-6
            t_norm = (times - t_mean) / t_std
            X_poly = np.column_stack([X_poly, t_norm])

        equations = []
        for i in range(latent_derivs.shape[1]):
            # Start with 'Coarse' search (1/4 resources)
            coarse_pop = self.max_pop // 4
            coarse_gen = self.max_gen // 2
            
            print(f"Distilling dz_{i}/dt (Coarse search: pop={coarse_pop}, gen={coarse_gen})...")
            est = self._get_regressor(coarse_pop, coarse_gen)
            est.fit(X_poly, Y[:, i])
            
            # Check fit quality (if possible via gplearn score)
            if est.score(X_poly, Y[:, i]) > 0.95:
                print(f"  -> Good fit found in coarse search (R^2={est.score(X_poly, Y[:, i]):.3f})")
                equations.append(est._program)
            else:
                print(f"  -> Low fit (R^2={est.score(X_poly, Y[:, i]):.3f}). Escalating to Fine search...")
                est = self._get_regressor(self.max_pop, self.max_gen)
                est.fit(X_poly, Y[:, i])
                equations.append(est._program)
                
        return equations, (z_mean, z_std, dz_mean, dz_std)

def extract_latent_data(model, dataset, dt, stats=None):
    model.eval()
    latent_states = []
    latent_derivs = []
    physical_states = []
    times = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            current_t = i * dt

            # 1. Latent State Extraction: Ensure explicit flattening to [n_super_nodes * latent_dim]
            z, s, _ = model.encode(x, edge_index, torch.zeros(x.size(0), dtype=torch.long, device=device))
            z_flat = z.view(z.size(0), -1)[0].cpu().numpy()

            # 2. Latent Derivative from Neural ODE function
            ode_device = next(model.ode_func.parameters()).device
            t_ode = torch.tensor([current_t], dtype=torch.float32, device=ode_device)
            z_for_ode = z.view(z.size(0), -1).to(ode_device)
            dz = model.ode_func(t_ode, z_for_ode)
            dz_flat = dz.view(dz.size(0), -1)[0].cpu().numpy()

            latent_states.append(z_flat)
            latent_derivs.append(dz_flat)
            times.append(current_t)
            
            # 3. Physical Aggregates (Anchors for interpretability)
            if stats is not None:
                # Denormalize positions and velocities for physical calculation
                pos_norm = data.x[:, :2].cpu().numpy()
                vel_norm = data.x[:, 2:].cpu().numpy()
                pos = pos_norm * stats['pos_std'] + stats['pos_mean']
                vel = vel_norm * stats['vel_std'] + stats['vel_mean']
                
                # Total Kinetic Energy (0.5 * m * v^2, m=1.0)
                ke = 0.5 * np.sum(vel**2)
                
                # Total Potential Energy (0.5 * k * (r - r0)^2, k=15.0, r0=1.0)
                pe = 0
                if edge_index.numel() > 0:
                    edges = edge_index.cpu().numpy()
                    mask = edges[0] < edges[1] # Avoid double counting
                    idx1, idx2 = edges[0][mask], edges[1][mask]
                    dist = np.linalg.norm(pos[idx1] - pos[idx2], axis=1)
                    pe = 0.5 * 15.0 * np.sum((dist - 1.0)**2)
                
                # Center of Mass Velocity Magnitude
                com_vel_mag = np.linalg.norm(np.mean(vel, axis=0))
                
                physical_states.append([ke + pe, com_vel_mag])

    z_array = np.array(latent_states)
    dz_array = np.array(latent_derivs)
    
    if stats is not None and len(physical_states) > 0:
        p_array = np.array(physical_states)
        # Compute numerical derivatives for physical targets
        dp_array = np.zeros_like(p_array)
        dp_array[:-1] = (p_array[1:] - p_array[:-1]) / dt
        dp_array[-1] = dp_array[-2] # Simple boundary padding
        
        # Append physical variables as additional targets (dz_4/dt, dz_5/dt etc.)
        z_array = np.column_stack([z_array, p_array])
        dz_array = np.column_stack([dz_array, dp_array])

    return z_array, dz_array, np.array(times)
