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
                                 parsimony_coefficient=0.05, # Increased to avoid trivial/overfit laws
                                 random_state=42)

    def distill(self, latent_states, latent_derivs, times=None):
        # Standardization (Z-score normalization)
        z_mean = latent_states.mean(axis=0)
        z_std = latent_states.std(axis=0) + 1e-6
        dz_mean = latent_derivs.mean(axis=0)
        dz_std = latent_derivs.std(axis=0) + 1e-6

        X = (latent_states - z_mean) / z_std
        Y = (latent_derivs - dz_mean) / dz_std

        if times is not None:
            # Normalize times as well if provided
            t_mean = times.mean()
            t_std = times.std() + 1e-6
            t_norm = (times - t_mean) / t_std
            X = np.column_stack([X, t_norm])

        equations = []
        for i in range(latent_derivs.shape[1]):
            # Start with 'Coarse' search (1/4 resources)
            coarse_pop = self.max_pop // 4
            coarse_gen = self.max_gen // 2
            
            print(f"Distilling dz_{i}/dt (Coarse search: pop={coarse_pop}, gen={coarse_gen})...")
            est = self._get_regressor(coarse_pop, coarse_gen)
            est.fit(X, Y[:, i])
            
            # Check fit quality (if possible via gplearn score)
            # score is R^2. If > 0.95, it's likely good enough.
            if est.score(X, Y[:, i]) > 0.95:
                print(f"  -> Good fit found in coarse search (R^2={est.score(X, Y[:, i]):.3f})")
                equations.append(est._program)
            else:
                print(f"  -> Low fit (R^2={est.score(X, Y[:, i]):.3f}). Escalating to Fine search...")
                est = self._get_regressor(self.max_pop, self.max_gen)
                est.fit(X, Y[:, i])
                equations.append(est._program)
                
        return equations, (z_mean, z_std, dz_mean, dz_std)

def extract_latent_data(model, dataset, dt):
    model.eval()
    latent_states = []
    latent_derivs = []
    times = []

    device = next(model.parameters()).device

    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            # Ensure data is on the correct device
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)

            current_t = i * dt

            z, s, _ = model.encode(x, edge_index, torch.zeros(x.size(0), dtype=torch.long, device=device))
            z_flat = z.view(-1).cpu().numpy()

            # Use the ODE function to get the derivative at this state and time
            # Handle potential device mismatch (e.g. if ode_func is on CPU for MPS compatibility)
            ode_device = next(model.ode_func.parameters()).device
            t = torch.tensor([current_t], dtype=torch.float32, device=ode_device)
            z_for_ode = z.view(1, -1).to(ode_device)
            
            dz = model.ode_func(t, z_for_ode)
            dz_flat = dz.view(-1).cpu().numpy()

            latent_states.append(z_flat)
            latent_derivs.append(dz_flat)
            times.append(current_t)

    return np.array(latent_states), np.array(latent_derivs), np.array(times)
