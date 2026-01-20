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
                                 p_crossover=0.4, p_subtree_mutation=0.2,
                                 p_hoist_mutation=0.1, p_point_mutation=0.2,
                                 max_samples=0.9, verbose=0, 
                                 parsimony_coefficient=0.01, # Slightly reduced to allow more complex terms
                                 random_state=0)

    def distill(self, latent_states, latent_derivs, times=None):
        X = latent_states
        if times is not None:
            X = np.column_stack([latent_states, times])

        equations = []
        for i in range(latent_derivs.shape[1]):
            # Start with 'Coarse' search (1/4 resources)
            coarse_pop = self.max_pop // 4
            coarse_gen = self.max_gen // 2
            
            print(f"Distilling dz_{i}/dt (Coarse search: pop={coarse_pop}, gen={coarse_gen})...")
            est = self._get_regressor(coarse_pop, coarse_gen)
            est.fit(X, latent_derivs[:, i])
            
            # Check fit quality (if possible via gplearn score)
            # score is R^2. If > 0.95, it's likely good enough.
            if est.score(X, latent_derivs[:, i]) > 0.95:
                print(f"  -> Good fit found in coarse search (R^2={est.score(X, latent_derivs[:, i]):.3f})")
                equations.append(est._program)
            else:
                print(f"  -> Low fit (R^2={est.score(X, latent_derivs[:, i]):.3f}). Escalating to Fine search...")
                est = self._get_regressor(self.max_pop, self.max_gen)
                est.fit(X, latent_derivs[:, i])
                equations.append(est._program)
                
        return equations

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
            t = torch.tensor([current_t], dtype=torch.float32, device=device)
            dz = model.ode_func(t, z.view(1, -1))
            dz_flat = dz.view(-1).cpu().numpy()

            latent_states.append(z_flat)
            latent_derivs.append(dz_flat)
            times.append(current_t)

    return np.array(latent_states), np.array(latent_derivs), np.array(times)
