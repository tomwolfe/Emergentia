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
                                 function_set=('add', 'sub', 'mul', 'div', 'inv', 'neg', 'sin', 'cos'),
                                 p_crossover=0.7, p_subtree_mutation=0.1,
                                 p_hoist_mutation=0.05, p_point_mutation=0.1,
                                 max_samples=0.9, verbose=0, # Quiet by default
                                 parsimony_coefficient=0.005, random_state=0)

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
    
    with torch.no_grad():
        for i in range(len(dataset)):
            data = dataset[i]
            # Use the appropriate time for non-autonomous systems
            current_t = i * dt
            
            z, s = model.encode(data.x, data.edge_index, torch.zeros(data.x.size(0), dtype=torch.long))
            z_flat = z.view(-1).numpy()
            
            # Use the ODE function to get the derivative at this state and time
            t = torch.tensor([current_t], dtype=torch.float)
            dz = model.ode_func(t, z.view(1, -1))
            dz_flat = dz.view(-1).numpy()
            
            latent_states.append(z_flat)
            latent_derivs.append(dz_flat)
            times.append(current_t)
            
    return np.array(latent_states), np.array(latent_derivs), np.array(times)
