from gplearn.genetic import SymbolicRegressor
import numpy as np
import torch

class SymbolicDistiller:
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001):
        # Using a wider set of function to capture physical laws (e.g. 1/r^2 patterns)
        self.est = SymbolicRegressor(population_size=populations,
                                     generations=generations, 
                                     stopping_criteria=stopping_criteria,
                                     function_set=('add', 'sub', 'mul', 'div', 'sin', 'cos'),
                                     p_crossover=0.7, p_subtree_mutation=0.1,
                                     p_hoist_mutation=0.05, p_point_mutation=0.1,
                                     max_samples=0.9, verbose=1,
                                     parsimony_coefficient=0.005, random_state=0)

    def distill(self, latent_states, latent_derivs, times=None):
        """
        latent_states: [N_samples, latent_dim * n_super_nodes]
        latent_derivs: [N_samples, latent_dim * n_super_nodes]
        times: [N_samples] (optional)
        """
        X = latent_states
        if times is not None:
            # Add time as an additional input feature for non-autonomous laws
            X = np.column_stack([latent_states, times])

        equations = []
        for i in range(latent_derivs.shape[1]):
            print(f"Distilling equation for component {i} (dz_{i}/dt)...")
            self.est.fit(X, latent_derivs[:, i])
            equations.append(self.est._program)
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
