import torch
import torch.nn as nn
from model import HamiltonianODEFunc, DiscoveryEngineModel as DiscoveryEngine
from torch_geometric.data import Data

def test_ode_stability():
    latent_dim = 4
    n_super_nodes = 2
    hidden_dim = 64
    
    # Create model
    model = DiscoveryEngine(
        n_particles=10,
        n_super_nodes=n_super_nodes,
        node_features=3,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        hamiltonian=True
    )
    
    # Random initial state
    z0 = torch.randn(2, n_super_nodes, latent_dim) # Batch size 2
    t = torch.linspace(0, 1, 10)
    
    print("Running forward_dynamics...")
    try:
        zt = model.forward_dynamics(z0, t)
        print(f"Success! Output shape: {zt.shape}")
    except Exception as e:
        print(f"Failed! Error: {e}")

if __name__ == '__main__':
    test_ode_stability()
