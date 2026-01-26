import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# 1. MINIMAL SIMULATOR (Lennard-Jones & Spring)
class PhysicsSim:
    def __init__(self, n=8, mode='lj'):
        self.n, self.mode, self.dt = n, mode, 0.01
        self.pos = np.random.rand(n, 2) * 5
        self.vel = np.random.randn(n, 2) * 0.5
    
    def compute_forces(self, pos):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6
        if self.mode == 'spring':
            f = -10.0 * (dist - 1.0) * (diff / dist)
        else: # Lennard-Jones
            f = 24 * 1.0 * (2 * (1.0/dist)**13 - (1.0/dist)**7) * (diff / dist)
        return np.sum(np.nan_to_num(f), axis=1)

    def generate(self, steps=200):
        traj_p, traj_v = [], []
        for _ in range(steps):
            f = self.compute_forces(self.pos)
            self.vel += f * self.dt
            self.pos += self.vel * self.dt
            traj_p.append(self.pos.copy()); traj_v.append(self.vel.copy())
        return torch.tensor(np.array(traj_p), dtype=torch.float32), torch.tensor(np.array(traj_v), dtype=torch.float32)

# 2. CORE ARCHITECTURE
class EquiLayer(MessagePassing):
    def __init__(self, input_dim, output_dim=16):
        super().__init__(aggr='mean')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.f = nn.Sequential(nn.Linear(input_dim*2 + 1, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))
    def forward(self, x, pos, edge_index):
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        return self.propagate(edge_index, x=x, dist=dist)
    def message(self, x_i, x_j, dist):
        return self.f(torch.cat([x_i, x_j, dist], dim=-1))

class DiscoveryNet(nn.Module):
    def __init__(self, k=2, particle_latent_dim=4):  # Each particle has 4 latent dimensions (2 pos + 2 mom)
        super().__init__()
        self.k = k  # Number of particles/components
        self.d = particle_latent_dim  # Dimensions per particle
        self.total_latent_dim = k * particle_latent_dim  # Total dimensions
        self.enc = EquiLayer(input_dim=4, output_dim=16)  # Input: 4 (pos+vel), Output: 16 (internal representation)
        self.pool = nn.Linear(16, k)
        self.to_z = nn.Linear(k * 16, self.total_latent_dim)  # Changed from 16 to k*16=2*16=32, output total_latent_dim
        self.V_net = nn.Sequential(nn.Linear(k * (particle_latent_dim//2), 32), nn.Softplus(), nn.Linear(32, 1))
        self.dec = nn.Sequential(nn.Linear(self.total_latent_dim, 16), nn.SiLU(), nn.Linear(16, 8*4))  # Output 32 features for 8 particles

    def encode(self, x, pos, edge_index, batch):
        h = torch.relu(self.enc(x, pos, edge_index))
        s = torch.softmax(self.pool(h), dim=-1)
        # The scatter operation creates [num_batches, k, hidden_dim] = [1, 2, 16]
        # We need to reshape or aggregate to get the final latent representation
        pooled_h = scatter(s.unsqueeze(-1) * h.unsqueeze(1), batch, dim=0, reduce='sum')  # [1, 2, 16]
        # Reshape to [1, 2*16] then apply a transformation to get [1, latent_dim]
        pooled_flat = pooled_h.view(pooled_h.size(0), -1)  # [1, 2*16] = [1, 32]
        # Apply a linear transformation to get the desired latent dimension
        z = self.to_z(pooled_flat)  # This needs adjustment since to_z expects input size 16, not 32
        return z, s

    def get_force(self, z):
        z = z.requires_grad_(True)
        q = z.view(z.size(0), self.k, self.d)[:, :, :self.d//2].reshape(z.size(0), -1)
        V = self.V_net(q)
        dVdq = torch.autograd.grad(V.sum(), z, create_graph=True)[0]
        # Hamiltonian: dq/dt = p, dp/dt = -dV/dq
        p = z.view(z.size(0), self.k, 2, self.d//2)[:, :, 1].reshape(z.size(0), -1)
        dqdt = p.reshape(z.size(0), -1)
        dpdt = -dVdq.view(z.size(0), self.k, 2, self.d//2)[:, :, 0].reshape(z.size(0), -1)
        return torch.cat([dqdt, dpdt], dim=-1), V

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    sim = PhysicsSim(mode=mode)
    p, v = sim.generate(400)
    model = DiscoveryNet(k=2, particle_latent_dim=4)  # Each particle has 4 dimensions (2 pos + 2 mom)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Prepare Graph Data
    edge_index = torch.combinations(torch.arange(8)).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    print(f"Training Neural-Hamiltonian for {mode}...")
    for epoch in range(300):
        idx = np.random.randint(0, 350)
        x = torch.cat([p[idx], v[idx]], dim=-1)
        z, s = model.encode(x, p[idx], edge_index, torch.zeros(8, dtype=torch.long))
        
        # Reconstruction Loss
        # Decode the single latent vector to reconstruct all particles
        decoded_features = model.dec(z)
        decoded = decoded_features.view(8, 4)  # Reshape to [8, 4]
        l_rec = torch.nn.functional.mse_loss(decoded, x)
        
        # Hamiltonian Consistency Loss
        dz_dt, V = model.get_force(z)
        z_next, _ = model.encode(torch.cat([p[idx+1], v[idx+1]], dim=-1), p[idx+1], edge_index, torch.zeros(8, dtype=torch.long))
        l_dyn = torch.nn.functional.mse_loss(z + dz_dt * sim.dt, z_next)
        
        loss = l_rec + l_dyn * 0.1
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 50 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

    # 4. SYMBOLIC DISTILLATION
    print("\nDistilling Symbolic Law...")
    model.eval()
    # Process trajectory data to collect inputs for symbolic regression
    all_x = torch.cat([p, v], dim=-1)
    z_list = []

    # Process each time step separately (keeping gradients for symbolic regression)
    for t_idx in range(min(100, p.size(0))):  # Limit for efficiency
        x_t = all_x[t_idx]  # Shape: [8, 4]
        p_t = p[t_idx]      # Shape: [8, 2]
        z_t, _ = model.encode(x_t, p_t, edge_index, torch.zeros(8, dtype=torch.long))
        z_list.append(z_t.detach())  # Detach for storage but compute V with fresh graph

    # Now compute potentials with gradients enabled
    q_list = []
    V_list = []
    for t_idx in range(len(z_list)):
        # Recompute with gradient tracking for symbolic regression
        x_t = all_x[t_idx]  # Shape: [8, 4]
        p_t = p[t_idx]      # Shape: [8, 2]
        z_t, _ = model.encode(x_t, p_t, edge_index, torch.zeros(8, dtype=torch.long))
        z_t.requires_grad_(True)  # Enable gradients for symbolic regression

        _, V_t = model.get_force(z_t)
        # Extract position components for symbolic regression
        q_t = z_t.view(1, model.k, model.d)[:, :, :model.d//2].reshape(1, -1)

        q_list.append(q_t.detach().cpu().numpy())
        V_list.append(V_t.detach().cpu().numpy())

    q_all = np.vstack(q_list).squeeze()
    y = np.hstack(V_list).squeeze()

    # Physics Basis Functions
    inv2 = make_function(function=lambda x: 1.0/(x**2 + 1e-4), name='inv2', arity=1)
    inv6 = make_function(function=lambda x: 1.0/(x**6 + 1e-4), name='inv6', arity=1)

    est = SymbolicRegressor(population_size=1000, generations=20,
                            function_set=('add', 'sub', 'mul', 'div', inv2, inv6),
                            parsimony_coefficient=0.01, verbose=1)
    est.fit(q_all, y)

    print(f"\nDiscovered Potential V(q): {est._program}")

    # Sympy Cleanup - Handle custom functions properly
    try:
        # Convert the expression to a form that Sympy can handle
        expr_str = str(est._program)
        # Replace custom functions with standard sympy notation
        expr_str = expr_str.replace('inv2(', '1/(').replace('**2)', '**2)')
        expr_str = expr_str.replace('inv6(', '1/(').replace('**6)', '**6)')

        X0, X1, X2, X3 = sp.symbols('q1_x q1_y q2_x q2_y')
        # Replace X variables with proper symbols
        expr_str = expr_str.replace('X0', 'q1_x').replace('X1', 'q1_y').replace('X2', 'q2_x').replace('X3', 'q2_y')

        expr = sp.sympify(expr_str)
        print(f"Simplified Physics: {sp.simplify(expr)}")
    except Exception as e:
        print(f"Could not simplify expression with Sympy: {e}")
        print("Raw discovered expression:", est._program)

if __name__ == "__main__":
    train_discovery(mode='lj')
