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
    def __init__(self, dim):
        super().__init__(aggr='mean')
        self.f = nn.Sequential(nn.Linear(dim*2 + 1, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, x, pos, edge_index):
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        return self.propagate(edge_index, x=x, dist=dist)
    def message(self, x_i, x_j, dist):
        return self.f(torch.cat([x_i, x_j, dist], dim=-1))

class DiscoveryNet(nn.Module):
    def __init__(self, k=2, latent_dim=4):
        super().__init__()
        self.k, self.d = k, latent_dim
        self.enc = EquiLayer(16)
        self.pool = nn.Linear(16, k)
        self.to_z = nn.Linear(16, latent_dim)
        self.V_net = nn.Sequential(nn.Linear(k * (latent_dim//2), 32), nn.Softplus(), nn.Linear(32, 1))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 16), nn.SiLU(), nn.Linear(16, 4))

    def encode(self, x, pos, edge_index, batch):
        h = torch.relu(self.enc(x, pos, edge_index))
        s = torch.softmax(self.pool(h), dim=-1)
        z = self.to_z(scatter(s.unsqueeze(-1) * h.unsqueeze(1), batch, dim=0, reduce='sum'))
        return z, s

    def get_force(self, z):
        z = z.requires_grad_(True)
        q = z.view(z.size(0), self.k, self.d)[:, :, :self.d//2].reshape(z.size(0), -1)
        V = self.V_net(q)
        dVdq = torch.autograd.grad(V.sum(), z, create_graph=True)[0]
        # Hamiltonian: dq/dt = p, dp/dt = -dV/dq
        p = z.view(z.size(0), self.k, 2, self.d//2)[:, :, 1]
        dqdt = p.reshape(z.size(0), -1)
        dpdt = -dVdq.view(z.size(0), self.k, 2, self.d//2)[:, :, 0].reshape(z.size(0), -1)
        return torch.cat([dqdt, dpdt], dim=-1), V

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    sim = PhysicsSim(mode=mode)
    p, v = sim.generate(400)
    model = DiscoveryNet(k=2, latent_dim=4)
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
        recon = model.dec(z).view(8, 4)
        l_rec = torch.nn.functional.mse_loss(recon, x)
        
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
    with torch.no_grad():
        all_x = torch.cat([p, v], dim=-1)
        z_all, _ = model.encode(all_x, p, edge_index, torch.zeros(p.size(0), 8, dtype=torch.long).view(-1))
        q_all = z_all.view(z_all.size(0), 2, 4)[:, :, :2].reshape(z_all.size(0), -1).numpy()
        _, V_all = model.get_force(z_all)
        y = V_all.detach().numpy().flatten()

    # Physics Basis Functions
    inv2 = make_function(function=lambda x: 1.0/(x**2 + 1e-4), name='inv2', arity=1)
    inv6 = make_function(function=lambda x: 1.0/(x**6 + 1e-4), name='inv6', arity=1)
    
    est = SymbolicRegressor(population_size=1000, generations=20, 
                            function_set=('add', 'sub', 'mul', 'div', inv2, inv6),
                            parsimony_coefficient=0.01, verbose=1)
    est.fit(q_all, y)
    
    print(f"\nDiscovered Potential V(q): {est._program}")
    
    # Sympy Cleanup
    X0, X1, X2, X3 = sp.symbols('q1_x q1_y q2_x q2_y')
    expr = sp.sympify(str(est._program), locals={'inv2': lambda x: 1/x**2, 'inv6': lambda x: 1/x**6})
    print(f"Simplified Physics: {sp.simplify(expr)}")

if __name__ == "__main__":
    train_discovery(mode='lj')
