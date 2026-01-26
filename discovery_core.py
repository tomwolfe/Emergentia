import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# 1. MINIMAL SIMULATOR
class PhysicsSim:
    def __init__(self, n=4, mode='spring'): # Fewer particles for clarity
        self.n, self.mode, self.dt = n, mode, 0.02
        self.pos = np.random.rand(n, 2) * 2.0
        self.vel = np.random.randn(n, 2) * 0.1
    
    def compute_forces(self, pos):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6
        if self.mode == 'spring':
            # V = 5 * (dist - 1.0)^2 => F = -10 * (dist - 1.0)
            f = -10.0 * (dist - 1.0) * (diff / dist)
        else: # Lennard-Jones
            f = 24 * 1.0 * (2 * (1.0/dist)**13 - (1.0/dist)**7) * (diff / dist)
        return np.sum(np.nan_to_num(f), axis=1)

    def generate(self, steps=1000):
        traj_p, traj_v = [], []
        for _ in range(steps):
            f = self.compute_forces(self.pos)
            self.vel += f * self.dt
            self.pos += self.vel * self.dt
            traj_p.append(self.pos.copy()); traj_v.append(self.vel.copy())
        return torch.tensor(np.array(traj_p), dtype=torch.float32), torch.tensor(np.array(traj_v), dtype=torch.float32)

# 2. CORE ARCHITECTURE
class EquiLayer(MessagePassing):
    def __init__(self, input_dim, output_dim=32):
        super().__init__(aggr='mean')
        self.f = nn.Sequential(nn.Linear(input_dim*2 + 1, output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))
    def forward(self, x, pos, edge_index):
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.norm(rel_pos, dim=-1, keepdim=True)
        return self.propagate(edge_index, x=x, dist=dist)
    def message(self, x_i, x_j, dist):
        return self.f(torch.cat([x_i, x_j, dist], dim=-1))

class DiscoveryNet(nn.Module):
    def __init__(self, k=4, particle_latent_dim=4):
        super().__init__()
        self.k = k
        self.enc = EquiLayer(input_dim=4, output_dim=32)
        self.pool = nn.Linear(32, k)
        self.to_z = nn.Linear(32, particle_latent_dim) 
        self.V_pair = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.dec = nn.Sequential(nn.Linear(particle_latent_dim, 32), nn.SiLU(), nn.Linear(32, 4))

    def encode(self, x, pos, edge_index, batch):
        h = torch.relu(self.enc(x, pos, edge_index))
        s = torch.softmax(self.pool(h), dim=-1)
        pooled_h = scatter(s.unsqueeze(-1) * h.unsqueeze(1), batch, dim=0, reduce='sum')
        z = self.to_z(pooled_h)
        return z, s

    def decode(self, z, s):
        return self.dec(torch.matmul(s, z))

    def get_potential(self, q):
        k = q.size(1)
        q_i, q_j = q.unsqueeze(2), q.unsqueeze(1)
        dist = torch.norm(q_i - q_j, dim=-1, keepdim=True) + 1e-6
        mask = torch.triu(torch.ones(k, k, device=q.device), diagonal=1).bool()
        dist_flat = dist[:, mask]
        return self.V_pair(dist_flat).sum(dim=1)

    def get_forces(self, q):
        q = q.requires_grad_(True)
        V = self.get_potential(q)
        dVdq = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
        return -dVdq, V

# 3. TRAINING & DISCOVERY
def train_discovery(mode='spring'):
    sim = PhysicsSim(n=4, mode=mode)
    p_traj, v_traj = sim.generate(1000)
    # NO SCALING for spring to keep physical units clear
    
    num_particles = p_traj.size(1)
    model = DiscoveryNet(k=num_particles, particle_latent_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    edge_index = torch.combinations(torch.arange(num_particles)).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    print(f"Training Discovery Pipeline for {mode}...")
    for epoch in range(1501):
        idx = np.random.randint(0, 900)
        x = torch.cat([p_traj[idx], v_traj[idx]], dim=-1)
        z, s = model.encode(x, p_traj[idx], edge_index, torch.zeros(num_particles, dtype=torch.long))
        
        l_rec = torch.nn.functional.mse_loss(model.decode(z, s), x.unsqueeze(0))
        q, p = z[:, :, :2], z[:, :, 2:]
        a, V = model.get_forces(q)
        
        dt = sim.dt
        q_next_pred = q + p * dt + 0.5 * a * (dt**2)
        z_next, _ = model.encode(torch.cat([p_traj[idx+1], v_traj[idx+1]], dim=-1), p_traj[idx+1], edge_index, torch.zeros(num_particles, dtype=torch.long))
        q_next, p_next = z_next[:, :, :2], z_next[:, :, 2:]
        a_next, _ = model.get_forces(q_next)
        p_next_pred = p + 0.5 * (a + a_next) * dt
        
        l_dyn = torch.nn.functional.mse_loss(q_next_pred, q_next) + torch.nn.functional.mse_loss(p_next_pred, p_next)
        l_cent = torch.nn.functional.mse_loss(q.squeeze(0), p_traj[idx]) # Direct alignment for k=num_particles

        # Focus on dynamics
        loss = l_rec * 0.1 + l_dyn * 100.0 + l_cent * 1.0
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if epoch % 250 == 0: 
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} (Rec: {l_rec.item():.4f}, Dyn: {l_dyn.item():.8f}, Cent: {l_cent.item():.4f})")

    print("\nDistilling Symbolic Law...")
    model.eval()
    dist_list, V_list = [], []
    with torch.no_grad():
        test_dists = torch.linspace(0.1, 4.0, 100).unsqueeze(-1)
        v_vals = model.V_pair(test_dists).squeeze(-1)
        for d, v in zip(test_dists, v_vals):
            dist_list.append([d.item()]); V_list.append(v.item())

    X_train, y_train = np.array(dist_list), np.array(V_list)
    est = SymbolicRegressor(population_size=2000, generations=40,
                            function_set=('add', 'sub', 'mul', 'div'),
                            parsimony_coefficient=0.01, verbose=0)
    est.fit(X_train, y_train)

    try:
        locals_dict = {'add':lambda a,b:a+b,'sub':lambda a,b:a-b,'mul':lambda a,b:a*b,'div':lambda a,b:a/b,'X0':sp.Symbol('r')}
        expr = sp.simplify(sp.sympify(str(est._program), locals=locals_dict))
        print(f"Discovered Physical Law: V(r) = {expr}")
    except: print(f"Raw discovered expression: {est._program}")

if __name__ == "__main__":
    train_discovery(mode='spring')
