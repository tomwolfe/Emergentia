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
    def __init__(self, n=8, mode='lj'):
        self.n, self.mode, self.dt = n, mode, 0.01
        self.pos = np.random.rand(n, 2) * 3.0
        self.vel = np.random.randn(n, 2) * 0.1
    
    def compute_forces(self, pos):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6
        if self.mode == 'spring':
            f = -10.0 * (dist - 1.0) * (diff / dist)
        else: # Lennard-Jones
            # Capped LJ to avoid numerical explosion during data gen
            d_inv = 1.0 / np.clip(dist, 0.5, 10.0)
            f = 24 * 1.0 * (2 * d_inv**13 - d_inv**7) * (diff / dist)
        return np.sum(np.nan_to_num(f), axis=1)

    def generate(self, steps=800):
        traj_p, traj_v = [], []
        for _ in range(steps):
            f = self.compute_forces(self.pos)
            self.vel += f * self.dt
            self.pos += self.vel * self.dt
            traj_p.append(self.pos.copy()); traj_v.append(self.vel.copy())
        return torch.tensor(np.array(traj_p), dtype=torch.float32), torch.tensor(np.array(traj_v), dtype=torch.float32)

class TrajectoryScaler:
    def __init__(self):
        self.mu_p, self.std_p = 0, 1
        self.mu_v, self.std_v = 0, 1

    def fit(self, p, v):
        self.mu_p = p.mean(dim=(0, 1), keepdim=True)
        self.std_p = p.std(dim=(0, 1), keepdim=True) + 1e-6
        self.mu_v = v.mean(dim=(0, 1), keepdim=True)
        self.std_v = v.std(dim=(0, 1), keepdim=True) + 1e-6

    def transform(self, p, v):
        return (p - self.mu_p) / self.std_p, (v - self.mu_v) / self.std_v

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
    def __init__(self, k=8, particle_latent_dim=4):
        super().__init__()
        self.k = k
        self.enc = EquiLayer(input_dim=4, output_dim=32)
        self.pool = nn.Linear(32, k)
        self.to_z = nn.Linear(32, particle_latent_dim) 
        # Potential MLP with robust Tanh activations
        self.V_pair = nn.Sequential(nn.Linear(1, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
        self.dec = nn.Sequential(nn.Linear(particle_latent_dim, 32), nn.SiLU(), nn.Linear(32, 4))

    def encode(self, x, pos, edge_index, batch):
        h = self.enc(x, pos, edge_index)
        s = torch.softmax(self.pool(h), dim=-1)
        pooled_h = scatter(s.unsqueeze(-1) * h.unsqueeze(1), batch, dim=0, reduce='sum')
        z = self.to_z(pooled_h)
        return z, s

    def decode(self, z, s):
        return self.dec(torch.matmul(s, z))

    def get_potential(self, q):
        k = q.size(1)
        q_i, q_j = q.unsqueeze(2), q.unsqueeze(1)
        dist = torch.norm(q_i - q_j, dim=-1, keepdim=True)
        # Clamping to avoid singularity in 1/r terms if they emerge
        dist = torch.clamp(dist, min=0.1)
        mask = torch.triu(torch.ones(k, k, device=q.device), diagonal=1).bool()
        dist_flat = dist[:, mask]
        return self.V_pair(dist_flat).sum(dim=1)

    def get_forces(self, q):
        q = q.requires_grad_(True)
        V = self.get_potential(q)
        dVdq = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
        return -dVdq, V

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    sim = PhysicsSim(mode=mode)
    p_traj_raw, v_traj_raw = sim.generate(800)
    scaler = TrajectoryScaler(); scaler.fit(p_traj_raw, v_traj_raw)
    p_traj, v_traj = scaler.transform(p_traj_raw, v_traj_raw)
    
    num_particles = p_traj.size(1)
    model = DiscoveryNet(k=num_particles, particle_latent_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    edge_index = torch.combinations(torch.arange(num_particles)).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    print(f"Training Discovery Pipeline for {mode}...")
    for epoch in range(1001):
        idx = np.random.randint(0, 750)
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
        
        # Centroid Alignment (Anchor latent space to physical space)
        l_cent = torch.nn.functional.mse_loss(q.squeeze(0), p_traj[idx])

        w_dyn = min(100.0, epoch / 5.0)
        loss = l_rec * 1.0 + l_dyn * w_dyn + l_cent * 1.0
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if epoch % 200 == 0: 
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} (Rec: {l_rec.item():.4f}, Dyn: {l_dyn.item():.8f}, Cent: {l_cent.item():.4f})")

    print("\nDistilling Symbolic Law...")
    model.eval()
    dist_list, V_list = [], []
    with torch.no_grad():
        # Test across a range of distances in the normalized space
        test_d = torch.linspace(0.1, 5.0, 200).unsqueeze(-1)
        v_vals = model.V_pair(test_d).squeeze(-1)
        for d, v in zip(test_d, v_vals):
            dist_list.append([d.item()]); V_list.append(v.item())

    X_train, y_train = np.array(dist_list), np.array(V_list)
    inv2 = make_function(function=lambda x: 1.0/(x**2 + 1e-4), name='inv2', arity=1)
    inv6 = make_function(function=lambda x: 1.0/(x**6 + 1e-4), name='inv6', arity=1)
    est = SymbolicRegressor(population_size=2000, generations=50,
                            function_set=('add', 'sub', 'mul', 'div', inv2, inv6),
                            parsimony_coefficient=0.001, verbose=0)
    est.fit(X_train, y_train)

    try:
        locals_dict = {'add':lambda a,b:a+b,'sub':lambda a,b:a-b,'mul':lambda a,b:a*b,'div':lambda a,b:a/b,
                       'inv2':lambda a:1/a**2,'inv6':lambda a:1/a**6,'X0':sp.Symbol('r')}
        expr = sp.simplify(sp.sympify(str(est._program), locals=locals_dict))
        print(f"Discovered Physical Law: V(r) = {expr}")
    except: print(f"Raw discovered expression: {est._program}")

if __name__ == "__main__":
    train_discovery(mode='lj')