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
    def __init__(self, k=2, particle_latent_dim=4):  # Each latent particle: 2 pos + 2 mom
        super().__init__()
        self.k = k
        self.d = particle_latent_dim
        self.d_q = particle_latent_dim // 2
        self.enc = EquiLayer(input_dim=4, output_dim=16)
        self.pool = nn.Linear(16, k)
        self.to_z = nn.Linear(16, particle_latent_dim) 
        # Potential acting on pairwise distances
        self.V_pair = nn.Sequential(nn.Linear(1, 32), nn.Softplus(), nn.Linear(32, 1))
        self.dec_bridge = nn.Linear(k * particle_latent_dim, 8 * 4)
        self.dec = nn.Sequential(nn.Linear(particle_latent_dim, 16), nn.SiLU(), nn.Linear(16, 4))

    def encode(self, x, pos, edge_index, batch):
        h = torch.relu(self.enc(x, pos, edge_index))
        s = torch.softmax(self.pool(h), dim=-1) # [num_nodes, k]
        # Soft-pooling to k super-nodes
        pooled_h = scatter(s.unsqueeze(-1) * h.unsqueeze(1), batch, dim=0, reduce='sum') # [batch, k, 16]
        z = self.to_z(pooled_h) # [batch, k, d]
        return z, s

    def decode(self, z):
        # z: [batch, k, d]
        h_flat = z.view(z.size(0), -1)
        out = self.dec_bridge(h_flat)
        return out.view(z.size(0), 8, 4)

    def get_potential(self, q):
        # q: [batch, k, d_q]
        k = q.size(1)
        q_i = q.unsqueeze(2) # [batch, k, 1, d_q]
        q_j = q.unsqueeze(1) # [batch, 1, k, d_q]
        # Differentiable norm with epsilon
        dist = torch.sqrt(torch.sum((q_i - q_j)**2, dim=-1, keepdim=True) + 1e-6) # [batch, k, k, 1]
        
        mask = torch.triu(torch.ones(k, k, device=q.device), diagonal=1).bool()
        dist_flat = dist[:, mask] # [batch, k*(k-1)/2, 1]
        v_pairs = self.V_pair(dist_flat)
        return v_pairs.sum(dim=1)

    def get_forces(self, q):
        q = q.requires_grad_(True)
        V = self.get_potential(q)
        dVdq = torch.autograd.grad(V.sum(), q, create_graph=True)[0]
        return -dVdq, V

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    sim = PhysicsSim(mode=mode)
    p_traj, v_traj = sim.generate(400)
    model = DiscoveryNet(k=2, particle_latent_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    edge_index = torch.combinations(torch.arange(8)).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    print(f"Training Neural-Hamiltonian for {mode}...")
    for epoch in range(400):
        idx = np.random.randint(0, 350)
        x = torch.cat([p_traj[idx], v_traj[idx]], dim=-1)
        z, s = model.encode(x, p_traj[idx], edge_index, torch.zeros(8, dtype=torch.long))
        
        # Reconstruction Loss
        decoded = model.decode(z)
        l_rec = torch.nn.functional.mse_loss(decoded, x.unsqueeze(0))
        
        # Hamiltonian Consistency Loss (Velocity Verlet)
        q = z[:, :, :2]
        p = z[:, :, 2:]
        a, V = model.get_forces(q)
        
        dt = sim.dt
        q_next_pred = q + p * dt + 0.5 * a * (dt**2)
        
        z_next, _ = model.encode(torch.cat([p_traj[idx+1], v_traj[idx+1]], dim=-1), p_traj[idx+1], edge_index, torch.zeros(8, dtype=torch.long))
        q_next = z_next[:, :, :2]
        p_next = z_next[:, :, 2:]
        
        a_next, V_next = model.get_forces(q_next)
        p_next_pred = p + 0.5 * (a + a_next) * dt
        
        l_dyn = torch.nn.functional.mse_loss(q_next_pred, q_next) + torch.nn.functional.mse_loss(p_next_pred, p_next)
        
        # Weighted Loss (1:1 ratio as requested)
        loss = l_rec + l_dyn * 1.0
        
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 50 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.6f} (Rec: {l_rec.item():.6f}, Dyn: {l_dyn.item():.6f})")

    # 4. SYMBOLIC DISTILLATION
    print("\nDistilling Symbolic Law from Pairwise Distances...")
    model.eval()
    dist_list = []
    V_list = []
    
    with torch.no_grad():
        for t_idx in range(min(200, p_traj.size(0))):
            x_t = torch.cat([p_traj[t_idx], v_traj[t_idx]], dim=-1)
            z_t, _ = model.encode(x_t, p_traj[t_idx], edge_index, torch.zeros(8, dtype=torch.long))
            q_t = z_t[:, :, :2]
            
            # Compute pairwise distances in latent space
            q_i = q_t.unsqueeze(2)
            q_j = q_t.unsqueeze(1)
            dists = torch.norm(q_i - q_j, dim=-1)
            mask = torch.triu(torch.ones(model.k, model.k), diagonal=1).bool()
            dist_flat = dists[0, mask]
            
            # We want to map each distance to its potential contribution
            # Since V = sum(V_pair(dist)), we can just use V_pair directly if we had it
            # But SymbolicRegressor will fit the total V. 
            # If k=2, there is only one distance.
            dist_list.append(dist_flat.cpu().numpy())
            V_list.append(model.get_potential(q_t).cpu().numpy())

    X_train = np.array(dist_list) # [N, k*(k-1)/2]
    y_train = np.array(V_list).squeeze()

    # Physics Basis Functions
    inv2 = make_function(function=lambda x: 1.0/(x**2 + 1e-4), name='inv2', arity=1)
    inv6 = make_function(function=lambda x: 1.0/(x**6 + 1e-4), name='inv6', arity=1)

    est = SymbolicRegressor(population_size=1000, generations=30,
                            function_set=('add', 'sub', 'mul', 'div', inv2, inv6),
                            parsimony_coefficient=0.005, verbose=1)
    est.fit(X_train, y_train)

    print(f"\nDiscovered Potential V(dist): {est._program}")

    try:
        expr_str = str(est._program)
        expr_str = expr_str.replace('inv2(', '1/(').replace('**2)', '**2)')
        expr_str = expr_str.replace('inv6(', '1/(').replace('**6)', '**6)')
        
        # X0 is the distance between the two latent particles (if k=2)
        r = sp.symbols('r')
        expr_str = expr_str.replace('X0', 'r')
        expr = sp.sympify(expr_str)
        print(f"Simplified Physics: V(r) = {sp.simplify(expr)}")
    except Exception as e:
        print(f"Raw discovered expression: {est._program}")

if __name__ == "__main__":
    train_discovery(mode='lj')
