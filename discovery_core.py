import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# 1. ENHANCED SIMULATOR & VALIDATION
class PhysicsSim:
    def __init__(self, n=8, mode='lj', seed=None):
        if seed is not None: np.random.seed(seed)
        self.n, self.mode, self.dt = n, mode, 0.01
        # Increased density to ensure collisions/repulsion exploration
        self.pos = np.random.rand(n, 2) * 1.5 
        self.vel = np.random.randn(n, 2) * 0.2
    
    def compute_forces(self, pos):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8
        if self.mode == 'spring':
            # k=10, r0=1.0
            f = -10.0 * (dist - 1.0) * (diff / dist)
        else: # Lennard-Jones
            # Standard LJ: 4*epsilon * [12*(sigma/r)^13 - 6*(sigma/r)^7]
            # Here epsilon=1, sigma=1 for simplicity in recovery
            d_inv = 1.0 / np.clip(dist, 0.1, 5.0)
            f = 24.0 * (2 * d_inv**13 - d_inv**7) * (diff / dist)
        
        # Zero out self-interaction
        np.fill_diagonal(f[:, :, 0], 0)
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

def validate_discovered_law(expr, mode, n_particles=16):
    """Zero-shot transfer: simulate with discovered law and compare to ground truth."""
    print(f"\nValidating Discovered Law for {mode} (N={n_particles})...")
    r = sp.Symbol('r')
    # Convert sympy expression to numeric force function: F = -dV/dr
    force_expr = -sp.diff(expr, r)
    f_func = sp.lambdify(r, force_expr, 'numpy')
    
    sim_gt = PhysicsSim(n=n_particles, mode=mode, seed=42)
    p_gt, _ = sim_gt.generate(steps=200)
    
    # Discovery-based simulation
    sim_disc = PhysicsSim(n=n_particles, mode=mode, seed=42)
    p_disc = []
    curr_pos = sim_disc.pos.copy()
    curr_vel = sim_disc.vel.copy()
    
    for _ in range(200):
        diff = curr_pos[:, None, :] - curr_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-8
        
        # Apply discovered force
        try:
            mag = f_func(dist)
            f = mag * (diff / dist)
            np.fill_diagonal(f[:, :, 0], 0)
            f = np.sum(np.nan_to_num(f), axis=1)
        except Exception as e:
            f = np.zeros_like(curr_pos)
            
        curr_vel += f * sim_disc.dt
        curr_pos += curr_vel * sim_disc.dt
        p_disc.append(curr_pos.copy())
    
    p_disc = np.array(p_disc)
    mse = np.mean((p_gt.numpy() - p_disc)**2)
    print(f"Validation MSE: {mse:.6f}")
    return mse

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
        self.V_pair = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 1))
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
        dist = torch.sqrt(torch.sum((q_i - q_j)**2, dim=-1, keepdim=True) + 1e-8)
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
    p_traj_raw, v_traj_raw = sim.generate(1000)
    scaler = TrajectoryScaler(); scaler.fit(p_traj_raw, v_traj_raw)
    p_traj, v_traj = scaler.transform(p_traj_raw, v_traj_raw)
    
    num_particles = p_traj.size(1)
    model = DiscoveryNet(k=num_particles, particle_latent_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    edge_index = torch.combinations(torch.arange(num_particles)).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    print(f"\n--- Training Discovery Pipeline: {mode} ---")
    for epoch in range(1501):
        idx = np.random.randint(0, 950)
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
        l_anchor_q = torch.nn.functional.mse_loss(q.squeeze(0), p_traj[idx])
        l_anchor_p = torch.nn.functional.mse_loss(p.squeeze(0), v_traj[idx])

        # Consistency Loss: V(r) -> 0 at large r
        v_inf = model.V_pair(torch.tensor([[10.0]]))
        l_cons = torch.abs(v_inf).mean()

        w_dyn = min(200.0, epoch / 2.0)
        loss = l_rec * 0.1 + l_dyn * w_dyn + l_anchor_q * 1.0 + l_anchor_p * 1.0 + l_cons * 0.5
        
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if epoch % 250 == 0: 
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Dyn: {l_dyn.item():.8f} | V_inf: {v_inf.item():.4f}")

    print("\nDistilling Symbolic Law via Log-Space Sampling...")
    model.eval()
    # Log-space sampling to capture the repulsive wall (small r)
    d_min, d_max = 0.6 if mode=='lj' else 0.1, 5.0
    r_samples = np.geomspace(d_min, d_max, 500)
    with torch.no_grad():
        test_d = torch.tensor(r_samples, dtype=torch.float32).unsqueeze(-1)
        v_vals = model.V_pair(test_d).squeeze(-1).numpy()

    # Parsimony Sweep
    best_expr, best_score = None, float('inf')
    for p_coeff in [0.0001, 0.001, 0.01]:
        est = SymbolicRegressor(population_size=5000, generations=40,
                                function_set=('add', 'sub', 'mul', 'div', 'inv', 'log'),
                                parsimony_coefficient=p_coeff, verbose=0, random_state=42)
        est.fit(r_samples.reshape(-1, 1), v_vals)
        
        program = str(est._program)
        complexity = est._program.length_ 
        error = est._program.raw_fitness_
        score = error + complexity * 0.01
        
        if score < best_score:
            best_score = score
            best_expr = program

    try:
        # Map gplearn names to sympy
        # Note: 'inv' in gplearn is 1/x
        locals_dict = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b, 
                       'div':lambda a,b:a/b, 'inv':lambda a:1/a, 'log':lambda a:sp.log(sp.Abs(a)),
                       'X0':sp.Symbol('r')}
        expr = sp.simplify(sp.sympify(best_expr, locals=locals_dict))
        print(f"SUCCESS: Discovered V(r) = {expr}")
        validate_discovered_law(expr, mode)
    except Exception as e: print(f"Symbolic recovery failed to simplify: {best_expr} | Error: {e}")

if __name__ == "__main__":
    for m in ['lj', 'spring']:
        train_discovery(mode=m)