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
class DiscoveryNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input features: [r, 1/r] to help with repulsive walls
        self.V_pair = nn.Sequential(
            nn.Linear(2, 128), 
            nn.SiLU(), 
            nn.Linear(128, 128), 
            nn.SiLU(), 
            nn.Linear(128, 1)
        )

    def get_potential(self, pos):
        # pos shape: (batch, n, 2)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        
        # Create mask for self-interactions
        n = pos.size(1)
        mask = ~torch.eye(n, device=pos.device).bool()
        dist_flat = dist[:, mask].view(pos.size(0), -1, 1)
        
        # Features: [r, 1/r]
        r = dist_flat
        r_inv = 1.0 / (r + 1e-6)
        features = torch.cat([r, r_inv], dim=-1)
        
        # Sum half of the pair potentials (since each pair is counted twice)
        return self.V_pair(features).sum(dim=1) * 0.5

    def get_forces(self, pos):
        pos = pos.requires_grad_(True)
        V = self.get_potential(pos)
        dVdq = torch.autograd.grad(V.sum(), pos, create_graph=True)[0]
        return -dVdq, V

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    sim = PhysicsSim(mode=mode)
    p_traj, v_traj = sim.generate(1000)
    
    num_particles = p_traj.size(1)
    model = DiscoveryNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"\n--- Training Discovery Pipeline: {mode} ---")
    for epoch in range(1001):
        # Batching: Sample 128 random transitions
        idxs = np.random.randint(0, 950, size=128)
        q = p_traj[idxs]
        p = v_traj[idxs]
        q_next = p_traj[idxs+1]
        p_next = v_traj[idxs+1]
        
        a, V = model.get_forces(q)
        
        dt = sim.dt
        # Velocity Verlet-style prediction
        q_next_pred = q + p * dt + 0.5 * a * (dt**2)
        
        # We need a_next for the velocity update
        # To save computation, we can use a simpler Euler or a_next from model
        a_next, _ = model.get_forces(q_next)
        p_next_pred = p + 0.5 * (a + a_next) * dt
        
        l_dyn = torch.nn.functional.mse_loss(q_next_pred, q_next) + \
                torch.nn.functional.mse_loss(p_next_pred, p_next)

        # Consistency Loss: V(r) -> 0 at large r
        # Input features for V_pair are [r, 1/r]
        large_r = torch.tensor([[10.0, 0.1]])
        v_inf = model.V_pair(large_r)
        l_cons = torch.abs(v_inf).mean()

        # Focus heavily on dynamics
        loss = l_dyn * 1000.0 + l_cons * 1.0
        
        opt.zero_grad()
        loss.backward()
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if epoch % 200 == 0: 
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Dyn: {l_dyn.item():.8f}")

    print("\nDistilling Symbolic Law...")
    model.eval()
    # Log-space sampling to capture the repulsive wall
    d_min, d_max = (0.5, 5.0) if mode=='lj' else (0.1, 5.0)
    r_samples = np.linspace(d_min, d_max, 500)
    with torch.no_grad():
        r_torch = torch.tensor(r_samples, dtype=torch.float32).unsqueeze(-1)
        r_inv_torch = 1.0 / (r_torch + 1e-6)
        features = torch.cat([r_torch, r_inv_torch], dim=-1)
        v_vals = model.V_pair(features).squeeze(-1).numpy()

    # Symbolic Regression with optimized parameters
    est = SymbolicRegressor(population_size=1000, generations=20,
                            function_set=('add', 'sub', 'mul', 'div', 'inv', 'neg'),
                            n_jobs=-1, parsimony_coefficient=0.01, 
                            verbose=0, random_state=42)
    est.fit(r_samples.reshape(-1, 1), v_vals)
    best_expr = str(est._program)

    try:
        locals_dict = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b, 
                       'div':lambda a,b:a/b, 'inv':lambda a:1/a, 'neg':lambda a:-a,
                       'X0':sp.Symbol('r')}
        expr = sp.simplify(sp.sympify(best_expr, locals=locals_dict))
        print(f"SUCCESS: Discovered V(r) = {expr}")
        validate_discovered_law(expr, mode)
    except Exception as e: print(f"Symbolic recovery failed: {best_expr} | Error: {e}")

if __name__ == "__main__":
    for m in ['lj', 'spring']:
        train_discovery(mode=m)