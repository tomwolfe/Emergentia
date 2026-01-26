import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# Custom functions for gplearn
def _square(x):
    return np.square(x)
square = make_function(function=_square, name='sq', arity=1)

def _cube(x):
    return x**3
cube = make_function(function=_cube, name='cube', arity=1)

def _inv_sq(x):
    return 1.0 / (np.square(x) + 1e-4)
inv_sq = make_function(function=_inv_sq, name='inv_sq', arity=1)

# 1. ENHANCED SIMULATOR & VALIDATION
class PhysicsSim:
    def __init__(self, n=8, mode='lj', seed=None):
        if seed is not None: np.random.seed(seed)
        self.n, self.mode, self.dt = n, mode, 0.01
        self.pos = np.random.rand(n, 2) * 1.5 
        self.vel = np.random.randn(n, 2) * 0.1
    
    def compute_forces(self, pos):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)
        # Avoid division by zero
        dist_clip = np.clip(dist, 1e-8, None)
        
        if self.mode == 'spring':
            # k=10, r0=1.0
            f = -10.0 * (dist - 1.0) * (diff / dist_clip)
        else: # Lennard-Jones
            # Standard LJ: 4*epsilon * [12*(sigma/r)^13 - 6*(sigma/r)^7]
            # Here we use a slightly more stable version for generation
            d_inv = 1.0 / np.clip(dist, 0.4, 5.0)
            f = 24.0 * (2 * d_inv**13 - d_inv**7) * (diff / dist_clip)
        
        # Zero out self-interaction
        for i in range(self.n): f[i, i, :] = 0
        return np.sum(np.nan_to_num(f), axis=1)

    def generate(self, steps=1000):
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

    def inverse_transform_p(self, p_scaled):
        return p_scaled * self.std_p + self.mu_p

def validate_discovered_law(expr, mode, n_particles=16):
    """Zero-shot transfer: simulate with discovered law and compare to ground truth."""
    print(f"\nValidating Discovered Law for {mode} (N={n_particles})...")
    r = sp.Symbol('r')
    # Convert sympy expression to numeric force function: F = -dV/dr
    try:
        force_expr = -sp.diff(expr, r)
        f_func = sp.lambdify(r, force_expr, 'numpy')
    except Exception as e:
        print(f"Failed to create force function: {e}")
        return 1.0

    sim_gt = PhysicsSim(n=n_particles, mode=mode, seed=42)
    p_gt, _ = sim_gt.generate(steps=200)
    
    sim_disc = PhysicsSim(n=n_particles, mode=mode, seed=42)
    p_disc = []
    curr_pos = sim_disc.pos.copy()
    curr_vel = sim_disc.vel.copy()
    
    for _ in range(200):
        diff = curr_pos[:, None, :] - curr_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)
        dist_clip = np.clip(dist, 1e-8, None)
        
        try:
            # Handle potential overflows in lambda
            with np.errstate(all='ignore'):
                mag = f_func(dist)
                mag = np.nan_to_num(mag, posinf=100.0, neginf=-100.0)
                mag = np.clip(mag, -100.0, 100.0)
            f = mag * (diff / dist_clip)
            for i in range(n_particles): f[i, i, :] = 0
            f = np.sum(f, axis=1)
        except Exception:
            f = np.zeros_like(curr_pos)
            
        curr_vel += f * sim_disc.dt
        curr_pos += curr_vel * sim_disc.dt
        p_disc.append(curr_pos.copy())
    
    p_disc = np.array(p_disc)
    mse = np.mean((p_gt.numpy() - p_disc)**2)
    mse = np.nan_to_num(mse, nan=1.0)
    print(f"Validation MSE: {mse:.6f}")
    return mse

# 2. CORE ARCHITECTURE
class DiscoveryNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input features: [r, 1/r, 1/r^2] to help with repulsive walls
        self.V_pair = nn.Sequential(
            nn.Linear(3, 128), 
            nn.SiLU(), 
            nn.Linear(128, 128), 
            nn.SiLU(), 
            nn.Linear(128, 1)
        )

    def get_potential(self, pos):
        # pos shape: (batch, n, 2)
        diff = pos.unsqueeze(2) - pos.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        
        # Soft-core clamping
        dist = torch.clamp(dist, min=0.05)
        
        n = pos.size(1)
        mask = ~torch.eye(n, device=pos.device).bool()
        dist_flat = dist[:, mask].view(pos.size(0), -1, 1)
        
        # Features: [r, 1/r, 1/r^2]
        r = dist_flat
        r_inv = 1.0 / r
        r_inv2 = 1.0 / (r**2)
        features = torch.cat([r, r_inv, r_inv2], dim=-1)
        
        return self.V_pair(features).sum(dim=1) * 0.5

    def get_forces(self, pos, std_p=1.0):
        pos = pos.requires_grad_(True)
        V = self.get_potential(pos)
        dVdq = torch.autograd.grad(V.sum(), pos, create_graph=True)[0]
        # Chain rule: F = -dV/dq_orig = -dV/dq_scaled * (dq_scaled/dq_orig) = -dV/dq_scaled * (1/std_p)
        return -dVdq / std_p, V

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    sim = PhysicsSim(mode=mode)
    p_traj, v_traj = sim.generate(1000)
    
    scaler = TrajectoryScaler()
    scaler.fit(p_traj, v_traj)
    p_scaled, v_scaled = scaler.transform(p_traj, v_traj)
    
    model = DiscoveryNet()
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    print(f"\n--- Training Discovery Pipeline: {mode} ---")
    std_p = scaler.std_p.mean().item()
    
    for epoch in range(1001):
        idxs = np.random.randint(0, 950, size=128)
        
        # We use scaled coordinates for the model but must account for it in forces
        q = p_scaled[idxs]
        p = v_scaled[idxs]
        q_next = p_scaled[idxs+1]
        p_next = v_scaled[idxs+1]
        
        a, V = model.get_forces(q, std_p=std_p)
        
        dt = sim.dt
        # Predicted next state in scaled space
        # Note: a is in original units, so we need to scale it to match v_scaled and q_scaled
        a_scaled = a / scaler.std_v.mean().item()
        # v_scaled = v_orig / std_v -> v_orig = v_scaled * std_v
        # q_next_orig = q_orig + v_orig * dt + 0.5 * a_orig * dt^2
        # (q_next_scaled * std_p) = (q_scaled * std_p) + (v_scaled * std_v) * dt + 0.5 * a * dt^2
        # q_next_scaled = q_scaled + v_scaled * (std_v/std_p) * dt + 0.5 * (a/std_p) * dt^2
        
        v_to_q = (scaler.std_v.mean() / scaler.std_p.mean()).item()
        a_to_q = (1.0 / scaler.std_p.mean()).item()
        
        q_next_pred = q + p * v_to_q * dt + 0.5 * a * a_to_q * (dt**2)
        
        a_next, _ = model.get_forces(q_next, std_p=std_p)
        # p_next_orig = p_orig + 0.5 * (a + a_next) * dt
        # p_next_scaled * std_v = p_scaled * std_v + 0.5 * (a + a_next) * dt
        # p_next_scaled = p_scaled + 0.5 * (a + a_next) / std_v * dt
        
        a_to_v = (1.0 / scaler.std_v.mean()).item()
        p_next_pred = p + 0.5 * (a + a_next) * a_to_v * dt
        
        l_dyn = torch.nn.functional.mse_loss(q_next_pred, q_next) + \
                torch.nn.functional.mse_loss(p_next_pred, p_next)

        # Consistency Loss: V(r) -> 0 at large r (for LJ mostly)
        # Sample distances in scaled space
        r_large = torch.linspace(2.0, 5.0, 20).view(-1, 1)
        r_inv = 1.0 / r_large
        r_inv2 = 1.0 / (r_large**2)
        feat_large = torch.cat([r_large, r_inv, r_inv2], dim=-1)
        v_inf = model.V_pair(feat_large)
        l_cons = torch.mean(v_inf**2)

        loss = l_dyn * 1e4 + l_cons * 10.0
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if epoch % 200 == 0: 
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Dyn: {l_dyn.item():.8f} | Cons: {l_cons.item():.6f}")

    print("\nDistilling Symbolic Law...")
    model.eval()
    # Sample in original distance units for symbolic regression
    r_samples = np.linspace(0.5, 4.0, 1000)
    with torch.no_grad():
        # The model expects scaled distances?
        # Actually DiscoveryNet.get_potential expects 'pos' and calculates dist.
        # Let's bypass and go to V_pair. V_pair expects [r_scaled, 1/r_scaled, 1/r_scaled^2]
        r_scaled = r_samples / std_p
        r_scaled_torch = torch.tensor(r_scaled, dtype=torch.float32).unsqueeze(-1)
        r_inv = 1.0 / r_scaled_torch
        r_inv2 = 1.0 / (r_scaled_torch**2)
        features = torch.cat([r_scaled_torch, r_inv, r_inv2], dim=-1)
        v_vals = model.V_pair(features).squeeze(-1).numpy()

    # Symbolic Regression
    est = SymbolicRegressor(population_size=2000, generations=50,
                            function_set=('add', 'sub', 'mul', 'div', 'inv', 'neg', square, cube, inv_sq),
                            n_jobs=-1, parsimony_coefficient=0.001, 
                            verbose=0, random_state=42)
    est.fit(r_samples.reshape(-1, 1), v_vals)
    best_expr = str(est._program)

    try:
        locals_dict = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b, 
                       'div':lambda a,b:a/b, 'inv':lambda a:1/a, 'neg':lambda a:-a,
                       'sq':lambda a:a**2, 'cube':lambda a:a**3, 'inv_sq':lambda a:1/(a**2),
                       'X0':sp.Symbol('r')}
        expr = sp.simplify(sp.sympify(best_expr, locals=locals_dict))
        print(f"SUCCESS: Discovered V(r) = {expr}")
        validate_discovered_law(expr, mode)
    except Exception as e: print(f"Symbolic recovery failed: {best_expr} | Error: {e}")

if __name__ == "__main__":
    for m in ['spring', 'lj']:
        train_discovery(mode=m)
