import torch
import torch.nn as nn
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import time

# Device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# Custom functions for gplearn
def _square(x):
    return np.square(x)
square = make_function(function=_square, name='sq', arity=1)

# 1. ENHANCED SIMULATOR & VALIDATION
class PhysicsSim:
    def __init__(self, n=12, mode='lj', seed=None):
        if seed is not None: 
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.n, self.mode, self.dt = n, mode, 0.01
        # For LJ, use a larger box to avoid immediate explosion
        scale = 2.0 if mode == 'spring' else 4.0
        self.pos = torch.rand((n, 2), device=device, dtype=torch.float32) * scale
        self.vel = torch.randn((n, 2), device=device, dtype=torch.float32) * 0.05
        self.mask = (~torch.eye(n, device=device).bool()).unsqueeze(-1)
    
    def get_ground_truth_potential(self, r):
        if self.mode == 'spring':
            return 0.5 * 10.0 * (r - 1.0)**2
        else:
            r_c = np.clip(r, 0.4, 5.0)
            return 4.0 * (r_c**-12 - r_c**-6)

    def compute_forces(self, pos):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=1e-8)
        
        if self.mode == 'spring':
            f = -10.0 * (dist - 1.0) * (diff / dist_clip)
        else:
            d_inv = 1.0 / torch.clamp(dist, min=0.4, max=5.0)
            f = 24.0 * (2 * torch.pow(d_inv, 13) - torch.pow(d_inv, 7)) * (diff / dist_clip)
        
        return torch.sum(f * self.mask, dim=1)

    def generate(self, steps=1000):
        traj_p = torch.zeros((steps, self.n, 2), device=device)
        traj_v = torch.zeros((steps, self.n, 2), device=device)
        
        curr_pos = self.pos.clone()
        curr_vel = self.vel.clone()
        
        with torch.no_grad():
            for i in range(steps):
                f = self.compute_forces(curr_pos)
                curr_vel += f * self.dt
                curr_pos += curr_vel * self.dt
                traj_p[i] = curr_pos
                traj_v[i] = curr_vel
            
        return traj_p, traj_v

class TrajectoryScaler:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.p_mid = 0
        self.p_scale = 1

    def fit(self, p, v):
        self.p_mid = p.mean(dim=(0, 1), keepdim=True)
        self.p_scale = p.std().item() + 1e-6

    def transform(self, p, v):
        return (p - self.p_mid) / self.p_scale, v * (self.dt / self.p_scale)

def validate_discovered_law(expr, mode, scaler, n_particles=16):
    r_sym = sp.Symbol('r')
    try:
        f_torch_func = sp.lambdify(r_sym, expr, 'torch')
    except Exception:
        return 1.0

    sim_gt = PhysicsSim(n=n_particles, mode=mode, seed=42)
    p_gt, _ = sim_gt.generate(steps=100)
    
    curr_pos = sim_gt.pos.clone()
    curr_vel = sim_gt.vel.clone()
    p_disc = torch.zeros((100, n_particles, 2), device=device)
    
    a_factor = scaler.p_scale / (sim_gt.dt**2)
    p_scale = scaler.p_scale
    dt = sim_gt.dt
    mask = (~torch.eye(n_particles, device=device).bool()).unsqueeze(-1)
    
    with torch.inference_mode():
        for i in range(100):
            diff = curr_pos.unsqueeze(1) - curr_pos.unsqueeze(0)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            dist_clip = torch.clamp(dist, min=1e-8)
            
            r_scaled = dist / p_scale
            try:
                mag_scaled = f_torch_func(r_scaled)
                if not isinstance(mag_scaled, torch.Tensor):
                    mag_scaled = torch.full_like(dist, float(mag_scaled))
                mag_scaled = torch.nan_to_num(mag_scaled, nan=0.0, posinf=10.0, neginf=-10.0)
                
                f_pair = (mag_scaled * a_factor) * (diff / dist_clip)
                f = torch.sum(f_pair * mask, dim=1)
            except Exception:
                f = torch.zeros_like(curr_pos)
                
            curr_vel += f * dt
            curr_pos += curr_vel * dt
            p_disc[i] = curr_pos
    
    mse = torch.mean((p_gt - p_disc)**2).item()
    return np.nan_to_num(mse, nan=1.0)

# 2. CORE ARCHITECTURE
class DiscoveryNet(nn.Module):
    def __init__(self, n=12):
        super().__init__()
        self.n = n
        self.register_buffer('mask', ~torch.eye(n).bool())
        self.V_pair = nn.Sequential(
            nn.Linear(3, 64), 
            nn.Tanh(), 
            nn.Linear(64, 64), 
            nn.Tanh(), 
            nn.Linear(64, 1)
        )

    def get_potential(self, pos_scaled):
        diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist = torch.clamp(dist, min=0.05)
        dist_flat = dist[:, self.mask].view(pos_scaled.size(0), -1, 1)
        features = torch.cat([dist_flat, torch.pow(dist_flat, -1), torch.pow(dist_flat, -2)], dim=-1)
        return self.V_pair(features).sum(dim=1) * 0.5

    def get_forces(self, pos_scaled):
        pos_scaled = pos_scaled.requires_grad_(True)
        V = self.get_potential(pos_scaled)
        grad = torch.autograd.grad(V.sum(), pos_scaled, create_graph=True)[0]
        return -grad, V

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    t_start = time.time()
    sim = PhysicsSim(mode=mode)
    p_traj, v_traj = sim.generate(1000)
    
    scaler = TrajectoryScaler(dt=sim.dt)
    scaler.fit(p_traj, v_traj)
    p_s, v_s = scaler.transform(p_traj, v_traj)
    
    model = DiscoveryNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    
    r_large = torch.linspace(2.0, 5.0, 10, device=device).view(-1, 1)
    feat_large = torch.cat([r_large, torch.pow(r_large, -1), torch.pow(r_large, -2)], dim=-1)
    
    print(f"--- Training: {mode} ---")
    
    for epoch in range(601):
        idxs = np.random.randint(0, len(p_s)-1, size=32)
        p_batch = torch.cat([p_s[idxs], p_s[idxs+1]], dim=0)
        a_all, _ = model.get_forces(p_batch)
        a, a_next = a_all[:32], a_all[32:]
        q, p, q_next, p_next = p_s[idxs], v_s[idxs], p_s[idxs+1], v_s[idxs+1]
        
        l_dyn = torch.nn.functional.mse_loss(q + p + 0.5 * a, q_next) + \
                torch.nn.functional.mse_loss(p + 0.5 * (a + a_next), p_next)
        l_cons = torch.mean(model.V_pair(feat_large)**2)
        loss = l_dyn * 1e8 + l_cons * (1.0 if epoch > 300 else 0.0)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.2e} | Dyn: {l_dyn.item():.2e}")

    t_train = time.time()
    
    model.eval()
    r_samples = np.linspace(0.1, 5.0, 400).astype(np.float32).reshape(-1, 1)
    r_torch = torch.tensor(r_samples, dtype=torch.float32, device=device, requires_grad=True)
    feat = torch.cat([r_torch, torch.pow(r_torch, -1), torch.pow(r_torch, -2)], dim=-1)
    v_vals = model.V_pair(feat)
    force_vals = -torch.autograd.grad(v_vals.sum(), r_torch)[0].cpu().detach().numpy().flatten()

    est = SymbolicRegressor(population_size=400, generations=15,
                            function_set=('add', 'sub', 'mul', 'div', 'neg', square),
                            max_samples=0.5, stopping_criteria=1e-5,
                            n_jobs=-1, parsimony_coefficient=1e-4, random_state=42)
    est.fit(r_samples, force_vals)
    best_expr = str(est._program)
    t_symbolic = time.time()
    
    try:
        locals_dict = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b, 
                       'div':lambda a,b:a/b, 'neg':lambda a:-a, 'sq':lambda a:a**2, 'X0':sp.Symbol('r')}
        expr = sp.simplify(sp.sympify(best_expr, locals=locals_dict))
        mse = validate_discovered_law(expr, mode, scaler)
        print(f"Result {mode}: {expr} | MSE: {mse:.4e}")
    except Exception: pass

    print(f"Total Time {mode}: {time.time() - t_start:.2f}s")

if __name__ == "__main__":
    for m in ['spring', 'lj']:
        train_discovery(mode=m)
