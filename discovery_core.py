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

def _inv(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 0.001, 1.0/x, 0.0)
inv = make_function(function=_inv, name='inv', arity=1)

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
    
    def compute_forces(self, pos):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=1e-8)
        
        if self.mode == 'spring':
            f = -10.0 * (dist - 1.0) * (diff / dist_clip)
        else:
            # LJ force: 24 * (2*r^-13 - r^-7)
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
        self.p_scale = 1.0

    def fit(self, p, v):
        self.p_mid = p.mean(dim=(0, 1), keepdim=True)
        self.p_scale = p.std().item() + 1e-6

    def transform(self, p, v):
        # p_scaled = (p - p_mid) / p_scale
        # v_scaled = v * (dt / p_scale)
        # a_scaled = a * (dt^2 / p_scale)
        return (p - self.p_mid) / self.p_scale, v * (self.dt / self.p_scale)

def validate_discovered_law(expr, mode, n_particles=16):
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
    
    dt = sim_gt.dt
    mask = (~torch.eye(n_particles, device=device).bool()).unsqueeze(-1)
    
    with torch.inference_mode():
        for i in range(100):
            diff = curr_pos.unsqueeze(1) - curr_pos.unsqueeze(0)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            dist_clip = torch.clamp(dist, min=1e-8)
            
            try:
                mag = f_torch_func(dist)
                if not isinstance(mag, torch.Tensor):
                    mag = torch.full_like(dist, float(mag))
                mag = torch.nan_to_num(mag, nan=0.0, posinf=100.0, neginf=-100.0)
                
                f_pair = mag * (diff / dist_clip)
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
            nn.Linear(3, 128), 
            nn.Tanh(), 
            nn.Linear(128, 128), 
            nn.Tanh(), 
            nn.Linear(128, 1)
        )

    def get_potential(self, pos_scaled):
        diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist = torch.clamp(dist, min=0.01)
        dist_flat = dist[:, self.mask].view(pos_scaled.size(0), -1, 1)
        # Physics-informed features: r, 1/r, 1/r^2
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
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    
    # Potential constraint to avoid trivial solutions
    r_far = torch.linspace(4.0, 10.0, 20, device=device).view(-1, 1)
    feat_far = torch.cat([r_far, torch.pow(r_far, -1), torch.pow(r_far, -2)], dim=-1)
    
    print(f"--- Training: {mode} ---")
    
    for epoch in range(1501):
        idxs = np.random.randint(0, len(p_s)-1, size=64)
        p_batch = torch.cat([p_s[idxs], p_s[idxs+1]], dim=0)
        a_all, _ = model.get_forces(p_batch)
        a, a_next = a_all[:64], a_all[64:]
        q, p, q_next, p_next = p_s[idxs], v_s[idxs], p_s[idxs+1], v_s[idxs+1]
        
        # Velocity Verlet-like loss
        l_pos = torch.nn.functional.mse_loss(q + p + 0.5 * a, q_next)
        l_vel = torch.nn.functional.mse_loss(p + 0.5 * (a + a_next), p_next)
        
        # Soft potential constraint: V(r) -> 0 for large r
        l_cons = torch.mean(model.V_pair(feat_far)**2)
        
        # Dynamic weighting
        w_dyn = 1e7 if mode == 'spring' else 1e8
        loss = (l_pos + l_vel) * w_dyn + l_cons * (0.1 if epoch > 500 else 0.0)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.2e} | Pos: {l_pos.item():.2e} | Vel: {l_vel.item():.2e}")

    model.eval()
    
    # HANDOVER TO SYMBOLIC REGRESSOR
    # Sample r in physical units
    r_phys_samples = np.linspace(0.4 if mode == 'lj' else 0.1, 5.0, 500).astype(np.float32).reshape(-1, 1)
    r_scaled = torch.tensor(r_phys_samples / scaler.p_scale, dtype=torch.float32, device=device, requires_grad=True)
    feat = torch.cat([r_scaled, torch.pow(r_scaled, -1), torch.pow(r_scaled, -2)], dim=-1)
    
    v_vals = model.V_pair(feat)
    # F_scaled = -dV/dr_scaled
    f_scaled_vals = -torch.autograd.grad(v_vals.sum(), r_scaled)[0].cpu().detach().numpy().flatten()
    # F_phys = F_scaled * (p_scale / dt^2)
    f_phys_vals = f_scaled_vals * (scaler.p_scale / (sim.dt**2))

    # Symbolic Discovery with Retry Loop
    parsimony = 1e-3
    best_mse = float('inf')
    best_final_expr = None
    
    for attempt in range(2):
        print(f"Symbolic Discovery Attempt {attempt+1} (parsimony={parsimony})...")
        est = SymbolicRegressor(population_size=1000, generations=100,
                                function_set=('add', 'sub', 'mul', 'div', 'neg', square, inv),
                                metric='mse', max_samples=0.9, stopping_criteria=1e-6,
                                n_jobs=-1, parsimony_coefficient=parsimony, random_state=42)
        
        est.fit(r_phys_samples, f_phys_vals)
        prog = str(est._program)
        
        try:
            locals_dict = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b, 
                           'div':lambda a,b:a/b, 'neg':lambda a:-a, 'sq':lambda a:a**2, 
                           'inv':lambda a:1.0/a, 'X0':sp.Symbol('r')}
            expr = sp.simplify(sp.sympify(prog, locals=locals_dict))
            
            # Check if constant
            if sp.diff(expr, sp.Symbol('r')) == 0:
                print("Discovered law is constant. Retrying with lower parsimony.")
                parsimony /= 10
                continue
                
            mse = validate_discovered_law(expr, mode)
            if mse < best_mse:
                best_mse = mse
                best_final_expr = expr
            
            if mse < (1e-5 if mode == 'spring' else 1e-2):
                break
        except Exception as e:
            print(f"Error in symbolic processing: {e}")
        
        parsimony /= 10

    print(f"Final Result {mode}: {best_final_expr} | MSE: {best_mse:.4e}")
    print(f"Total Time {mode}: {time.time() - t_start:.2f}s")

if __name__ == "__main__":
    for m in ['spring', 'lj']:
        train_discovery(mode=m)