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
        if seed is not None: np.random.seed(seed)
        self.n, self.mode, self.dt = n, mode, 0.01
        self.pos = np.random.rand(n, 2) * 2.0 
        self.vel = np.random.randn(n, 2) * 0.05
    
    def get_ground_truth_potential(self, r):
        if self.mode == 'spring':
            return 0.5 * 10.0 * (r - 1.0)**2
        else:
            # LJ stable version: f = 24 * (2*r^-13 - r^-7) -> V = 4*r^-12 - 4*r^-6
            r_c = np.clip(r, 0.4, 5.0)
            return 4.0 * (r_c**-12 - r_c**-6)

    def compute_forces(self, pos):
        diff = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)
        dist_clip = np.clip(dist, 1e-8, None)
        
        if self.mode == 'spring':
            f = -10.0 * (dist - 1.0) * (diff / dist_clip)
        else:
            d_inv = 1.0 / np.clip(dist, 0.4, 5.0)
            f = 24.0 * (2 * d_inv**13 - d_inv**7) * (diff / dist_clip)
        
        for i in range(self.n): f[i, i, :] = 0
        return np.sum(np.nan_to_num(f), axis=1)

    def generate(self, steps=1500):
        traj_p, traj_v = [], []
        for _ in range(steps):
            f = self.compute_forces(self.pos)
            self.vel += f * self.dt
            self.pos += self.vel * self.dt
            traj_p.append(self.pos.copy()); traj_v.append(self.vel.copy())
        return torch.tensor(np.array(traj_p), dtype=torch.float32, device=device), \
               torch.tensor(np.array(traj_v), dtype=torch.float32, device=device)

class TrajectoryScaler:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.p_mid = 0
        self.p_scale = 1
        self.v_scale = 1

    def fit(self, p, v):
        self.p_mid = p.mean(dim=(0, 1), keepdim=True)
        self.p_scale = p.std().item() + 1e-6
        self.v_scale = self.p_scale / self.dt

    def transform(self, p, v):
        return (p - self.p_mid) / self.p_scale, v / self.v_scale

    def inverse_transform_p(self, p_s):
        return p_s * self.p_scale + self.p_mid

def validate_discovered_law(expr, mode, scaler, n_particles=16):
    print(f"\nValidating Discovered Law for {mode} (N={n_particles})...")
    r_sym = sp.Symbol('r')
    try:
        f_torch_func = sp.lambdify(r_sym, expr, 'torch')
    except Exception as e:
        print(f"Failed to create force function: {e}")
        return 1.0

    sim_gt = PhysicsSim(n=n_particles, mode=mode, seed=42)
    p_gt, _ = sim_gt.generate(steps=200)
    
    curr_pos = torch.tensor(sim_gt.pos, device=device, dtype=torch.float32)
    curr_vel = torch.tensor(sim_gt.vel, device=device, dtype=torch.float32)
    p_disc = []
    
    a_factor = scaler.p_scale / (sim_gt.dt**2)
    p_scale = scaler.p_scale
    dt = sim_gt.dt
    
    mask = ~torch.eye(n_particles, device=device).bool()
    
    with torch.no_grad():
        for _ in range(200):
            diff = curr_pos.unsqueeze(1) - curr_pos.unsqueeze(0)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            dist_clip = torch.clamp(dist, min=1e-8)
            
            r_scaled = dist / p_scale
            try:
                mag_scaled = f_torch_func(r_scaled)
                mag_scaled = torch.nan_to_num(mag_scaled, nan=0.0, posinf=10.0, neginf=-10.0)
                
                f_pair = (mag_scaled * a_factor) * (diff / dist_clip)
                f_pair = f_pair * mask.unsqueeze(-1)
                f = torch.sum(f_pair, dim=1)
            except Exception:
                f = torch.zeros_like(curr_pos)
                
            curr_vel += f * dt
            curr_pos += curr_vel * dt
            p_disc.append(curr_pos.clone())
    
    p_disc = torch.stack(p_disc)
    mse = torch.mean((p_gt - p_disc)**2).item()
    mse = np.nan_to_num(mse, nan=1.0)
    print(f"Validation MSE: {mse:.6f}")
    return mse

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
        dist = torch.clamp(dist, min=0.05)
        
        # Use pre-calculated mask
        dist_flat = dist[:, self.mask].view(pos_scaled.size(0), -1, 1)
        
        r = dist_flat
        features = torch.cat([r, 1.0/r, 1.0/(r**2)], dim=-1)
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
    p_traj, v_traj = sim.generate(1500)
    
    scaler = TrajectoryScaler(dt=sim.dt)
    scaler.fit(p_traj, v_traj)
    p_s, v_s = scaler.transform(p_traj, v_traj)
    p_s, v_s = p_s.to(device), v_s.to(device)
    
    model = DiscoveryNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2000)
    
    # Pre-calculate consistency features
    r_large = torch.linspace(2.0, 5.0, 20, device=device).view(-1, 1)
    feat_large = torch.cat([r_large, 1.0/r_large, 1.0/(r_large**2)], dim=-1)
    
    print(f"\n--- Training Discovery Pipeline: {mode} ---")
    
    for epoch in range(2001):
        idxs = np.random.randint(0, len(p_s)-1, size=128)
        
        # Optimize force calculation: Batch p_t and p_{t+1} to compute forces in one go
        p_batch = torch.cat([p_s[idxs], p_s[idxs+1]], dim=0)
        a_all, _ = model.get_forces(p_batch)
        
        a = a_all[:128]
        a_next = a_all[128:]
        
        q, p = p_s[idxs], v_s[idxs]
        q_next, p_next = p_s[idxs+1], v_s[idxs+1]
        
        q_next_pred = q + p + 0.5 * a
        p_next_pred = p + 0.5 * (a + a_next)
        
        l_dyn = torch.nn.functional.mse_loss(q_next_pred, q_next) + \
                torch.nn.functional.mse_loss(p_next_pred, p_next)

        v_inf = model.V_pair(feat_large)
        l_cons = torch.mean(v_inf**2)
        
        l_cons_weight = 0.0 if epoch < 1000 else 100.0
        loss = l_dyn * 1e7 + l_cons * l_cons_weight
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if epoch % 200 == 0: 
            a_mag = a.abs().mean().item()
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4e} | Dyn: {l_dyn.item():.4e} | a_mag: {a_mag:.4e}")

    t_train = time.time()
    print(f"Training Phase Duration: {t_train - t_start:.2f}s")
    
    print("\nDistilling Symbolic Force...")
    model.eval()
    
    # Smart sampling for GPlearn: 400 log-spaced near well, 400 linear for tail
    r_log = np.logspace(np.log10(0.05), np.log10(1.5), 400)
    r_lin = np.linspace(1.5, 5.0, 400)
    r_samples = np.concatenate([r_log, r_lin]).astype(np.float32).reshape(-1, 1)

    r_torch = torch.tensor(r_samples, dtype=torch.float32, device=device, requires_grad=True)
    feat = torch.cat([r_torch, 1.0/r_torch, 1.0/(r_torch**2)], dim=-1)
    v_vals = model.V_pair(feat)
    force_vals = -torch.autograd.grad(v_vals.sum(), r_torch)[0].cpu().detach().numpy().flatten()

    est = SymbolicRegressor(population_size=800, generations=30,
                            function_set=('add', 'sub', 'mul', 'div', 'inv', 'neg', square),
                            n_jobs=-1, parsimony_coefficient=0.001, 
                            verbose=0, random_state=42)
    est.fit(r_samples, force_vals)
    best_expr = str(est._program)
    t_symbolic = time.time()
    print(f"Symbolic Phase Duration: {t_symbolic - t_train:.2f}s")

    try:
        locals_dict = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b, 
                       'div':lambda a,b:a/b, 'inv':lambda a:1/a, 'neg':lambda a:-a,
                       'sq':lambda a:a**2, 'X0':sp.Symbol('r')}
        expr = sp.simplify(sp.sympify(best_expr, locals=locals_dict))
        print(f"SUCCESS: Discovered F_scaled(r_scaled) = {expr}")
        
        mse = validate_discovered_law(expr, mode, scaler)
        t_val = time.time()
        print(f"Validation Phase Duration: {t_val - t_symbolic:.2f}s")
        
        r_plot = np.linspace(max(0.1, r_samples.min()), min(5.0, r_samples.max()), 100)
        r_plot_phys = r_plot * scaler.p_scale
        v_gt = np.array([sim.get_ground_truth_potential(r) for r in r_plot_phys])
        
        with torch.no_grad():
            r_t = torch.tensor(r_plot.reshape(-1, 1), dtype=torch.float32, device=device)
            feat_p = torch.cat([r_t, 1.0/r_t, 1.0/(r_t**2)], dim=-1)
            v_nn_scaled = model.V_pair(feat_p).cpu().numpy().flatten()
            v_nn = v_nn_scaled * (scaler.p_scale**2 / sim.dt**2)
            v_nn -= (v_nn[-1] - v_gt[-1])

        r_s = sp.Symbol('r')
        v_disc_expr = -sp.integrate(expr, r_s)
        v_disc_func = sp.lambdify(r_s, v_disc_expr, 'numpy')
        v_disc_scaled = v_disc_func(r_plot)
        v_disc = v_disc_scaled * (scaler.p_scale**2 / sim.dt**2)
        v_disc -= (v_disc[-1] - v_gt[-1])

        plt.figure(figsize=(8, 5))
        plt.plot(r_plot_phys, v_gt, 'k--', label='Ground Truth')
        plt.plot(r_plot_phys, v_nn, 'r-', alpha=0.5, label='Neural Network')
        plt.plot(r_plot_phys, v_disc, 'b:', label='Symbolic Discovery')
        plt.ylim(v_gt.min() - 2, v_gt.max() + 2)
        plt.title(f"Discovery Result: {mode} (MSE: {mse:.6f})")
        plt.legend()
        plt.savefig(f"discovery_{mode}.png")
        plt.close()

    except Exception as e: print(f"Symbolic recovery failed: {best_expr} | Error: {e}")

if __name__ == "__main__":
    for m in ['spring', 'lj']:
        train_discovery(mode=m)
