import torch
import torch.nn as nn
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# Custom functions for gplearn
def _square(x):
    return np.square(x)
square = make_function(function=_square, name='sq', arity=1)

# 1. ENHANCED SIMULATOR & VALIDATION
class PhysicsSim:
    def __init__(self, n=8, mode='lj', seed=None):
        if seed is not None: np.random.seed(seed)
        self.n, self.mode, self.dt = n, mode, 0.01
        self.pos = np.random.rand(n, 2) * 1.5 
        self.vel = np.random.randn(n, 2) * 0.1
    
    def get_ground_truth_potential(self, r):
        if self.mode == 'spring':
            # V = 0.5 * k * (r - r0)**2 -> k=10, r0=1.0
            return 0.5 * 10.0 * (r - 1.0)**2
        else:
            # LJ: V = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6]
            # Here we used a stable version in compute_forces, let's match it
            # f = 24 * (2*r^-13 - r^-7) -> V = 24 * ( (2/12)*r^-12 - (1/6)*r^-6 ) = 4*r^-12 - 4*r^-6
            return 4.0 * (r**-12 - r**-6)

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

    def generate(self, steps=1000):
        traj_p, traj_v = [], []
        for _ in range(steps):
            f = self.compute_forces(self.pos)
            self.vel += f * self.dt
            self.pos += self.vel * self.dt
            traj_p.append(self.pos.copy()); traj_v.append(self.vel.copy())
        return torch.tensor(np.array(traj_p), dtype=torch.float32), torch.tensor(np.array(traj_v), dtype=torch.float32)

class TrajectoryScaler:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.p_mid = 0
        self.p_scale = 1
        self.v_scale = 1

    def fit(self, p, v):
        self.p_mid = p.mean(dim=(0, 1), keepdim=True)
        p_centered = p - self.p_mid
        self.p_scale = p_centered.abs().max().item() + 1e-6
        # Dimensionless v scale: v_scaled = v * dt / p_scale
        self.v_scale = self.p_scale / self.dt

    def transform(self, p, v):
        return (p - self.p_mid) / self.p_scale, v / self.v_scale

    def inverse_transform_p(self, p_s):
        return p_s * self.p_scale + self.p_mid

def validate_discovered_law(expr, mode, scaler, n_particles=16):
    print(f"\nValidating Discovered Law for {mode} (N={n_particles})...")
    r_sym = sp.Symbol('r')
    # Force distillation means 'expr' is F(r_scaled)
    try:
        f_scaled_func = sp.lambdify(r_sym, expr, 'numpy')
    except Exception as e:
        print(f"Failed to create force function: {e}")
        return 1.0

    sim_gt = PhysicsSim(n=n_particles, mode=mode, seed=42)
    p_gt, _ = sim_gt.generate(steps=200)
    
    sim_disc = PhysicsSim(n=n_particles, mode=mode, seed=42)
    p_disc = []
    curr_pos = sim_disc.pos.copy()
    curr_vel = sim_disc.vel.copy()
    
    # Scale conversion for force:
    # a_phys = (p_scale / dt^2) * a_scaled
    # a_scaled = f_scaled_func(r_phys / p_scale)
    a_factor = scaler.p_scale / (sim_gt.dt**2)
    
    for _ in range(200):
        diff = curr_pos[:, None, :] - curr_pos[None, :, :]
        dist = np.linalg.norm(diff, axis=-1, keepdims=True)
        dist_clip = np.clip(dist, 1e-8, None)
        
        try:
            r_scaled = dist / scaler.p_scale
            with np.errstate(all='ignore'):
                mag_scaled = f_scaled_func(r_scaled)
                mag_scaled = np.nan_to_num(mag_scaled, posinf=10.0, neginf=-10.0)
            
            f = (mag_scaled * a_factor) * (diff / dist_clip)
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
        self.V_pair = nn.Sequential(
            nn.Linear(3, 128), 
            nn.Softplus(), 
            nn.Linear(128, 128), 
            nn.Softplus(), 
            nn.Linear(128, 1)
        )

    def get_potential(self, pos_scaled):
        diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist = torch.clamp(dist, min=0.01)
        
        n = pos_scaled.size(1)
        mask = ~torch.eye(n, device=pos_scaled.device).bool()
        dist_flat = dist[:, mask].view(pos_scaled.size(0), -1, 1)
        
        # Features: [r, 1/r, 1/r^2]
        r = dist_flat
        features = torch.cat([r, 1.0/r, 1.0/(r**2)], dim=-1)
        return self.V_pair(features).sum(dim=1) * 0.5

    def get_forces(self, pos_scaled):
        pos_scaled = pos_scaled.requires_grad_(True)
        V = self.get_potential(pos_scaled)
        # In dimensionless space, a = -grad(V)
        grad = torch.autograd.grad(V.sum(), pos_scaled, create_graph=True)[0]
        return -grad, V

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    sim = PhysicsSim(mode=mode)
    p_traj, v_traj = sim.generate(1000)
    
    scaler = TrajectoryScaler(dt=sim.dt)
    scaler.fit(p_traj, v_traj)
    p_s, v_s = scaler.transform(p_traj, v_traj)
    
    model = DiscoveryNet()
    # L2 penalty to prevent over-fitting
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
    
    print(f"\n--- Training Discovery Pipeline: {mode} ---")
    
    for epoch in range(1001):
        idxs = np.random.randint(0, 950, size=128)
        q, p = p_s[idxs], v_s[idxs]
        q_next, p_next = p_s[idxs+1], v_s[idxs+1]
        
        a, V = model.get_forces(q)
        
        # Dimensionless integration (dt_eff = 1)
        q_next_pred = q + p + 0.5 * a
        a_next, _ = model.get_forces(q_next)
        p_next_pred = p + 0.5 * (a + a_next)
        
        l_dyn = torch.nn.functional.mse_loss(q_next_pred, q_next) + \
                torch.nn.functional.mse_loss(p_next_pred, p_next)

        # Consistency Loss warm-up
        r_large = torch.linspace(1.5, 4.0, 20).view(-1, 1)
        feat_large = torch.cat([r_large, 1.0/r_large, 1.0/(r_large**2)], dim=-1)
        v_inf = model.V_pair(feat_large)
        l_cons = torch.mean(v_inf**2)
        
        l_cons_weight = 0.0 if epoch < 500 else 10.0
        loss = l_dyn * 1e4 + l_cons * l_cons_weight
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()
        
        if epoch % 200 == 0: 
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Dyn: {l_dyn.item():.8f} | Cons: {l_cons.item():.6f}")

    print("\nDistilling Symbolic Force...")
    model.eval()
    
    # Data-driven sampling for r
    diff = p_s.unsqueeze(2) - p_s.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    mask = dist > 0.01
    r_samples = dist[mask].detach().numpy()
    if len(r_samples) > 2000:
        r_samples = np.random.choice(r_samples, 2000, replace=False)
    r_samples = np.sort(r_samples).reshape(-1, 1)

    # Calculate force F(r) = -dV/dr from the model
    r_torch = torch.tensor(r_samples, dtype=torch.float32, requires_grad=True)
    # To get force F(r) = -dV/dr, we pass [r, 1/r, 1/r^2] to V_pair
    feat = torch.cat([r_torch, 1.0/r_torch, 1.0/(r_torch**2)], dim=-1)
    v_vals = model.V_pair(feat)
    force_vals = -torch.autograd.grad(v_vals.sum(), r_torch)[0].detach().numpy().flatten()

    # Symbolic Regression on Force
    est = SymbolicRegressor(population_size=2000, generations=50,
                            function_set=('add', 'sub', 'mul', 'div', 'inv', 'neg', square),
                            n_jobs=-1, parsimony_coefficient=0.001, 
                            verbose=0, random_state=42)
    est.fit(r_samples, force_vals)
    best_expr = str(est._program)

    try:
        locals_dict = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b, 
                       'div':lambda a,b:a/b, 'inv':lambda a:1/a, 'neg':lambda a:-a,
                       'sq':lambda a:a**2, 'X0':sp.Symbol('r')}
        expr = sp.simplify(sp.sympify(best_expr, locals=locals_dict))
        print(f"SUCCESS: Discovered F_scaled(r_scaled) = {expr}")
        
        mse = validate_discovered_law(expr, mode, scaler)
        
        # Visualization
        r_plot = np.linspace(r_samples.min(), r_samples.max(), 100)
        r_plot_phys = r_plot * scaler.p_scale
        
        v_gt = sim.get_ground_truth_potential(r_plot_phys)
        
        with torch.no_grad():
            r_t = torch.tensor(r_plot.reshape(-1, 1), dtype=torch.float32)
            feat_p = torch.cat([r_t, 1.0/r_t, 1.0/(r_t**2)], dim=-1)
            v_nn_scaled = model.V_pair(feat_p).numpy().flatten()
            # Unscale V: V_phys = V_scaled * (p_scale^2 / dt^2)
            v_nn = v_nn_scaled * (scaler.p_scale**2 / sim.dt**2)
            # Offset to match GT for comparison
            v_nn -= (v_nn[-1] - v_gt[-1])

        # Discovered V from F integration
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
        plt.ylim(v_gt.min() - 1, v_gt.max() + 1)
        plt.title(f"Discovery Result: {mode} (MSE: {mse:.6f})")
        plt.legend()
        plt.savefig(f"discovery_{mode}.png")
        plt.close()

    except Exception as e: print(f"Symbolic recovery failed: {best_expr} | Error: {e}")

if __name__ == "__main__":
    for m in ['spring', 'lj']:
        train_discovery(mode=m)