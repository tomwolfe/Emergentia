import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import time
import pandas as pd
import warnings
import os
import datetime

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. PROTECTED CUSTOM FUNCTIONS
def _protected_inv(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 0.01, 1.0/x, 0.0)
inv = make_function(function=_protected_inv, name='inv', arity=1)

def _protected_negpow(x, n):
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        abs_x = np.abs(x)
        abs_x = np.where(abs_x < 1e-6, 1e-6, abs_x)
        result = np.power(abs_x, -np.abs(n))
        result = np.where(np.isfinite(result), result, 0.0)
        result = np.where(np.abs(result) < 1e10, result, 0.0)
        return result
negpow = make_function(function=_protected_negpow, name='negpow', arity=2)

# 2. ENHANCED SIMULATOR
class PhysicsSim:
    def __init__(self, n=8, mode='lj', seed=None, device='cpu'):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.device = device
        self.n, self.mode, self.dt = n, mode, 0.01
        scale = 2.0 if mode == 'spring' else 3.5
        self.pos = torch.rand((n, 2), device=self.device, dtype=torch.float32) * scale
        self.vel = torch.randn((n, 2), device=self.device, dtype=torch.float32) * 0.1
        self.mask = (~torch.eye(n, device=self.device).bool()).unsqueeze(-1)

    def compute_forces(self, pos):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=1e-6)
        if self.mode == 'spring':
            f_mag = -10.0 * (dist - 1.0)
        else:
            d_inv = 1.0 / torch.clamp(dist, min=0.8, max=5.0)
            f_mag = 24.0 * (2 * torch.pow(d_inv, 13) - torch.pow(d_inv, 7))
        f = f_mag * (diff / dist_clip) * self.mask
        return torch.sum(f, dim=1)

    def generate(self, steps=1000):
        traj_p = torch.zeros((steps, self.n, 2), device=self.device)
        traj_f = torch.zeros((steps, self.n, 2), device=self.device)
        curr_pos, curr_vel = self.pos.clone(), self.vel.clone()
        with torch.inference_mode():
            for i in range(steps):
                f = self.compute_forces(curr_pos)
                traj_f[i] = f
                curr_vel += f * self.dt
                curr_pos += curr_vel * self.dt
                curr_pos = torch.clamp(curr_pos, -2.0, 6.0)
                traj_p[i] = curr_pos
        return traj_p, traj_f

class TrajectoryScaler:
    def __init__(self, mode='spring'):
        self.p_scale = 5.0
        self.f_scale = 50.0 if mode == 'spring' else 500.0
    def transform(self, p, f): return p / self.p_scale, f / self.f_scale
    def inverse_transform_f(self, f_scaled): return f_scaled * self.f_scale

# 3. CORE ARCHITECTURE
class DiscoveryNet(nn.Module):
    def __init__(self, n_particles=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def _get_features(self, dist):
        dist_clip = torch.clamp(dist, min=0.5)
        inv_r = 1.0 / dist_clip
        return torch.cat([dist, inv_r, inv_r**6, inv_r**12, inv_r**7, inv_r**13], dim=-1)

    def forward(self, pos_scaled):
        diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        feat = self._get_features(dist)
        mag = self.net(feat)
        mask = (~torch.eye(pos_scaled.shape[1], device=pos_scaled.device).bool()).unsqueeze(-1)
        pair_forces = mag * (diff / torch.clamp(dist, min=0.01)) * mask
        return torch.sum(pair_forces, dim=2)

    def predict_mag(self, r_scaled):
        return self.net(self._get_features(r_scaled))

def validate_discovered_law(expr, mode, device='cpu'):
    r_sym = sp.Symbol('r')
    try: f_torch_func = sp.lambdify(r_sym, expr, 'torch')
    except: return 1e6
    sim_gt = PhysicsSim(n=8, mode=mode, seed=42, device=device)
    p_gt, _ = sim_gt.generate(steps=50)
    curr_pos, curr_vel = sim_gt.pos.clone(), sim_gt.vel.clone()
    p_disc = torch.zeros_like(p_gt)
    mask = (~torch.eye(8, device=device).bool()).unsqueeze(-1)
    with torch.inference_mode():
        for i in range(50):
            diff = curr_pos.unsqueeze(1) - curr_pos.unsqueeze(0)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            dist_clip = torch.clamp(dist, min=0.1)
            try:
                mag = f_torch_func(dist_clip)
                if not isinstance(mag, torch.Tensor): mag = torch.full_like(dist_clip, float(mag))
                mag = torch.nan_to_num(mag, nan=0.0, posinf=1000.0, neginf=-1000.0)
                f = torch.sum(mag * (diff / dist_clip) * mask, dim=1)
            except: f = torch.zeros_like(curr_pos)
            curr_vel += f * sim_gt.dt
            curr_pos += curr_vel * sim_gt.dt
            p_disc[i] = curr_pos
    return torch.mean((p_gt - p_disc)**2).item()

# 4. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    sim = PhysicsSim(mode=mode, device=device)
    p_traj, f_traj = sim.generate(1200)
    scaler = TrajectoryScaler(mode=mode)
    p_s, f_s = scaler.transform(p_traj, f_traj)
    model = DiscoveryNet(n_particles=sim.n).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=50, factor=0.5)
    
    print(f"\n--- Training {mode} on {device} ---")
    best_loss = 1e6
    epochs = 400 if mode == 'spring' else 1200
    for epoch in range(epochs):
        idxs = np.random.randint(0, p_s.shape[0], size=512)
        f_pred = model(p_s[idxs])
        loss = torch.nn.functional.mse_loss(f_pred, f_s[idxs])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(loss)
        if loss.item() < best_loss: best_loss = loss.item()
        if epoch % 200 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.2e}")
    
    threshold = 0.05 if mode == 'spring' else 0.5
    if best_loss > threshold:
        print(f"Skipping SR: NN Loss {best_loss:.4f} > {threshold}")
        return f"Fail: {mode}", best_loss, 0, "N/A", 1e6

    r_phys = np.linspace(0.9, 3.5, 300).astype(np.float32).reshape(-1, 1)
    with torch.no_grad():
        f_mag_phys = scaler.inverse_transform_f(model.predict_mag(torch.tensor(r_phys/scaler.p_scale, device=device))).cpu().numpy().ravel()

    p_coeff, f_set = 0.005, ['add', 'sub', 'mul']
    if mode == 'lj': f_set += ['div', inv, negpow]
    
    for attempt in range(5):
        est = SymbolicRegressor(population_size=2000, generations=30, function_set=f_set,
                                parsimony_coefficient=p_coeff, metric='mse', random_state=42)
        est.fit(r_phys, f_mag_phys)
        if est._program.depth_ <= 10: break
        p_coeff *= 2
        print(f"Attempt {attempt}: Depth {est._program.depth_} too high, p_coeff -> {p_coeff}")

    ld = {'inv': lambda x: 1/x, 'negpow': lambda x,n: x**(-abs(n)), 'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'div': lambda x,y: x/y}
    expr = sp.simplify(sp.sympify(str(est._program), locals=ld))
    mse = validate_discovered_law(expr, mode, device=device)
    
    res_data = {"Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Mode": mode, "NN_Loss": best_loss, "Parsimony_Coeff": p_coeff, "Final_Formula": str(expr), "MSE": mse}
    pd.DataFrame([res_data]).to_csv('experiment_results.csv', mode='a', header=not os.path.exists('experiment_results.csv'), index=False)
    print(f"Discovered {mode}: {expr} | MSE: {mse:.2e}")
    return f"Success: {mode}", best_loss, p_coeff, expr, mse

if __name__ == "__main__":
    results = [train_discovery('spring'), train_discovery('lj')]
    print("\n--- Final Results ---")
    for r in results: print(r)



if __name__ == "__main__":
    results = [train_discovery('spring'), train_discovery('lj')]
    print("\n--- Final Results ---")
    for r in results: print(r)