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
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import copy

# Set matplotlib backend to avoid GUI issues in multiprocessing
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. ENHANCED SIMULATOR
class PhysicsSim:
    def __init__(self, n=6, mode='lj', seed=None, device='cpu'):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.device = device
        self.n, self.mode, self.dt = n, mode, 0.001 if mode == 'lj' else 0.01
        scale = 2.0 if mode == 'spring' else 3.0
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
            d_inv = 1.0 / torch.clamp(dist, min=0.5, max=5.0)
            f_mag = 48.0 * torch.pow(d_inv, 13) - 24.0 * torch.pow(d_inv, 7)
        f = f_mag * (diff / dist_clip) * self.mask
        return torch.sum(f, dim=1)

    def generate(self, steps=6000):
        traj_p = torch.zeros((steps, self.n, 2), device=self.device, dtype=torch.float32)
        traj_f = torch.zeros((steps, self.n, 2), device=self.device, dtype=torch.float32)
        curr_pos, curr_vel = self.pos.clone(), self.vel.clone()
        with torch.no_grad():
            for i in range(steps):
                traj_p[i] = curr_pos
                f = self.compute_forces(curr_pos)
                traj_f[i] = f
                if i % 100 == 0:
                    curr_vel += torch.randn_like(curr_vel) * (0.6 if self.mode == 'lj' else 0.4)
                curr_vel += f * self.dt
                curr_pos += curr_vel * self.dt
        return traj_p, traj_f

# 3. CORE ARCHITECTURE
class DiscoveryNet(nn.Module):
    def __init__(self, p_scale=1.0, hidden_size=128, mode='lj'):
        super().__init__()
        self.p_scale = p_scale
        self.mode = mode
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
            nn.Linear(hidden_size, 1)
        )
        self.register_buffer('feat_mean', torch.zeros(6))
        self.register_buffer('feat_std', torch.ones(6))

    def _get_raw_features(self, dist_phys):
        d = torch.clamp(dist_phys, min=0.4, max=10.0)
        inv_r = 1.0 / d
        inv_r6 = inv_r**6
        inv_r12 = inv_r6**2
        inv_r7 = inv_r6 * inv_r
        inv_r13 = inv_r12 * inv_r
        return torch.cat([d, inv_r, inv_r6, inv_r12, inv_r7, inv_r13], dim=-1)

    def update_stats(self, dist_phys):
        with torch.no_grad():
            feat_flat = self._get_raw_features(dist_phys).view(-1, 6)
            self.feat_mean.copy_(feat_flat.mean(dim=0))
            self.feat_std.copy_(feat_flat.std(dim=0) + 1e-6)

    def forward(self, pos_scaled):
        pos_phys = pos_scaled * self.p_scale
        diff_phys = pos_phys.unsqueeze(2) - pos_phys.unsqueeze(1)
        dist_phys = torch.norm(diff_phys, dim=-1, keepdim=True)
        feat = (self._get_raw_features(dist_phys) - self.feat_mean) / self.feat_std
        mag_pred = self.net(feat)
        if self.mode == 'lj':
            mag_phys = torch.sign(mag_pred) * (torch.exp(torch.abs(mag_pred)) - 1.0)
        else:
            mag_phys = mag_pred
        mask = (~torch.eye(pos_scaled.shape[1], device=pos_scaled.device).bool()).unsqueeze(-1)
        pair_forces_phys = mag_phys * (diff_phys / torch.clamp(dist_phys, min=0.01)) * mask
        return torch.sum(pair_forces_phys, dim=2)

    def predict_mag_phys(self, dist_phys):
        feat = (self._get_raw_features(dist_phys) - self.feat_mean) / self.feat_std
        mag_pred = self.net(feat)
        if self.mode == 'lj':
            return torch.sign(mag_pred) * (torch.exp(torch.abs(mag_pred)) - 1.0)
        else:
            return mag_pred

def validate_discovered_law(expr, mode):
    r_sym = sp.Symbol('r')
    try: f_func = sp.lambdify(r_sym, expr, 'numpy')
    except: return 1e10
    r_test = np.linspace(0.5, 3.5, 1000).reshape(-1, 1).astype(np.float32)
    if mode == 'spring':
        f_gt = -10.0 * (r_test - 1.0)
    else:
        d_inv = 1.0 / np.clip(r_test, 0.5, 5.0)
        f_gt = 48.0 * np.power(d_inv, 13) - 24.0 * np.power(d_inv, 7)
    try:
        f_pred = f_func(r_test)
        if np.isscalar(f_pred): f_pred = np.full_like(r_test, f_pred)
        f_pred = np.nan_to_num(f_pred, nan=0.0, posinf=1e10, neginf=-1e10)
        mse = np.mean((f_gt - f_pred)**2)
    except: return 1e10
    return float(mse)

def extract_coefficients(expr, mode='lj'):
    r = sp.Symbol('r')
    try:
        expanded = sp.expand(expr)
        if mode == 'lj':
            c13 = float(expanded.coeff(r**-13)) if expanded.has(r**-13) else 0.0
            c7 = float(expanded.coeff(r**-7)) if expanded.has(r**-7) else 0.0
            return {'coeff_r_neg13': c13, 'coeff_r_neg7': c7}
        else:
            ck = float(expanded.coeff(r, 1))
            return {'k_extracted': abs(ck)}
    except: return {}

# 4. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    sim = PhysicsSim(mode=mode, device=device, seed=42)
    p_traj, f_traj = sim.generate(6000)
    p_scale = torch.max(torch.abs(p_traj)).item() or 1.0
    
    if mode == 'lj':
        f_target = torch.sign(f_traj) * torch.log1p(torch.abs(f_traj))
    else:
        f_target = f_traj
    
    p_s, f_target = (p_traj / p_scale).to(device), f_target.to(device)
    model = DiscoveryNet(p_scale=p_scale, hidden_size=128, mode=mode).to(device)
    with torch.no_grad():
        dist_phys_sample = torch.linspace(0.4, 5.0, 3000, device=device).reshape(-1, 1, 1)
        model.update_stats(dist_phys_sample)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=200, factor=0.5)
    
    def get_pred_transformed(pos_s):
        f_phys = model(pos_s)
        if mode == 'lj':
            return torch.sign(f_phys) * torch.log1p(torch.abs(f_phys))
        return f_phys

    print(f"\n--- Training {mode} ---")
    best_loss, best_model_state = 1e10, None
    epochs = 7000 if mode == 'lj' else 4000
    for epoch in range(epochs):
        idxs = np.random.randint(0, p_s.shape[0], size=1024)
        loss = torch.mean((get_pred_transformed(p_s[idxs]) - f_target[idxs])**2)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = copy.deepcopy(model.state_dict())
        if epoch % 1000 == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.2e} | Best: {best_loss:.2e}")

    if best_model_state: model.load_state_dict(best_model_state)

    # SR Phase - Sampling ONLY within the explored range to avoid extrapolation
    r_max = float(torch.max(torch.norm(p_traj.unsqueeze(2)-p_traj.unsqueeze(1), dim=-1)).item())
    r_phys = np.linspace(0.5, min(r_max, 4.0), 1500).reshape(-1, 1).astype(np.float32)
    with torch.no_grad():
        f_mag_phys = model.predict_mag_phys(torch.tensor(r_phys, device=device)).cpu().numpy().ravel()
        if mode == 'lj':
            X_basis = np.stack([r_phys.ravel()**-13, r_phys.ravel()**-7], axis=1)
            f_set = ('add', 'sub', 'mul')
        else:
            X_basis = (r_phys - 1.0)
            f_set = ('add', 'sub', 'mul')

    est = SymbolicRegressor(population_size=10000 if mode == 'lj' else 2500, 
                            generations=1 if mode == 'lj' else 50, 
                            function_set=f_set,
                            parsimony_coefficient=0.01, 
                            metric='mse', random_state=42,
                            init_depth=(1, 2) if mode == 'lj' else (1, 2), 
                            max_samples=0.9, n_jobs=1)
    est.fit(X_basis, f_mag_phys)
    
    r = sp.Symbol('r')
    mapping = {sp.Symbol('X0'): r**-13, sp.Symbol('X1'): r**-7} if mode == 'lj' else {sp.Symbol('X0'): (r-1)}
    expr = sp.sympify(str(est._program), locals={'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y})
    for old, new in mapping.items(): expr = expr.subs(old, new)
    expr = sp.expand(sp.simplify(expr))

    mse = validate_discovered_law(expr, mode)
    coeffs = extract_coefficients(expr, mode)
    if mode == 'lj':
        a, b = coeffs.get('coeff_r_neg13', 0), -coeffs.get('coeff_r_neg7', 0)
        acc = (abs(a - 48)/48 + abs(b - 24)/24) / 2
    else:
        k = coeffs.get('k_extracted', 0)
        acc = abs(k - 10)/10

    print(f"Discovered {mode}: {expr} | MSE: {mse:.2e} | Acc: {acc:.3f}")
    return {"Mode": mode, "Formula": str(expr), "MSE": mse, "Coeff_Accuracy": acc, "Depth": est._program.depth_}

def run_single_mode(mode):
    try: return train_discovery(mode)
    except Exception as e: return {"Mode": mode, "Error": str(e)}

def main():
    multiprocessing.set_start_method('spawn', force=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(run_single_mode, ['spring', 'lj']))
    print("\n--- Final Summary ---", flush=True)
    df = pd.DataFrame([r for r in results if isinstance(r, dict) and "Error" not in r])
    if not df.empty: print(df.to_string(), flush=True)
    else:
        for r in results: print(r, flush=True)

if __name__ == "__main__":
    main()
