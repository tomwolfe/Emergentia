import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
import warnings
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import copy

warnings.filterwarnings('ignore')

class PhysicsSim:
    def __init__(self, n=6, mode='lj', device='cpu'):
        self.device, self.n, self.mode = device, n, mode
        self.dt = 0.001 if mode == 'lj' else 0.01
        self.pos = torch.rand((n, 2), device=device) * (3.0 if mode == 'lj' else 2.5)
        self.vel = torch.randn((n, 2), device=device) * 0.1
        self.mask = (~torch.eye(n, device=device).bool()).unsqueeze(-1)

    def compute_forces(self, pos):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        if self.mode == 'spring':
            f_mag = -10.0 * (dist - 1.0)
        else:
            d_inv = 1.0 / torch.clamp(dist, min=0.5, max=5.0)
            f_mag = 48.0 * torch.pow(d_inv, 13) - 24.0 * torch.pow(d_inv, 7)
        return torch.sum(f_mag * (diff / torch.clamp(dist, min=1e-6)) * self.mask, dim=1)

    def generate(self, steps=5000):
        traj_p, traj_f = torch.zeros((steps, self.n, 2)), torch.zeros((steps, self.n, 2))
        curr_p, curr_v = self.pos.clone(), self.vel.clone()
        for i in range(steps):
            traj_p[i] = curr_p
            f = self.compute_forces(curr_p)
            traj_f[i] = f
            if i % 100 == 0: curr_v += torch.randn_like(curr_v) * (0.5 if self.mode == 'lj' else 0.4)
            curr_v += f * self.dt
            curr_p += curr_v * self.dt
        return traj_p, traj_f

class DiscoveryNet(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1))
        self.register_buffer('mu', torch.zeros(6))
        self.register_buffer('std', torch.ones(6))

    def _get_feats(self, d):
        d = torch.clamp(d, 0.4, 10.0)
        return torch.cat([d, 1/d, 1/d**6, 1/d**12, 1/d**7, 1/d**13], dim=-1)

    def forward(self, p_s, p_scale):
        d_p = torch.norm(p_s.unsqueeze(2) - p_s.unsqueeze(1), dim=-1, keepdim=True) * p_scale
        feat = (self._get_feats(d_p) - self.mu) / self.std
        mag_t = self.net(feat)
        diff_s = p_s.unsqueeze(2) - p_s.unsqueeze(1)
        mask = (~torch.eye(p_s.shape[1], device=p_s.device).bool()).unsqueeze(-1)
        return torch.sum(mag_t * (diff_s / torch.clamp(d_p/p_scale, 1e-3)) * mask, dim=2)

def train_discovery(mode):
    device = torch.device('cpu')
    sim = PhysicsSim(mode=mode, device=device)
    p_t, f_t = sim.generate()
    p_scale = p_t.abs().max().item()
    if mode == 'lj': f_target = torch.sign(f_t) * torch.log1p(torch.abs(f_t))
    else: f_target = f_t / f_t.abs().max().item()
    
    p_s, f_target = p_t / p_scale, f_target
    model = DiscoveryNet()
    with torch.no_grad():
        sample_d = torch.linspace(0.4, 5.0, 2000).view(-1, 1, 1)
        feats = model._get_feats(sample_d).view(-1, 6)
        model.mu.copy_(feats.mean(0)); model.std.copy_(feats.std(0) + 1e-6)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for e in range(6000):
        ix = np.random.randint(0, p_s.shape[0], 1024)
        loss = torch.mean((model(p_s[ix], p_scale) - f_target[ix])**2)
        opt.zero_grad(); loss.backward(); opt.step()

    r_v = np.linspace(0.9, 3.5, 1000).reshape(-1, 1).astype(np.float32)
    with torch.no_grad():
        mag_t = model.net((model._get_feats(torch.tensor(r_v)) - model.mu)/model.std)
        if mode == 'lj': f_p = (torch.sign(mag_t) * (torch.exp(torch.abs(mag_t)) - 1.0)).numpy()
        else: f_p = (mag_t * f_t.abs().max()).numpy()
    
    r = sp.Symbol('r')
    if mode == 'lj':
        X = np.stack([r_v.ravel()**-13, r_v.ravel()**-7], axis=1)
        mapping = {sp.Symbol('X0'): r**-13, sp.Symbol('X1'): r**-7}
    else:
        X = (r_v - 1.0).reshape(-1, 1)
        mapping = {sp.Symbol('X0'): (r-1)}

    est = SymbolicRegressor(population_size=5000, generations=40, 
                            function_set=('add', 'sub', 'mul'), 
                            parsimony_coefficient=0.05, init_depth=(1, 2), random_state=42)
    est.fit(X, f_p.ravel())
    
    expr = sp.sympify(str(est._program), locals={'add':lambda x,y:x+y,'sub':lambda x,y:x-y,'mul':lambda x,y:x*y})
    for k, v in mapping.items(): expr = expr.subs(k, v)
    expr = sp.expand(sp.simplify(expr))
    
    rv_test = np.array([1.0, 1.1, 1.3, 1.5, 2.0])
    try: fv_pred = [float(expr.subs(r, v)) for v in rv_test]
    except: fv_pred = [0.0]*len(rv_test)
    
    if mode == 'lj':
        fv_gt = 48*rv_test**-13 - 24*rv_test**-7
        A = np.stack([rv_test**-13, rv_test**-7], axis=1)
        try:
            c, _, _, _ = np.linalg.lstsq(A, fv_pred, rcond=None)
            acc = 1.0 - (abs(c[0]-48)/48 + abs(-c[1]-24)/24)/2
        except:
            acc = 0.0
    else:
        fv_gt = -10*(rv_test - 1)
        k_ext = -float(expr.coeff(r, 1)) if expr.has(r) else 0.0
        acc = 1.0 - abs(k_ext-10)/10
    
    mse = np.mean((np.array(fv_gt) - np.array(fv_pred))**2)
    print(f"Final {mode}: {expr} | Acc {acc:.3f} | MSE {mse:.2e}")
    return {"Mode": mode, "Formula": str(expr), "MSE": mse, "Coeff_Accuracy": max(0, acc)}

def main():
    multiprocessing.set_start_method('spawn', force=True)
    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(train_discovery, ['spring', 'lj']))
    print("\n--- Summary ---")
    print(pd.DataFrame(results).to_string())

if __name__ == "__main__":
    main()