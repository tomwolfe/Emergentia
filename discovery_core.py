import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import time
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. PROTECTED CUSTOM FUNCTIONS
def _protected_sq(x):
    return np.square(np.clip(x, -1e3, 1e3))
square = make_function(function=_protected_sq, name='sq', arity=1)

def _protected_inv(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 0.05, 1.0/x, 0.0)
inv = make_function(function=_protected_inv, name='inv', arity=1)

# 2. ENHANCED SIMULATOR
class PhysicsSim:
    def __init__(self, n=12, mode='lj', seed=None, device='cpu'):
        if seed is not None: 
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.device = device
        self.n, self.mode, self.dt = n, mode, 0.01
        scale = 2.0 if mode == 'spring' else 4.0
        self.pos = torch.rand((n, 2), device=self.device, dtype=torch.float32) * scale
        self.vel = torch.randn((n, 2), device=self.device, dtype=torch.float32) * 0.05
    
    def compute_forces(self, pos):
        n = pos.size(0)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=1e-8)
        mask = (~torch.eye(n, device=self.device).bool()).unsqueeze(-1)
        
        if self.mode == 'spring':
            f = -10.0 * (dist - 1.0) * (diff / dist_clip)
        else:
            d_inv = 1.0 / torch.clamp(dist, min=0.5, max=5.0)
            f = 24.0 * (2 * torch.pow(d_inv, 13) - torch.pow(d_inv, 7)) * (diff / dist_clip)
        
        return torch.sum(f * mask, dim=1)

    def generate(self, steps=1000):
        traj_p = torch.zeros((steps, self.n, 2), device=self.device)
        traj_v = torch.zeros((steps, self.n, 2), device=self.device)
        curr_pos, curr_vel = self.pos.clone(), self.vel.clone()
        
        with torch.no_grad():
            for i in range(steps):
                f = self.compute_forces(curr_pos)
                curr_vel += f * self.dt
                curr_pos += curr_vel * self.dt
                traj_p[i], traj_v[i] = curr_pos, curr_vel
        return traj_p, traj_v

class TrajectoryScaler:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.p_mid, self.p_scale = 0, 1.0

    def fit(self, p, v):
        self.p_mid = p.mean(dim=(0, 1), keepdim=True)
        self.p_scale = p.std().item() + 1e-6

    def transform(self, p, v):
        return (p - self.p_mid) / self.p_scale, v * (self.dt / self.p_scale)

def validate_discovered_law(expr, mode, n_particles=12, device='cpu'):
    r_sym = sp.Symbol('r')
    try:
        f_torch_func = sp.lambdify(r_sym, expr, 'torch')
    except:
        return 1e6

    sim_gt = PhysicsSim(n=n_particles, mode=mode, seed=42, device=device)
    p_gt, _ = sim_gt.generate(steps=50)
    curr_pos, curr_vel = sim_gt.pos.clone(), sim_gt.vel.clone()
    p_disc = torch.zeros((50, n_particles, 2), device=device)
    
    with torch.inference_mode():
        for i in range(50):
            diff = curr_pos.unsqueeze(1) - curr_pos.unsqueeze(0)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            dist_clip = torch.clamp(dist, min=1e-8)
            mask = (~torch.eye(n_particles, device=device).bool()).unsqueeze(-1)
            
            try:
                mag = f_torch_func(dist)
                if not isinstance(mag, torch.Tensor): mag = torch.full_like(dist, float(mag))
                mag = torch.nan_to_num(mag, nan=0.0, posinf=500.0, neginf=-500.0)
                f = torch.sum(mag * (diff / dist_clip) * mask, dim=1)
            except: f = torch.zeros_like(curr_pos)
            
            curr_vel += f * sim_gt.dt
            curr_pos += curr_vel * sim_gt.dt
            p_disc[i] = curr_pos
    
    return np.nan_to_num(torch.mean((p_gt - p_disc)**2).item(), nan=1e6)

# 2. CORE ARCHITECTURE
class DiscoveryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.V_pair = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(), 
            nn.Linear(64, 64), nn.Tanh(), 
            nn.Linear(64, 1)
        )

    def get_forces(self, pos_scaled):
        pos_scaled = pos_scaled.requires_grad_(True)
        B, N, D = pos_scaled.shape
        diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1)
        dist = torch.clamp(torch.norm(diff, dim=-1, keepdim=True), min=0.05)
        
        mask = ~torch.eye(N, device=pos_scaled.device).bool()
        dist_flat = dist[:, mask].view(B, -1, 1)
        
        feat = torch.cat([dist_flat, 1.0/dist_flat, 1.0/(dist_flat**2)], dim=-1)
        V = self.V_pair(feat).sum(dim=1) * 0.5
        grad = torch.autograd.grad(V.sum(), pos_scaled, create_graph=True)[0]
        return -grad

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    t_start = time.time()
    sim = PhysicsSim(mode=mode, device=device)
    p_traj, v_traj = sim.generate(1000)
    scaler = TrajectoryScaler(dt=sim.dt)
    scaler.fit(p_traj, v_traj)
    p_s, v_s = scaler.transform(p_traj, v_traj)
    
    model = DiscoveryNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"--- Training: {mode} on {device} ---")
    best_loss, patience = float('inf'), 0
    for epoch in range(1001):
        idxs = np.random.randint(0, len(p_s)-1, size=64)
        p_batch = torch.cat([p_s[idxs], p_s[idxs+1]], dim=0)
        a_all = model.get_forces(p_batch)
        a, a_next = a_all[:64], a_all[64:]
        
        scale_f = 1e4 if mode == 'spring' else 1e5
        loss = scale_f * (torch.nn.functional.mse_loss(p_s[idxs] + v_s[idxs] + 0.5*a, p_s[idxs+1]) + 
                          torch.nn.functional.mse_loss(v_s[idxs] + 0.5*(a + a_next), v_s[idxs+1]))
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        if loss.item() < best_loss * 0.999:
            best_loss, patience = loss.item(), 0
        else: patience += 1
        
        if patience >= 150:
            print(f"[{mode}] Early stopping at {epoch}")
            break
        if epoch % 500 == 0: print(f"[{mode}] Epoch {epoch} | Loss: {loss.item():.2e}")

    # Vectorized Symbolic Extraction
    r_phys = np.linspace(0.6 if mode == 'lj' else 0.2, 4.0, 300).astype(np.float32).reshape(-1, 1)
    r_scaled = torch.tensor(r_phys / scaler.p_scale, dtype=torch.float32, device=device, requires_grad=True)
    
    # Create a batch of pairs to extract the force curve in one pass
    # Shape: (300, 2, 2) -> 300 samples, 2 particles each
    p_pairs = torch.zeros((len(r_scaled), 2, 2), device=device)
    p_pairs[:, 1, 0] = r_scaled.squeeze()
    
    f_nn_scaled = model.get_forces(p_pairs)
    # The force on particle 1 along x-axis is our radial force
    f_nn_phys = -f_nn_scaled[:, 1, 0].detach().cpu().numpy() * (scaler.p_scale / (sim.dt**2))
    f_nn_phys = np.clip(f_nn_phys, -1000, 1000)

    best_expr, best_mse = None, float('inf')
    for p_coeff in [1e-3, 1e-4]:
        est = SymbolicRegressor(population_size=600, generations=40,
                                function_set=('add', 'sub', 'mul', 'div', square, inv),
                                metric='mse', max_samples=0.8, n_jobs=1, 
                                parsimony_coefficient=p_coeff, random_state=42)
        est.fit(r_phys, f_nn_phys)
        try:
            ld = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b, 
                  'div':lambda a,b:a/b, 'sq':lambda a:a**2, 'inv':lambda a:1./a, 'X0':sp.Symbol('r')}
            expr = sp.simplify(sp.sympify(str(est._program), locals=ld))
            mse = validate_discovered_law(expr, mode, device=device)
            if mse < best_mse:
                best_mse, best_expr = mse, expr
        except: continue

    res = f"Result {mode}: {best_expr} | MSE: {best_mse:.2e}"
    print(res)
    return res

if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(train_discovery, ['spring', 'lj']))
    print(f"\n--- SUMMARY (Total Time: {time.time()-t0:.2f}s) ---\\n" + "\n".join(results))