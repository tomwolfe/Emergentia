import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import time
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. PROTECTED CUSTOM FUNCTIONS
def _protected_inv(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 0.01, 1.0/x, 0.0)
inv = make_function(function=_protected_inv, name='inv', arity=1)

# 2. ENHANCED SIMULATOR
class PhysicsSim:
    def __init__(self, n=12, mode='lj', seed=None, device='cpu'):
        if seed is not None: 
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.device = device
        self.n, self.mode, self.dt = n, mode, 0.01
        # Use more restricted initialization to keep particles in a reasonable range
        scale = 2.0 if mode == 'spring' else 3.0
        self.pos = torch.rand((n, 2), device=self.device, dtype=torch.float32) * scale
        self.vel = torch.randn((n, 2), device=self.device, dtype=torch.float32) * 0.05
    
    def compute_forces(self, pos):
        n = pos.size(0)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=1e-8)
        mask = (~torch.eye(n, device=self.device).bool()).unsqueeze(-1)
        
        if self.mode == 'spring':
            # F = -k(r - r0) * r_hat
            f = -10.0 * (dist - 1.0) * (diff / dist_clip)
        else:
            # LJ: F = 24 * (2/r^13 - 1/r^7) * r_hat
            d_inv = 1.0 / torch.clamp(dist, min=0.5, max=5.0)
            f = 24.0 * (2 * torch.pow(d_inv, 13) - torch.pow(d_inv, 7)) * (diff / dist_clip)
        
        return torch.sum(f * mask, dim=1)

    def generate(self, steps=1000):
        traj_p = torch.zeros((steps, self.n, 2), device=self.device)
        traj_v = torch.zeros((steps, self.n, 2), device=self.device)
        traj_f = torch.zeros((steps, self.n, 2), device=self.device)
        curr_pos, curr_vel = self.pos.clone(), self.vel.clone()
        
        with torch.no_grad():
            for i in range(steps):
                f = self.compute_forces(curr_pos)
                traj_f[i] = f
                curr_vel += f * self.dt
                curr_pos += curr_vel * self.dt
                traj_p[i], traj_v[i] = curr_pos, curr_vel
        return traj_p, traj_v, traj_f

class TrajectoryScaler:
    def __init__(self):
        self.p_scale = 5.0 # Normalizing by a constant box size/cutoff
        self.f_scale = 100.0 # Force scaling constant

    def transform(self, p, f):
        return p / self.p_scale, f / self.f_scale

    def inverse_transform_f(self, f_scaled):
        return f_scaled * self.f_scale

def validate_discovered_law(expr, mode, n_particles=12, device='cpu'):
    r_sym = sp.Symbol('r')
    try:
        f_torch_func = sp.lambdify(r_sym, expr, 'torch')
    except:
        return 1e6

    sim_gt = PhysicsSim(n=n_particles, mode=mode, seed=42, device=device)
    p_gt, _, _ = sim_gt.generate(steps=100)
    curr_pos, curr_vel = sim_gt.pos.clone(), sim_gt.vel.clone()
    p_disc = torch.zeros((100, n_particles, 2), device=device)
    
    with torch.inference_mode():
        for i in range(100):
            diff = curr_pos.unsqueeze(1) - curr_pos.unsqueeze(0)
            dist = torch.norm(diff, dim=-1, keepdim=True)
            dist_clip = torch.clamp(dist, min=1e-8)
            mask = (~torch.eye(n_particles, device=device).bool()).unsqueeze(-1)
            
            try:
                mag = f_torch_func(dist)
                if not isinstance(mag, torch.Tensor): mag = torch.full_like(dist, float(mag))
                mag = torch.nan_to_num(mag, nan=0.0, posinf=1000.0, neginf=-1000.0)
                f = torch.sum(mag * (diff / dist_clip) * mask, dim=1)
            except: f = torch.zeros_like(curr_pos)
            
            curr_vel += f * sim_gt.dt
            curr_pos += curr_vel * sim_gt.dt
            p_disc[i] = curr_pos
    
    mse = torch.mean((p_gt - p_disc)**2).item()
    return np.nan_to_num(mse, nan=1e6)

# 2. CORE ARCHITECTURE
class DiscoveryNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Direct Force Model (predicts magnitude based on distance)
        self.net = nn.Sequential(
            nn.Linear(1, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, pos_scaled):
        # pos_scaled: (B, N, 2)
        B, N, _ = pos_scaled.shape
        diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1) # (B, N, N, 2)
        dist = torch.norm(diff, dim=-1, keepdim=True) # (B, N, N, 1)
        dist_clip = torch.clamp(dist, min=1e-8)
        
        # Predict force magnitude for each pair
        # We only care about non-self interactions
        mask = (~torch.eye(N, device=pos_scaled.device).bool()).unsqueeze(-1)
        
        # Input features for force magnitude: just distance for now
        # We can add 1/r or other features if needed, but let's see if MLP learns it.
        # Physics-informed features often help.
        mag = self.net(dist) # (B, N, N, 1)
        
        # Total force on each particle: sum_j mag(r_ij) * (pos_i - pos_j) / r_ij
        # Actually forces are usually F_ij = f(r_ij) * (r_i - r_j)/r_ij
        # The prompt says predict force vector.
        
        pair_forces = mag * (diff / dist_clip) * mask # (B, N, N, 2)
        total_force = torch.sum(pair_forces, dim=2) # (B, N, 2)
        return total_force

    def predict_mag(self, r_scaled):
        # r_scaled: (M, 1)
        return self.net(r_scaled)

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    sim = PhysicsSim(mode=mode, device=device)
    p_traj, v_traj, f_traj = sim.generate(2000)
    scaler = TrajectoryScaler()
    p_s, f_s = scaler.transform(p_traj, f_traj)
    
    model = DiscoveryNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print(f"--- Training: {mode} on {device} ---")
    best_loss = float('inf')
    
    # Dataset Preparation
    dataset_size = p_s.shape[0]
    
    for epoch in range(2001):
        idxs = np.random.randint(0, dataset_size, size=64)
        p_batch = p_s[idxs]
        f_batch = f_s[idxs]
        
        f_pred = model(p_batch)
        loss = torch.nn.functional.mse_loss(f_pred, f_batch)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if epoch % 500 == 0: 
            print(f"[{mode}] Epoch {epoch} | Loss: {loss.item():.2e}")

    # Symbolic Extraction
    # We want to fit the force magnitude function: F_mag(r)
    r_phys = np.linspace(0.6 if mode == 'lj' else 0.2, 4.0, 500).astype(np.float32).reshape(-1, 1)
    r_scaled = torch.tensor(r_phys / scaler.p_scale, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        f_mag_scaled = model.predict_mag(r_scaled).cpu().numpy()
    
    f_mag_phys = scaler.inverse_transform_f(f_mag_scaled)

    est = SymbolicRegressor(population_size=2000, generations=100,
                            function_set=('add', 'sub', 'mul', 'div', inv),
                            metric='mse', max_samples=0.9, n_jobs=1, 
                            parsimony_coefficient=0.01, random_state=42)
    est.fit(r_phys, f_mag_phys.ravel())
    
    ld = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b, 
          'div':lambda a,b:a/b, 'inv':lambda a:1./a, 'X0':sp.Symbol('r')}
    
    raw_str = str(est._program)
    # Basic cleanup for sympy
    expr = sp.simplify(sp.sympify(raw_str, locals=ld))
    mse = validate_discovered_law(expr, mode, device=device)

    res = f"Result {mode}: {expr} | MSE: {mse:.2e}"
    print(res)
    return res

if __name__ == "__main__":
    t0 = time.time()
    res1 = train_discovery('spring')
    res2 = train_discovery('lj')
    print(f"\n--- SUMMARY (Total Time: {time.time()-t0:.2f}s) ---")
    print(res1)
    print(res2)
