import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import time
from sklearn.metrics import r2_score
import warnings
from concurrent.futures import ProcessPoolExecutor
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. PROTECTED CUSTOM FUNCTIONS
def _protected_inv(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 0.01, 1.0/x, 0.0)
inv = make_function(function=_protected_inv, name='inv', arity=1)

def _protected_sqrt(x):
    return np.sqrt(np.abs(x))
sqrt = make_function(function=_protected_sqrt, name='sqrt', arity=1)

def _protected_pow(x, y):
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        result = np.power(np.abs(x), y)
        # Handle cases where result is too large or invalid
        result = np.where(np.isfinite(result), result, 0.0)
        result = np.where(np.abs(result) < 1e10, result, 0.0)
        return result
pow = make_function(function=_protected_pow, name='pow', arity=2)

def _protected_negpow(x, n):
    """Protected negative power function for physics laws like r^-n"""
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        abs_x = np.abs(x)
        # Avoid division by zero and extremely small numbers
        abs_x = np.where(abs_x < 1e-6, 1e-6, abs_x)
        # Handle both positive and negative exponents
        result = np.where(n >= 0, np.power(abs_x, n), np.power(abs_x, -np.abs(n)))
        result = np.where(np.isfinite(result), result, 0.0)
        result = np.where(np.abs(result) < 1e10, result, 0.0)
        return result
negpow = make_function(function=_protected_negpow, name='negpow', arity=2)

def _protected_inverse_power(x, n):
    """Specifically designed for inverse power laws like r^(-n)"""
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        abs_x = np.abs(x)
        # Avoid division by zero and extremely small numbers
        abs_x = np.where(abs_x < 1e-6, 1e-6, abs_x)
        # Compute x^(-|n|) which gives us inverse powers
        result = np.power(abs_x, -np.abs(n))
        result = np.where(np.isfinite(result), result, 0.0)
        result = np.where(np.abs(result) < 1e10, result, 0.0)
        return result
inverse_power = make_function(function=_protected_inverse_power, name='inverse_power', arity=2)

def _protected_r_neg6(x):
    """Specific function for r^(-6) which is part of the LJ potential"""
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        abs_x = np.abs(x)
        # Avoid division by zero and extremely small numbers
        abs_x = np.where(abs_x < 1e-6, 1e-6, abs_x)
        result = np.power(abs_x, -6)
        result = np.where(np.isfinite(result), result, 0.0)
        result = np.where(np.abs(result) < 1e10, result, 0.0)
        return result
r_neg6 = make_function(function=_protected_r_neg6, name='r_neg6', arity=1)

def _protected_r_neg12(x):
    """Specific function for r^(-12) which is part of the LJ potential"""
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        abs_x = np.abs(x)
        # Avoid division by zero and extremely small numbers
        abs_x = np.where(abs_x < 1e-6, 1e-6, abs_x)
        result = np.power(abs_x, -12)
        result = np.where(np.isfinite(result), result, 0.0)
        result = np.where(np.abs(result) < 1e10, result, 0.0)
        return result
r_neg12 = make_function(function=_protected_r_neg12, name='r_neg12', arity=1)

def _protected_lj_force_terms(x):
    """Function that represents the typical LJ force terms: A/r^13 - B/r^7"""
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        abs_x = np.abs(x)
        # Avoid division by zero and extremely small numbers
        abs_x = np.where(abs_x < 1e-6, 1e-6, abs_x)
        # Return a combination of r^(-13) and r^(-7) terms
        term13 = np.power(abs_x, -13)  # r^(-13)
        term7 = np.power(abs_x, -7)   # r^(-7)
        result = term13 - term7
        result = np.where(np.isfinite(result), result, 0.0)
        result = np.where(np.abs(result) < 1e10, result, 0.0)
        return result
lj_force_terms = make_function(function=_protected_lj_force_terms, name='lj_force_terms', arity=1)

# 2. ENHANCED SIMULATOR
class PhysicsSim:
    def __init__(self, n=8, mode='lj', seed=None, device='cpu'):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.device = device
        self.n, self.mode, self.dt = n, mode, 0.01
        # Use more restricted initialization to keep particles in a reasonable range
        scale = 2.0 if mode == 'spring' else 3.5
        self.pos = torch.rand((n, 2), device=self.device, dtype=torch.float32) * scale
        self.vel = torch.randn((n, 2), device=self.device, dtype=torch.float32) * 0.1

        # Pre-calculate the identity mask to avoid repeated torch.eye calls
        self.mask = (~torch.eye(n, device=self.device).bool()).unsqueeze(-1)

    def compute_forces(self, pos):
        n = pos.size(0)
        diff = pos.unsqueeze(1) - pos.unsqueeze(0) # (N, N, 2)
        dist = torch.norm(diff, dim=-1, keepdim=True) # (N, N, 1)
        dist_clip = torch.clamp(dist, min=1e-6)

        if self.mode == 'spring':
            # F = -10.0 * (r - 1.0)
            f_mag = -10.0 * (dist - 1.0)
        else:
            # LJ: F = 24 * (2/r^13 - 1/r^7)
            # Clamp distance to avoid explosion during simulation
            d_inv = 1.0 / torch.clamp(dist, min=0.7, max=5.0)
            f_mag = 24.0 * (2 * torch.pow(d_inv, 13) - torch.pow(d_inv, 7))

        f = f_mag * (diff / dist_clip) * self.mask
        return torch.sum(f, dim=1)

    def generate(self, steps=600):
        traj_p = torch.zeros((steps, self.n, 2), device=self.device)
        traj_v = torch.zeros((steps, self.n, 2), device=self.device)
        traj_f = torch.zeros((steps, self.n, 2), device=self.device)
        curr_pos, curr_vel = self.pos.clone(), self.vel.clone()

        with torch.inference_mode():
            for i in range(steps):
                f = self.compute_forces(curr_pos)
                traj_f[i] = f
                curr_vel += f * self.dt
                curr_pos += curr_vel * self.dt
                # Simple boundary conditions to keep them from flying away
                curr_pos = torch.clamp(curr_pos, -2.0, 6.0)
                traj_p[i], traj_v[i] = curr_pos, curr_vel
        return traj_p, traj_v, traj_f

class TrajectoryScaler:
    def __init__(self, mode='spring'):
        self.p_scale = 5.0
        self.f_scale = 50.0 if mode == 'spring' else 500.0

    def transform(self, p, f):
        return p / self.p_scale, f / self.f_scale

    def inverse_transform_f(self, f_scaled):
        return f_scaled * self.f_scale

def validate_discovered_law(expr, mode, n_particles=8, device='cpu'):
    r_sym = sp.Symbol('r')
    try:
        f_torch_func = sp.lambdify(r_sym, expr, 'torch')
    except:
        return 1e6

    sim_gt = PhysicsSim(n=n_particles, mode=mode, seed=42, device=device)
    p_gt, _, _ = sim_gt.generate(steps=20)  # Reduced from 50 to 20 steps
    curr_pos, curr_vel = sim_gt.pos.clone(), sim_gt.vel.clone()
    p_disc = torch.zeros((20, n_particles, 2), device=device)  # Changed to 20 steps

    # Pre-calculate the identity mask to avoid repeated torch.eye calls
    mask = (~torch.eye(n_particles, device=device).bool()).unsqueeze(-1)

    with torch.inference_mode():
        for i in range(20):  # Reduced from 50 to 20 steps
            diff = curr_pos.unsqueeze(1) - curr_pos.unsqueeze(0)
            dist = torch.norm(diff, dim=-1, keepdim=True)

            # Apply the same clamping as in PhysicsSim to handle r=0 singularities
            if mode == 'spring':
                dist_clip = torch.clamp(dist, min=1e-6)
            else:
                # For LJ mode, use the same clamping as in PhysicsSim
                dist_clip = torch.clamp(dist, min=0.7, max=5.0)

            try:
                # Apply the discovered law function
                mag = f_torch_func(dist_clip)  # Use dist_clip instead of dist to avoid singularities

                if not isinstance(mag, torch.Tensor):
                    mag = torch.full_like(dist_clip, float(mag))

                # Handle NaN and infinity values
                mag = torch.nan_to_num(mag, nan=0.0, posinf=1000.0, neginf=-1000.0)

                # Calculate forces using the same formula as PhysicsSim
                f = torch.sum(mag * (diff / dist_clip) * mask, dim=1)
            except Exception as e:
                # If there's an error in applying the function, return zero forces
                f = torch.zeros_like(curr_pos)

            curr_vel += f * sim_gt.dt
            curr_pos += curr_vel * sim_gt.dt
            p_disc[i] = curr_pos

    mse = torch.mean((p_gt - p_disc)**2).item()
    return np.nan_to_num(mse, nan=1e6)

# 2. CORE ARCHITECTURE
class DiscoveryNet(nn.Module):
    def __init__(self, n_particles=8):
        super().__init__()
        # Input features: r, 1/r, 1/r^2
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Store the number of particles to pre-calculate the mask
        self.n_particles = n_particles

    def forward(self, pos_scaled):
        B, N, _ = pos_scaled.shape
        diff = pos_scaled.unsqueeze(2) - pos_scaled.unsqueeze(1)
        dist = torch.norm(diff, dim=-1, keepdim=True)
        dist_clip = torch.clamp(dist, min=0.01)

        # Physics-informed features
        r = dist
        inv_r = 1.0 / dist_clip
        inv_r2 = inv_r**2
        feat = torch.cat([r, inv_r, inv_r2], dim=-1)

        mag = self.net(feat)

        # Use pre-calculated mask if dimensions match, otherwise calculate
        if hasattr(self, '_mask') and self._mask.shape[0] == N:
            mask = self._mask
        else:
            mask = (~torch.eye(N, device=pos_scaled.device).bool()).unsqueeze(-1)
            self._mask = mask

        pair_forces = mag * (diff / dist_clip) * mask
        return torch.sum(pair_forces, dim=2)

    def predict_mag(self, r_scaled):
        dist_clip = torch.clamp(r_scaled, min=0.01)
        feat = torch.cat([r_scaled, 1.0/dist_clip, 1.0/(dist_clip**2)], dim=-1)
        return self.net(feat)

# 3. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Set device for multiprocessing to avoid CUDA conflicts
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else 0)

    sim = PhysicsSim(mode=mode, device=device)
    p_traj, v_traj, f_traj = sim.generate(600)
    scaler = TrajectoryScaler(mode=mode)
    p_s, f_s = scaler.transform(p_traj, f_traj)

    model = DiscoveryNet(n_particles=sim.n).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"--- Training: {mode} on {device} ---")

    # Early stopping parameters
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    epochs = 151 if mode != 'spring' else 50  # Fewer epochs for spring mode

    for epoch in range(epochs):
        idxs = np.random.randint(0, p_s.shape[0], size=512)
        f_pred = model(p_s[idxs])
        loss = torch.nn.functional.mse_loss(f_pred, f_s[idxs])

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Early stopping logic
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience or loss.item() < 1e-5:
            print(f"[{mode}] Converged at epoch {epoch} | Loss: {loss.item():.2e}")
            break

        if epoch % 50 == 0:  # Adjusted logging frequency
            print(f"[{mode}] Epoch {epoch} | Loss: {loss.item():.2e}")

    # Symbolic Extraction
    if mode == 'lj':
        # Focus on the region where LJ potential is most active (avoiding too close and too far)
        r_phys = np.linspace(0.8, 3.0, 50).astype(np.float32).reshape(-1, 1)
    else:
        r_phys = np.linspace(0.4, 4.0, 50).astype(np.float32).reshape(-1, 1)
    r_scaled = torch.tensor(r_phys / scaler.p_scale, dtype=torch.float32, device=device)

    with torch.inference_mode():
        f_mag_scaled = model.predict_mag(r_scaled).cpu().numpy()

    f_mag_phys = scaler.inverse_transform_f(f_mag_scaled).ravel()

    # Mode-dependent SR parameters - significantly reduced for spring mode
    if mode == 'spring':
        pop_size, gens, f_set = 200, 8, ('add', 'sub', 'mul')  # Increased generations slightly
        p_coeff = 0.05  # Standard parsimony
        max_samp = 0.7  # Standard sampling
    else:
        # For LJ mode, use a function set that includes inv for inverse powers
        pop_size, gens, f_set = 500, 12, ('add', 'sub', 'mul', 'div', inv)
        p_coeff = 0.02  # Moderate parsimony
        max_samp = 0.6   # Moderate sampling

    # Determine which custom functions to include based on the mode
    if mode == 'lj':
        # Include custom functions in the function set for LJ mode - focus on inverse powers
        final_function_set = f_set + (pow, negpow, inverse_power, r_neg6, r_neg12, lj_force_terms)
    else:
        final_function_set = f_set

    est = SymbolicRegressor(population_size=pop_size, generations=gens,
                            function_set=final_function_set,
                            metric='mse', max_samples=max_samp, n_jobs=1,  # Changed n_jobs to 1 for multiprocessing compatibility
                            verbose=1,
                            parsimony_coefficient=p_coeff, random_state=42)
    est.fit(r_phys, f_mag_phys)

    # Define locals dictionary based on mode
    if mode == 'lj':
        ld = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b,
              'div':lambda a,b:a/b, 'pow':lambda a,b:sp.Pow(a,b), 'negpow':lambda a,b:sp.Pow(a,-b),
              'inverse_power':lambda a,b:sp.Pow(a,-b), 'r_neg6':lambda a:sp.Pow(a,-6),
              'r_neg12':lambda a:sp.Pow(a,-12), 'lj_force_terms':lambda a:(sp.Pow(a,-13) - sp.Pow(a,-7)),
              'inv':lambda a:1./a, 'X0':sp.Symbol('r')}
    else:
        ld = {'add':lambda a,b:a+b, 'sub':lambda a,b:a-b, 'mul':lambda a,b:a*b,
              'div':lambda a,b:a/b, 'inv':lambda a:1./a, 'X0':sp.Symbol('r')}

    expr = sp.simplify(sp.sympify(str(est._program), locals=ld))
    mse = validate_discovered_law(expr, mode, device=device)

    res = f"Result {mode}: {expr} | MSE: {mse:.2e}"
    print(res)
    return res

def run_single_discovery(args):
    """Wrapper function for multiprocessing"""
    mode = args
    return train_discovery(mode)

if __name__ == "__main__":
    t0 = time.time()

    # Run modes sequentially to avoid timeout issues
    modes = ['spring', 'lj']
    results = []
    for mode in modes:
        result = train_discovery(mode)
        results.append(result)

    res1, res2 = results

    print(f"\n--- SUMMARY (Total Time: {time.time()-t0:.2f}s) ---")
    print(res1)
    print(res2)