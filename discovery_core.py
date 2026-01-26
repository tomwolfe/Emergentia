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

warnings.filterwarnings('ignore', category=RuntimeWarning)

def optional_compile(func, **kwargs):
    # Disable torch.compile to avoid pickling issues with multiprocessing
    # and known issues with inductor backend
    return func

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

    @optional_compile
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

    @optional_compile
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
    def __init__(self, n_particles=8, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
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
    try:
        # Apply mathematical simplification and expansion to the expression
        simplified_expr = sp.simplify(expr)
        expanded_expr = sp.expand(simplified_expr)
        f_torch_func = sp.lambdify(r_sym, expanded_expr, 'torch')
        f_torch_func = optional_compile(f_torch_func)
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

def extract_coefficients(expr, mode='lj'):
    """
    Extract coefficients from the discovered symbolic expression and compare against ground truth.
    For LJ mode: expects A*(1/r^13) - B*(1/r^7) where A ≈ 48 and B ≈ 24
    For Spring mode: expects -k*(r-1) where k ≈ 10
    """
    if mode == 'lj':
        # Convert to polynomial form to extract coefficients
        r = sp.Symbol('r')
        try:
            # Expand the expression to get polynomial terms
            expanded = sp.expand(expr)

            # Look for terms with r^(-13) and r^(-7) (which appear as 1/r^13 and 1/r^7)
            # Since we expect the form A*(1/r^13) - B*(1/r^7), we need to extract these coefficients

            # Extract coefficients for negative powers of r
            coeff_r_neg13 = expanded.coeff(r**(-13))
            coeff_r_neg7 = expanded.coeff(r**(-7))

            # Also check for positive powers that might represent equivalent forms
            # The LJ potential is typically expressed as 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
            # which becomes 4*epsilon*(sigma^12*r^(-12) - sigma^6*r^(-6))

            # For the force law, we expect terms like 24*(1/r^13) - 12*(1/r^7) based on derivative
            # Actually, the force is derivative of potential: d/dr[4*epsilon*((sigma/r)^12 - (sigma/r)^6)]
            # = 4*epsilon * (12*sigma^12*r^(-13) - 6*sigma^6*r^(-7))
            # = 48*epsilon*sigma^12*r^(-13) - 24*epsilon*sigma^6*r^(-7)

            return {
                'coeff_r_neg13': float(coeff_r_neg13) if coeff_r_neg13 else 0,
                'coeff_r_neg7': float(coeff_r_neg7) if coeff_r_neg7 else 0,
                'expected_A': 48,  # Expected coefficient for 1/r^13 term
                'expected_B': 24,  # Expected coefficient for 1/r^7 term
            }
        except:
            return {
                'coeff_r_neg13': 0,
                'coeff_r_neg7': 0,
                'expected_A': 48,
                'expected_B': 24,
            }
    elif mode == 'spring':
        # For spring mode, we expect -k*(r-1) where k ≈ 10
        try:
            r = sp.Symbol('r')
            expanded = sp.expand(expr)
            # Extract the coefficient of r and the constant term
            coeff_r = expanded.coeff(r, 1)  # Coefficient of r^1
            const_term = expanded.subs(r, 0)  # Constant term when r=0

            return {
                'coeff_r': float(coeff_r) if coeff_r else 0,
                'const_term': float(const_term) if const_term else 0,
                'expected_k': 10,  # Expected spring constant
            }
        except:
            return {
                'coeff_r': 0,
                'const_term': 0,
                'expected_k': 10,
            }

    return {}

# 4. TRAINING & DISCOVERY
def train_discovery(mode='lj'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    sim = PhysicsSim(mode=mode, device=device)
    p_traj, f_traj = sim.generate(1200)
    scaler = TrajectoryScaler(mode=mode)
    p_s, f_s = scaler.transform(p_traj, f_traj)
    # Train DiscoveryNet with default hidden size first
    model = DiscoveryNet(n_particles=sim.n, hidden_size=128).to(device)
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

    # If neural network loss is still too high, try with smaller hidden layer
    if best_loss > threshold:
        print(f"Initial NN Loss {best_loss:.4f} > {threshold}, trying smaller hidden layer...")
        # Retrain with smaller hidden layer (32 neurons) to force smoother representation
        model = DiscoveryNet(n_particles=sim.n, hidden_size=32).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=50, factor=0.5)

        best_loss = 1e6
        for epoch in range(epochs):
            idxs = np.random.randint(0, p_s.shape[0], size=512)
            f_pred = model(p_s[idxs])
            loss = torch.nn.functional.mse_loss(f_pred, f_s[idxs])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(loss)
            if loss.item() < best_loss: best_loss = loss.item()
            if epoch % 200 == 0: print(f"Epoch {epoch} (smaller net) | Loss: {loss.item():.2e}")

    if best_loss > threshold:
        print(f"Skipping SR: NN Loss {best_loss:.4f} > {threshold}")
        return f"Fail: {mode}", best_loss, 0, "N/A", 1e6

    # Prepare basis functions for symbolic regression to align with neural network features
    r_phys_raw = np.linspace(0.9, 3.5, 300).astype(np.float32)
    r_phys = r_phys_raw.reshape(-1, 1)

    # Create the same basis functions that the neural network uses internally
    dist_tensor = torch.tensor(r_phys / scaler.p_scale, device=device, dtype=torch.float32)
    with torch.no_grad():
        # Get the features that the neural network uses internally
        features = model._get_features(dist_tensor)
        f_mag_phys = scaler.inverse_transform_f(model.predict_mag(torch.tensor(r_phys/scaler.p_scale, device=device))).cpu().numpy().ravel()

        # Convert features to numpy for symbolic regression
        X_basis = features.cpu().numpy()

    # Use the basis functions as input variables for symbolic regression
    # For LJ mode, only use basic arithmetic operations since basis functions are provided
    p_coeff, f_set = 0.005, ['add', 'sub', 'mul']
    if mode != 'lj':  # Only add extra functions for non-LJ modes
        f_set += ['div', inv, negpow]

    # Simplicity First loop: if formula depth exceeds 10, simplify further
    for attempt in range(5):
        est = SymbolicRegressor(population_size=2000, generations=30, function_set=f_set,
                                parsimony_coefficient=p_coeff, metric='mse', random_state=42,
                                max_samples=0.9, stopping_criteria=0.0001)
        est.fit(X_basis, f_mag_phys)
        if est._program.depth_ <= 10: break

        # If depth is still too high, increase parsimony coefficient
        if est._program.depth_ > 10:
            p_coeff *= 10  # Increase by an order of magnitude as required
            print(f"Attempt {attempt}: Depth {est._program.depth_} too high, p_coeff -> {p_coeff}")

            # If we're still having issues after increasing parsimony, simplify function set further
            if attempt >= 2 and mode == 'lj':
                # For LJ mode, restrict to only addition and subtraction after failed attempts
                f_set = ['add', 'sub']
                print(f"Attempt {attempt}: Simplifying function set to {f_set} for LJ mode")

    ld = {'inv': lambda x: 1/x, 'negpow': lambda x,n: x**(-abs(n)), 'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'div': lambda x,y: x/y}
    expr = sp.simplify(sp.sympify(str(est._program), locals=ld))
    mse = validate_discovered_law(expr, mode, device=device)

    # Extract coefficients and compare against ground truth
    coeffs = extract_coefficients(expr, mode)

    # Calculate coefficient accuracy metrics
    if mode == 'lj':
        # Calculate how close the discovered coefficients are to expected values
        coeff_accuracy = min(
            abs(coeffs.get('coeff_r_neg13', 0) - coeffs.get('expected_A', 48)) / coeffs.get('expected_A', 48),
            abs(coeffs.get('coeff_r_neg7', 0) - coeffs.get('expected_B', 24)) / coeffs.get('expected_B', 24)
        ) if coeffs.get('expected_A', 48) != 0 and coeffs.get('expected_B', 24) != 0 else float('inf')
    elif mode == 'spring':
        coeff_accuracy = abs(coeffs.get('coeff_r', 0) - coeffs.get('expected_k', 10)) / coeffs.get('expected_k', 10) if coeffs.get('expected_k', 10) != 0 else float('inf')
    else:
        coeff_accuracy = float('inf')

    res_data = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mode": mode,
        "NN_Loss": best_loss,
        "Parsimony_Coeff": p_coeff,
        "Final_Formula": str(expr),
        "MSE": mse,
        "Coeff_Accuracy": coeff_accuracy,
        "Formula_Depth": est._program.depth_
    }
    pd.DataFrame([res_data]).to_csv('experiment_results.csv', mode='a', header=not os.path.exists('experiment_results.csv'), index=False)

    print(f"Discovered {mode}: {expr} | MSE: {mse:.2e} | Depth: {est._program.depth_}")
    print(f"Coefficient Accuracy: {coeff_accuracy:.3f}")

    # Check success criteria
    validation_success = mse < 0.001 and est._program.depth_ < 8
    if mode == 'lj':
        expected_form = "A*(1/r^13) - B*(1/r^7)"
        print(f"LJ Target: {expected_form} where A≈48, B≈24")
    elif mode == 'spring':
        expected_form = "-k*(r-1)"
        print(f"Spring Target: {expected_form} where k≈10")

    result_status = f"Success: {mode}" if validation_success else f"Partial: {mode}"
    return result_status, best_loss, p_coeff, expr, mse

def run_single_mode(mode):
    """Wrapper function to run a single mode with proper error handling"""
    try:
        return train_discovery(mode)
    except Exception as e:
        # Return a safe error representation that can be pickled
        return f"Error in {mode}: {str(e)}"

if __name__ == "__main__":

    modes = ['spring', 'lj']

    print(f"Starting discovery on {len(modes)} modes in parallel...")

    # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
    # with complex PyTorch objects and compiled functions
    with concurrent.futures.ThreadPoolExecutor() as executor:

        futures = {executor.submit(run_single_mode, mode): mode for mode in modes}

        results = []

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Exception in thread for mode {futures[future]}: {e}")


    print("\n--- Final Results ---")

    for r in results: print(r)
