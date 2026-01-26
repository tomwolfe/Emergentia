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
    # and known issues with inductor backend as mentioned in original code
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

            # Extract coefficients for negative powers of r in multiple ways to handle equivalent forms
            coeff_r_neg13 = expanded.coeff(r**(-13))

            # Also check for 1/r^13 form
            if coeff_r_neg13 == 0 or coeff_r_neg13 is None:
                # Convert expression to a form where we can extract coefficients of 1/r^13
                # We need to collect terms that have r^(-13)
                collected = sp.collect(expanded, r**(-13), evaluate=False)
                if r**(-13) in collected:
                    coeff_r_neg13 = collected[r**(-13)]
                else:
                    # Try to find coefficient of 1/r^13 by rewriting
                    simplified_expr = sp.simplify(expanded)
                    # Look for terms that contain r^(-13) or 1/r^13
                    terms = simplified_expr.as_ordered_terms() if hasattr(simplified_expr, 'as_ordered_terms') else [simplified_expr]

                    for term in terms:
                        if r**(-13) in sp.preorder_traversal(term):
                            coeff_r_neg13 = term.coeff(r**(-13))
                            break
                        elif sp.Mul(1, r**(-13)).has(r**(-13)) and term.has(r**(-13)):
                            # Check if term is of the form coeff * r^(-13)
                            if term.as_coeff_mul(r)[0] != 0:
                                coeff_r_neg13 = term.as_coeff_mul(r)[0]
                                break

            # Similarly for r^(-7)
            coeff_r_neg7 = expanded.coeff(r**(-7))
            if coeff_r_neg7 == 0 or coeff_r_neg7 is None:
                collected = sp.collect(expanded, r**(-7), evaluate=False)
                if r**(-7) in collected:
                    coeff_r_neg7 = collected[r**(-7)]
                else:
                    # Try to find coefficient of 1/r^7 by rewriting
                    simplified_expr = sp.simplify(expanded)
                    terms = simplified_expr.as_ordered_terms() if hasattr(simplified_expr, 'as_ordered_terms') else [simplified_expr]

                    for term in terms:
                        if r**(-7) in sp.preorder_traversal(term):
                            coeff_r_neg7 = term.coeff(r**(-7))
                            break
                        elif sp.Mul(1, r**(-7)).has(r**(-7)) and term.has(r**(-7)):
                            # Check if term is of the form coeff * r^(-7)
                            if term.as_coeff_mul(r)[0] != 0:
                                coeff_r_neg7 = term.as_coeff_mul(r)[0]
                                break

            # Alternative approach: convert to polynomial in r^-1
            # This handles cases like 1/r^13 which is (r^-1)^13
            r_inv = sp.Symbol('r_inv')
            expr_substituted = expanded.subs(r**(-1), r_inv)
            # Now extract coefficients of r_inv^13 and r_inv^7
            if coeff_r_neg13 == 0 or coeff_r_neg13 is None:
                coeff_r_neg13 = expr_substituted.coeff(r_inv**13)
            if coeff_r_neg7 == 0 or coeff_r_neg7 is None:
                coeff_r_neg7 = expr_substituted.coeff(r_inv**7)

            return {
                'coeff_r_neg13': float(coeff_r_neg13) if coeff_r_neg13 and coeff_r_neg13 != 0 else 0,
                'coeff_r_neg7': float(coeff_r_neg7) if coeff_r_neg7 and coeff_r_neg7 != 0 else 0,
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
    patience_counter = 0
    patience_limit = 100  # Number of epochs to wait before considering convergence

    for epoch in range(epochs):
        idxs = np.random.randint(0, p_s.shape[0], size=512)
        f_pred = model(p_s[idxs])
        loss = torch.nn.functional.mse_loss(f_pred, f_s[idxs])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(loss.detach())
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0  # Reset patience counter when we find a better loss
        else:
            patience_counter += 1

        if epoch % 200 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.2e}")

        # Early stopping: if loss is below threshold, break the loop
        threshold = 0.05 if mode == 'spring' else 0.5
        if loss.item() < threshold:
            print(f"Early stopping at epoch {epoch}, loss: {loss.item():.2e}")
            break

        # Additional early stopping based on patience
        if patience_counter >= patience_limit:
            print(f"No improvement for {patience_limit} epochs, stopping at epoch {epoch}")
            break

    threshold = 0.05 if mode == 'spring' else 0.5

    # Short-Circuit: If the 128-hidden network reaches the threshold, skip the 32-hidden network training entirely
    # (Since we only enter this condition if best_loss > threshold, the short-circuit is already implemented)
    if best_loss > threshold:
        print(f"Initial NN Loss {best_loss:.4f} > {threshold}, trying smaller hidden layer...")
        # Retrain with smaller hidden layer (32 neurons) to force smoother representation
        # Use even smaller learning rate for fine-tuning
        model = DiscoveryNet(n_particles=sim.n, hidden_size=32).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=50, factor=0.5)

        best_loss = 1e6
        patience_counter = 0
        patience_limit = 100  # Number of epochs to wait before considering convergence

        for epoch in range(epochs):
            idxs = np.random.randint(0, p_s.shape[0], size=512)
            f_pred = model(p_s[idxs])
            loss = torch.nn.functional.mse_loss(f_pred, f_s[idxs])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(loss.detach())
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0  # Reset patience counter when we find a better loss
            else:
                patience_counter += 1

            if epoch % 200 == 0: print(f"Epoch {epoch} (smaller net) | Loss: {loss.item():.2e}")

            # Early stopping: if loss is below threshold, break the loop
            if loss.item() < threshold:
                print(f"Early stopping at epoch {epoch} (smaller net), loss: {loss.item():.2e}")
                break

            # Additional early stopping based on patience
            if patience_counter >= patience_limit:
                print(f"No improvement for {patience_limit} epochs, stopping at epoch {epoch} (smaller net)")
                break

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

        # Basis Pruning: In the LJ mode, limit the X_basis sent to SymbolicRegressor to only the r^{-7} and r^{-13} terms
        if mode == 'lj':
            # Select only the r^{-7} and r^{-13} columns from the features
            # The features are [dist, inv_r, inv_r**6, inv_r**12, inv_r**7, inv_r**13]
            # So we want indices 4 (inv_r**7 = r^{-7}) and 5 (inv_r**13 = r^{-13})
            X_basis = features[:, [4, 5]].cpu().numpy()
        else:
            # For other modes, use all features
            X_basis = features.cpu().numpy()

    # Use the basis functions as input variables for symbolic regression
    # For LJ mode, only use basic arithmetic operations since basis functions are provided
    p_coeff = 0.01  # Changed initial p_coeff to 0.01 to favor simpler formulas

    # Restrict function set for LJ mode to prevent 'mathematical loops'
    if mode == 'lj':
        f_set = ['add', 'sub', 'mul']
    else:
        f_set = ['add', 'sub', 'mul', 'div', inv, negpow]

    # Single fit with one retry option instead of the attempt loop
    est = SymbolicRegressor(population_size=1000, generations=15, function_set=f_set,
                            parsimony_coefficient=p_coeff, metric='mse', random_state=42,
                            max_samples=0.9, stopping_criteria=0.0001)
    est.fit(X_basis, f_mag_phys)

    # If the result is poor, do exactly one retry with p_coeff * 5
    if est._program.depth_ > 10:  # If formula depth exceeds 10, try again with higher parsimony
        print(f"Depth {est._program.depth_} too high, retrying with p_coeff * 5")
        est_retry = SymbolicRegressor(population_size=1000, generations=15, function_set=f_set,
                                      parsimony_coefficient=p_coeff * 5, metric='mse', random_state=42,
                                      max_samples=0.9, stopping_criteria=0.0001)
        est_retry.fit(X_basis, f_mag_phys)
        est = est_retry  # Use the retry result

    # Map the symbolic regressor variables (X0, X1, etc.) back to physical variables (r)
    # The X_basis contains features derived from r, so we need to map them back properly
    program_str = str(est._program)

    # Replace X0, X1, etc. with the corresponding physical variable based on the mode
    # For LJ mode, X0 corresponds to r^(-7) and X1 corresponds to r^(-13)
    # For other modes, we need to map appropriately
    if mode == 'lj':
        # In LJ mode, X_basis contains [r^(-7), r^(-13)] at indices [0, 1]
        # So X0 -> r^(-7), X1 -> r^(-13)
        # But we want to express the final formula in terms of r, not r^(-7) or r^(-13)
        # So we need to map X0 to r^(-7) and X1 to r^(-13), then simplify to get in terms of r
        r = sp.Symbol('r')

        # Replace X0 with r^(-7) and X1 with r^(-13)
        program_str = program_str.replace('X0', 'r**(-7)')
        program_str = program_str.replace('X1', 'r**(-13)')

        # Now parse the expression and simplify
        ld = {'inv': lambda x: 1/x, 'negpow': lambda x,n: x**(-abs(n)), 'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'div': lambda x,y: x/y, 'r': r}
        expr = sp.simplify(sp.sympify(program_str, locals=ld))
    else:
        # For spring mode, X0 might correspond to r itself
        r = sp.Symbol('r')
        # Replace X0 with r (and any other X variables with r if needed)
        program_str = program_str.replace('X0', 'r')
        ld = {'inv': lambda x: 1/x, 'negpow': lambda x,n: x**(-abs(n)), 'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'div': lambda x,y: x/y, 'r': r}
        expr = sp.simplify(sp.sympify(program_str, locals=ld))
    mse = validate_discovered_law(expr, mode, device=device)

    # Extract coefficients and compare against ground truth
    coeffs = extract_coefficients(expr, mode)

    # Calculate coefficient accuracy metrics
    if mode == 'lj':
        # Calculate how close the discovered coefficients are to expected values
        coeff_r_neg13 = coeffs.get('coeff_r_neg13', 0)
        coeff_r_neg7 = coeffs.get('coeff_r_neg7', 0)
        expected_A = coeffs.get('expected_A', 48)
        expected_B = coeffs.get('expected_B', 24)

        # Calculate relative errors for both coefficients
        rel_error_13 = abs(coeff_r_neg13 - expected_A) / abs(expected_A) if expected_A != 0 else float('inf')
        rel_error_7 = abs(coeff_r_neg7 - expected_B) / abs(expected_B) if expected_B != 0 else float('inf')

        # Use the average of both relative errors as the coefficient accuracy
        # This gives a more comprehensive measure of how well both coefficients match
        coeff_accuracy = (rel_error_13 + rel_error_7) / 2.0 if expected_A != 0 and expected_B != 0 else float('inf')
    elif mode == 'spring':
        coeff_r = coeffs.get('coeff_r', 0)
        expected_k = coeffs.get('expected_k', 10)
        coeff_accuracy = abs(coeff_r - expected_k) / abs(expected_k) if expected_k != 0 else float('inf')
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
