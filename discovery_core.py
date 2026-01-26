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
            # First, expand and simplify the expression to get it in a standard form
            simplified_expr = sp.simplify(expr)
            expanded_expr = sp.expand(simplified_expr)

            # Try multiple approaches to extract coefficients

            # Approach 1: Direct coefficient extraction
            coeff_r_neg13 = expanded_expr.coeff(r**(-13))
            coeff_r_neg7 = expanded_expr.coeff(r**(-7))

            # Approach 2: If direct extraction failed, try collecting terms
            if coeff_r_neg13 == 0 or coeff_r_neg13 is None or coeff_r_neg13 == 0 or coeff_r_neg7 is None:
                # Collect terms with r^(-13) and r^(-7)
                collected = sp.collect(expanded_expr, [r**(-13), r**(-7)], evaluate=False)

                # The collected dict might have the terms we need
                if len(collected) > 1:
                    # If there are multiple terms, look for the ones with r^(-13) and r^(-7)
                    for term, coeff in collected.items():
                        if term == r**(-13):
                            coeff_r_neg13 = coeff
                        elif term == r**(-7):
                            coeff_r_neg7 = coeff
                        # If terms are products like r^(-13)*some_const, extract the coefficient
                        elif term.has(r**(-13)):
                            coeff_r_neg13 = coeff  # The coefficient part
                        elif term.has(r**(-7)):
                            coeff_r_neg7 = coeff  # The coefficient part

            # Approach 3: If still not found, try to separate terms
            if coeff_r_neg13 == 0 or coeff_r_neg13 is None or coeff_r_neg7 == 0 or coeff_r_neg7 is None:
                # Convert to a form where we can separate terms more easily
                # Handle expressions like A/r^13 - B/r^7
                terms = expanded_expr.as_ordered_terms() if hasattr(expanded_expr, 'as_ordered_terms') else [expanded_expr]

                for term in terms:
                    # Check if the term contains r^(-13)
                    if r**(-13) in sp.preorder_traversal(term):
                        coeff_r_neg13 = term.as_coefficient(r**(-13))
                        if coeff_r_neg13 is None:
                            # If as_coefficient doesn't work, try to extract the coefficient manually
                            # by dividing out the r^(-13) part
                            simplified_term = sp.simplify(term / r**(-13))
                            if simplified_term != 0 and not simplified_term.has(r):
                                coeff_r_neg13 = simplified_term
                    elif r**(-7) in sp.preorder_traversal(term):
                        coeff_r_neg7 = term.as_coefficient(r**(-7))
                        if coeff_r_neg7 is None:
                            # Similar approach for r^(-7)
                            simplified_term = sp.simplify(term / r**(-7))
                            if simplified_term != 0 and not simplified_term.has(r):
                                coeff_r_neg7 = simplified_term

            # Convert to float, handling cases where the coefficient might still be None
            coeff_r_neg13 = float(coeff_r_neg13) if coeff_r_neg13 is not None and coeff_r_neg13 != 0 else 0
            coeff_r_neg7 = float(coeff_r_neg7) if coeff_r_neg7 is not None and coeff_r_neg7 != 0 else 0

            return {
                'coeff_r_neg13': coeff_r_neg13,
                'coeff_r_neg7': coeff_r_neg7,
                'expected_A': 48,  # Expected coefficient for 1/r^13 term
                'expected_B': 24,  # Expected coefficient for 1/r^7 term
            }
        except Exception as e:
            print(f"Error in LJ coefficient extraction: {e}")
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
            # Simplify and expand the expression to get it in standard form
            simplified_expr = sp.simplify(expr)
            expanded_expr = sp.expand(simplified_expr)

            # The expected form is -k*(r-1) = -k*r + k
            # So we need to extract coefficients of r^1 and the constant term
            coeff_r = expanded_expr.coeff(r, 1)  # Coefficient of r^1
            const_term = expanded_expr.subs(r, 0)  # Constant term when r=0

            # For the form -k*r + k, the coefficient of r is -k and the constant is k
            # So the spring constant k is the absolute value of the r coefficient
            k_value = abs(float(coeff_r)) if coeff_r is not None else 0

            return {
                'coeff_r': float(coeff_r) if coeff_r is not None else 0,
                'const_term': float(const_term) if const_term is not None else 0,
                'k_extracted': k_value,
                'expected_k': 10,  # Expected spring constant
            }
        except Exception as e:
            print(f"Error in spring coefficient extraction: {e}")
            return {
                'coeff_r': 0,
                'const_term': 0,
                'k_extracted': 0,
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
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=25, factor=0.5)

    print(f"\n--- Training {mode} on {device} ---")
    best_loss = 1e6
    epochs = 200 if mode == 'spring' else 600  # Reduced epochs
    patience_counter = 0
    patience_limit = 15  # Reduced patience for early stopping
    min_epochs = 20  # Reduced min_epochs
    
    # Track previous losses for early stopping based on minimal improvement
    prev_losses = []
    improvement_threshold = 1e-7

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

        # Track recent losses for improvement checking
        prev_losses.append(loss.item())
        if len(prev_losses) > 15:  # Keep last 15 losses
            prev_losses.pop(0)

        if epoch % 100 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.2e}")

        # Early stopping: if loss is below threshold AND we've trained for minimum epochs, break the loop
        threshold = 0.05 if mode == 'spring' else 0.5
        if loss.item() < threshold and epoch >= min_epochs:
            print(f"Early stopping at epoch {epoch}, loss: {loss.item():.2e}")
            break

        # Additional early stopping based on patience
        if patience_counter >= patience_limit and epoch >= min_epochs:
            print(f"No improvement for {patience_limit} epochs, stopping at epoch {epoch}")
            break
            
        # Check for minimal improvement over recent epochs
        if len(prev_losses) >= 15:
            recent_improvement = abs(prev_losses[0] - prev_losses[-1])
            if recent_improvement < improvement_threshold and epoch >= min_epochs:
                print(f"Loss improvement too small ({recent_improvement:.2e}), stopping at epoch {epoch}")
                break

    threshold = 0.05 if mode == 'spring' else 0.5

    # Check if the loss is within 2x of the threshold - if so, don't retry with smaller network
    if best_loss <= threshold * 2:  # Within 2x of threshold, skip smaller network
        print(f"Initial NN Loss {best_loss:.4f} is within 2x threshold, skipping smaller hidden layer...")
    elif best_loss > threshold:
        print(f"Initial NN Loss {best_loss:.4f} > {threshold}, trying smaller hidden layer...")
        # Retrain with smaller hidden layer (32 neurons) to force smoother representation
        # Use even smaller learning rate for fine-tuning
        model = DiscoveryNet(n_particles=sim.n, hidden_size=32).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=25, factor=0.5)

        best_loss = 1e6
        patience_counter = 0
        patience_limit = 15  # Reduced patience for early stopping
        min_epochs = 20  # Reduced min_epochs
        
        # Track previous losses for early stopping based on minimal improvement
        prev_losses = []
        improvement_threshold = 1e-7

        epochs = 200 if mode == 'spring' else 600  # Same reduced epochs for smaller network

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

            # Track recent losses for improvement checking
            prev_losses.append(loss.item())
            if len(prev_losses) > 15:  # Keep last 15 losses
                prev_losses.pop(0)

            if epoch % 100 == 0: print(f"Epoch {epoch} (smaller net) | Loss: {loss.item():.2e}")

            # Early stopping: if loss is below threshold AND we've trained for minimum epochs, break the loop
            if loss.item() < threshold and epoch >= min_epochs:
                print(f"Early stopping at epoch {epoch} (smaller net), loss: {loss.item():.2e}")
                break

            # Additional early stopping based on patience
            if patience_counter >= patience_limit and epoch >= min_epochs:
                print(f"No improvement for {patience_limit} epochs, stopping at epoch {epoch} (smaller net)")
                break
                
            # Check for minimal improvement over recent epochs
            if len(prev_losses) >= 15:
                recent_improvement = abs(prev_losses[0] - prev_losses[-1])
                if recent_improvement < improvement_threshold and epoch >= min_epochs:
                    print(f"Loss improvement too small ({recent_improvement:.2e}), stopping at epoch {epoch} (smaller net)")
                    break

    if best_loss > threshold:
        print(f"Skipping SR: NN Loss {best_loss:.4f} > {threshold}")
        return f"Fail: {mode}", best_loss, 0, "N/A", 1e6

    # Prepare basis functions for symbolic regression to align with neural network features
    r_phys_raw = np.linspace(0.9, 3.5, 300).astype(np.float32)
    r_phys = r_phys_raw.reshape(-1, 1)

    # Create the same basis functions that the neural network uses internally
    dist_tensor = torch.tensor(r_phys / scaler.p_scale, device=device, dtype=torch.float32)
    with torch.inference_mode():
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
            # For spring mode, we want the deviation from equilibrium (r-1) as the main feature
            # The features are [dist, inv_r, inv_r**6, inv_r**12, inv_r**7, inv_r**13]
            # For spring, we expect a linear relationship with distance from equilibrium (r-1)
            # So we'll create a feature that represents (r-1) where r is the distance
            dist_values = features[:, [0]].cpu().numpy()  # Raw distance values
            equilibrium_deviation = dist_values - 1.0  # (r-1) from equilibrium position
            X_basis = equilibrium_deviation  # Use only (r-1) as the feature

    # Use the basis functions as input variables for symbolic regression
    # For LJ mode, only use basic arithmetic operations since basis functions are provided
    p_coeff = 0.1  # Increased p_coeff to strongly favor simpler formulas

    # Restrict function set for LJ mode to prevent 'mathematical loops'
    if mode == 'lj':
        f_set = ['add', 'sub', 'mul']
        # Reduced population size and generations for efficiency
        population_size = 500
        generations = 10
    else:
        f_set = ['add', 'sub', 'mul']  # Limit operations for spring too to keep it simple
        population_size = 500
        generations = 10

    # Single fit without retry logic
    est = SymbolicRegressor(population_size=population_size, generations=generations, function_set=f_set,
                            parsimony_coefficient=p_coeff, metric='mse', random_state=42,
                            max_samples=0.9, stopping_criteria=0.0001)
    est.fit(X_basis, f_mag_phys)

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

        # Further simplify to ensure we get the expected form: A*(1/r^13) - B*(1/r^7)
        # Convert to a more standard form if needed
        expr = sp.expand(expr)
    else:
        # For spring mode, X0 corresponds to (r-1) where r is the physical distance
        r = sp.Symbol('r')
        # Replace X0 with (r-1) (the deviation from equilibrium)
        program_str = program_str.replace('X0', '(r-1)')

        ld = {'inv': lambda x: 1/x, 'negpow': lambda x,n: x**(-abs(n)), 'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'div': lambda x,y: x/y, 'r': r}
        expr = sp.simplify(sp.sympify(program_str, locals=ld))

        # Expand the expression to get it in a standard form
        expr = sp.expand(expr)

        # For spring mode, we expect the form -k*(r-1), so let's try to put it in that form
        # if possible
        if mode == 'spring':
            # Try to factor or rearrange to get closer to the expected form
            expr = sp.simplify(expr)
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
        # For spring mode, use the extracted k value instead of the raw coefficient
        k_extracted = coeffs.get('k_extracted', 0)
        expected_k = coeffs.get('expected_k', 10)
        coeff_accuracy = abs(k_extracted - expected_k) / abs(expected_k) if expected_k != 0 else float('inf')
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

    # Use ProcessPoolExecutor instead of ThreadPoolExecutor to bypass GIL
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_single_mode, mode): mode for mode in modes}

        results = []

        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Exception in process for mode {futures[future]}: {e}")

    print("\n--- Final Results ---")
    for r in results: 
        print(r)