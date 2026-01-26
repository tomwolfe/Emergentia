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
            # For LJ mode, ensure we properly calculate the force from the potential
            # The Lennard-Jones force is the negative derivative of the potential
            # V(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6), force = -dV/dr
            # This gives us F = 4*epsilon * (12*(sigma^12/r^13) - 6*(sigma^6/r^7))
            # For simplicity, using epsilon=1 and sigma=1, we get F = 48*(1/r^13) - 24*(1/r^7)
            d_inv = 1.0 / torch.clamp(dist, min=0.8, max=5.0)  # Use 0.8 to avoid extreme forces
            # Clamp the powers to prevent overflow
            d_inv_13 = torch.clamp(torch.pow(d_inv, 13), max=1e10)
            d_inv_7 = torch.clamp(torch.pow(d_inv, 7), max=1e6)
            f_mag = 24.0 * (2 * d_inv_13 - d_inv_7)
        f = f_mag * (diff / dist_clip) * self.mask
        return torch.sum(f, dim=1)

    @optional_compile
    def generate(self, steps=1000):
        traj_p = torch.zeros((steps, self.n, 2), device=self.device)
        traj_f = torch.zeros((steps, self.n, 2), device=self.device)
        curr_pos, curr_vel = self.pos.clone(), self.vel.clone()

        # Add warm-up phase with random impulse injections to explore repulsive regions
        with torch.inference_mode():
            for i in range(steps):
                f = self.compute_forces(curr_pos)
                traj_f[i] = f

                # Inject random impulses occasionally to explore high-energy configurations
                if i % 50 == 0:  # Every 50 steps
                    # Random impulse to increase velocity and potentially bring particles closer
                    impulse_factor = 2.0 if self.mode == 'lj' else 1.5  # Higher for LJ to overcome repulsion
                    random_impulse = torch.randn_like(curr_vel) * impulse_factor
                    curr_vel += random_impulse

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
        dist_clip = torch.clamp(dist, min=0.5, max=10.0)  # Clamp both min and max to prevent extreme values
        inv_r = 1.0 / dist_clip
        # Clamp the powers to prevent overflow
        inv_r_6 = torch.clamp(inv_r**6, max=1e6)
        inv_r_12 = torch.clamp(inv_r**12, max=1e12)
        inv_r_7 = torch.clamp(inv_r**7, max=1e7)
        inv_r_13 = torch.clamp(inv_r**13, max=1e13)
        return torch.cat([dist, inv_r, inv_r_6, inv_r_12, inv_r_7, inv_r_13], dim=-1)

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

    # Generate a static vector of 200 points between 0.5 and 3.5 for validation
    r_test = torch.linspace(0.5, 3.5, 200, device=device, dtype=torch.float32).reshape(-1, 1)

    # Compute the ground truth forces based on the physics mode
    with torch.no_grad():
        if mode == 'spring':
            # Ground truth for spring: -10.0 * (r - 1.0)
            f_ground_truth = -10.0 * (r_test - 1.0)
        else:  # LJ mode
            # Ground truth for LJ: 24.0 * (2 * (1/r)^13 - (1/r)^7)
            r_inv = 1.0 / torch.clamp(r_test, min=0.8, max=5.0)
            r_inv_13 = torch.clamp(torch.pow(r_inv, 13), max=1e10)
            r_inv_7 = torch.clamp(torch.pow(r_inv, 7), max=1e6)
            f_ground_truth = 24.0 * (2 * r_inv_13 - r_inv_7)

        # Compute the predicted forces using the discovered formula
        try:
            f_predicted = f_torch_func(r_test)
            if not isinstance(f_predicted, torch.Tensor):
                f_predicted = torch.full_like(r_test, float(f_predicted))

            # Handle potential numerical issues
            f_predicted = torch.nan_to_num(f_predicted, nan=0.0, posinf=1e6, neginf=-1e6)
            f_predicted = torch.clamp(f_predicted, min=-1e6, max=1e6)
        except:
            # If evaluation fails, return a large error value
            return 1e6

    # Calculate MSE between predicted and ground truth forces
    mse = torch.mean((f_ground_truth - f_predicted)**2).item()

    # If MSE is infinite or NaN, return a large error value
    if not np.isfinite(mse):
        return 1e6

    return mse

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

            # For Lennard-Jones potential, we expect the form: 48*(1/r^13) - 24*(1/r^7)
            # Let's try to identify terms with r^(-13) and r^(-7) specifically

            # Approach 1: Use as_ordered_terms to separate individual terms
            terms = expanded_expr.as_ordered_terms() if hasattr(expanded_expr, 'as_ordered_terms') else [expanded_expr]

            coeff_r_neg13 = 0
            coeff_r_neg7 = 0

            for term in terms:
                # Check if this term contains r^(-13)
                if term.has(r**(-13)):
                    # Extract the coefficient of r^(-13)
                    coeff = term.as_coefficient(r**(-13))
                    if coeff is not None:
                        try:
                            coeff_r_neg13 = float(coeff)
                        except:
                            # If conversion to float fails, try evaluating numerically
                            coeff_r_neg13 = float(sp.N(coeff))
                    else:
                        # Alternative method: divide the term by r^(-13) and simplify
                        simplified_coeff = sp.simplify(term / r**(-13))
                        if not simplified_coeff.has(r):  # If it's just a number
                            try:
                                coeff_r_neg13 = float(simplified_coeff)
                            except:
                                coeff_r_neg13 = 0

                # Check if this term contains r^(-7)
                elif term.has(r**(-7)):
                    # Extract the coefficient of r^(-7)
                    coeff = term.as_coefficient(r**(-7))
                    if coeff is not None:
                        try:
                            coeff_r_neg7 = float(coeff)
                        except:
                            # If conversion to float fails, try evaluating numerically
                            coeff_r_neg7 = float(sp.N(coeff))
                    else:
                        # Alternative method: divide the term by r^(-7) and simplify
                        simplified_coeff = sp.simplify(term / r**(-7))
                        if not simplified_coeff.has(r):  # If it's just a number
                            try:
                                coeff_r_neg7 = float(simplified_coeff)
                            except:
                                coeff_r_neg7 = 0

            # If we didn't find the expected terms with as_ordered_terms, try using collect
            if coeff_r_neg13 == 0 and coeff_r_neg7 == 0:
                collected = sp.collect(expanded_expr, [r**(-13), r**(-7)], evaluate=False)
                if collected:
                    for power_term, coeff in collected.items():
                        if power_term == r**(-13):
                            try:
                                coeff_r_neg13 = float(coeff) if not coeff.has(r) else 0
                            except:
                                coeff_r_neg13 = 0
                        elif power_term == r**(-7):
                            try:
                                coeff_r_neg7 = float(coeff) if not coeff.has(r) else 0
                            except:
                                coeff_r_neg7 = 0

            # If still no coefficients found, try a more general approach
            if coeff_r_neg13 == 0 and coeff_r_neg7 == 0:
                # Try to match against the expected form: A*r^(-13) + B*r^(-7)
                # by substituting specific values and solving
                try:
                    # Substitute r=1 to get an equation
                    val_at_1 = float(expanded_expr.subs(r, 1))

                    # Try to solve for coefficients by substituting multiple points
                    r_vals = [0.5, 1.0, 1.5, 2.0]
                    eqns = []
                    for rval in r_vals:
                        lhs = float(expanded_expr.subs(r, rval))
                        rhs_A = float((rval**(-13)))
                        rhs_B = float((rval**(-7)))
                        eqns.append((lhs, rhs_A, rhs_B))

                    # Solve the system of equations: lhs = A*rhs_A + B*rhs_B
                    # Using least squares approach
                    A_matrix = np.array([[eq[1], eq[2]] for eq in eqns])
                    b_vector = np.array([eq[0] for eq in eqns])

                    # Solve using least squares
                    coeffs_fit, residuals, rank, s = np.linalg.lstsq(A_matrix, b_vector, rcond=None)
                    coeff_r_neg13, coeff_r_neg7 = coeffs_fit[0], coeffs_fit[1]
                except:
                    # If all methods fail, return zeros
                    pass

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
            # Simplify the expression to get it in standard form
            simplified_expr = sp.simplify(expr)
            expanded_expr = sp.expand(simplified_expr)

            # The expected form is -k*(r-1) = -k*r + k
            # So we need to extract coefficients of r^1 and the constant term
            coeff_r = expanded_expr.coeff(r, 1)  # Coefficient of r^1
            const_term = expanded_expr.subs(r, 0)  # Constant term when r=0

            # For the form -k*(r-1) = -k*r + k, the coefficient of r is -k and the constant term is k
            # So k = -coeff_r (since the coefficient of r should be -k)
            k_from_r_coeff = -float(coeff_r) if coeff_r is not None else 0
            k_from_const = float(const_term) if const_term is not None else 0

            # Take the average if both are valid, or use the one that makes more sense
            if abs(k_from_r_coeff - k_from_const) < 1:  # Both are similar
                k_extracted = (abs(k_from_r_coeff) + abs(k_from_const)) / 2.0
            else:
                # Use the coefficient of r (which should be -k for -k*(r-1) form)
                k_extracted = abs(k_from_r_coeff)

            # More robust approach: try to factor the expression to match -k*(r-1)
            try:
                factored = sp.factor(expanded_expr)
                # Check if it's in the form -k*(r-1) or k*(1-r)
                if factored.is_Mul:
                    args = factored.args
                    for arg in args:
                        if arg.equals(r - 1):
                            # Expression is coeff*(r-1), so the other factor is the coefficient
                            k_val = -abs(float(factored / (r - 1)))  # Negative because we expect -k*(r-1)
                            k_extracted = abs(k_val)
                            break
                        elif arg.equals(1 - r):
                            # Expression is coeff*(1-r), which is -coeff*(r-1), so the effective k is coeff
                            k_val = abs(float(factored / (1 - r)))
                            k_extracted = abs(k_val)
                            break
            except:
                pass  # If factoring fails, use the coefficient-based approach

            return {
                'coeff_r': float(coeff_r) if coeff_r is not None else 0,
                'const_term': float(const_term) if const_term is not None else 0,
                'k_extracted': k_extracted,
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
    
    # Train DiscoveryNet with fixed hidden size
    model = DiscoveryNet(n_particles=sim.n, hidden_size=64).to(device)  # Fixed hidden size of 64
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=25, factor=0.5)

    print(f"\n--- Training {mode} on {device} ---")
    best_loss = 1e6
    epochs = 100 if mode == 'spring' else 250  # Reduced epochs: 100 for spring, 250 for lj
    patience_counter = 0
    patience_limit = 15  # Reduced patience for early stopping
    min_epochs = 20  # Reduced min_epochs

    # Track previous losses for early stopping based on minimal improvement
    prev_losses = []
    improvement_threshold = 1e-7

    for epoch in range(epochs):
        idxs = np.random.randint(0, p_s.shape[0], size=512)
        f_pred = model(p_s[idxs])

        # Use Log-Cosh loss instead of MSE to handle large force values at small distances
        # Log-Cosh is less sensitive to outliers than MSE
        diff = f_pred - f_s[idxs]
        loss = torch.mean(torch.log(torch.cosh(diff + 1e-12)))  # Add small epsilon to prevent log(0)

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

        if epoch % 50 == 0: print(f"Epoch {epoch} | Loss: {loss.item():.2e}")

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

    if best_loss > threshold:
        print(f"Skipping SR: NN Loss {best_loss:.4f} > {threshold}")
        return f"Fail: {mode}", best_loss, 0, "N/A", 1e6

    # Prepare basis functions for symbolic regression to align with neural network features
    # Extend the range to include more repulsive region samples for LJ mode
    if mode == 'lj':
        # Use a range that emphasizes the repulsive region (small r) and attractive region (larger r)
        r_phys_raw = np.concatenate([
            np.linspace(0.8, 1.0, 150),  # Emphasize repulsive region
            np.linspace(1.0, 2.0, 200), # Transition region
            np.linspace(2.0, 3.5, 150)  # Attractive region
        ]).astype(np.float32)
    else:
        r_phys_raw = np.linspace(0.9, 3.5, 300).astype(np.float32)
    r_phys = r_phys_raw.reshape(-1, 1)

    # Create the same basis functions that the neural network uses internally
    dist_tensor = torch.tensor(r_phys / scaler.p_scale, device=device, dtype=torch.float32)
    with torch.inference_mode():
        # Get the features that the neural network uses internally
        features = model._get_features(dist_tensor)
        f_mag_phys = scaler.inverse_transform_f(model.predict_mag(torch.tensor(r_phys/scaler.p_scale, device=device))).cpu().numpy().ravel()

        # Basis Pruning: In the LJ mode, provide the right basis functions for symbolic regression
        if mode == 'lj':
            # For LJ, we want to provide the fundamental building blocks for the expected form: A*(1/r^13) - B*(1/r^7)
            # The features are [dist, inv_r, inv_r**6, inv_r**12, inv_r**7, inv_r**13]
            # So we want indices 5 (inv_r**13 = r^{-13}) and 4 (inv_r**7 = r^{-7}) as the primary features
            X_basis = features[:, [5, 4]].cpu().numpy()  # [r^{-13}, r^{-7}] in that order

            # Also add the raw distance as a feature to help with normalization
            dist_feature = features[:, [0]].cpu().numpy()
            X_basis = np.column_stack([X_basis, dist_feature])  # [r^{-13}, r^{-7}, r]
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

    # Set optimized parameters for symbolic regression
    population_size = 200  # Reduced population size
    generations = 10  # Reduced number of generations

    # Restrict function set for LJ mode to strongly encourage use of basis functions
    if mode == 'lj':
        f_set = ['add', 'sub', 'mul', 'div']  # Allow division to help form ratios
    else:
        f_set = ['add', 'sub', 'mul']  # Limit operations for spring too to keep it simple

    # Run SymbolicRegressor once with optimized parameters
    est = SymbolicRegressor(population_size=population_size,
                            generations=generations,
                            function_set=f_set,
                            parsimony_coefficient=0.001,  # Fixed parsimony coefficient
                            metric='mse',
                            random_state=42,
                            max_samples=0.9,
                            stopping_criteria=0.0001,
                            const_range=(-50.0, 50.0),  # Wider constant range for better fitting
                            init_depth=(2, 6),  # Moderate initial depth
                            init_method='half and half',
                            verbose=0)

    est.fit(X_basis, f_mag_phys)
    print(f"SR completed: depth={est._program.depth_}, fitness={est._program.raw_fitness_:.4f}")

    # Define the parsimony coefficient that was used
    p_coeff = 0.001

    # Map the symbolic regressor variables (X0, X1, etc.) back to physical variables (r)
    # Using a robust sympy-based substitution system instead of string replacement
    program_str = str(est._program)

    # Define the symbol for distance
    r = sp.Symbol('r')

    # Create a mapping dictionary for variables based on the mode
    if mode == 'lj':
        # In LJ mode, X_basis contains [r^(-13), r^(-7), r] at indices [0, 1, 2]
        # So X0 -> r^(-13), X1 -> r^(-7), X2 -> r
        var_mapping = {
            sp.Symbol('X0'): r**(-13),
            sp.Symbol('X1'): r**(-7),
            sp.Symbol('X2'): r
        }
    else:
        # For spring mode, X0 corresponds to (r-1) where r is the physical distance
        var_mapping = {
            sp.Symbol('X0'): (r - 1)
        }

    # Parse the program string into a SymPy expression
    # First, handle any custom functions like 'inv', 'negpow', etc.
    custom_functions = {
        'inv': lambda x: 1/x,
        'negpow': lambda x, n: x**(-sp.Abs(n)),
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: x / y
    }

    # Parse the expression using sympify with custom function handling
    try:
        expr = sp.sympify(program_str, locals=custom_functions)

        # Perform the variable substitution using SymPy's subs method
        for old_var, new_var in var_mapping.items():
            expr = expr.subs(old_var, new_var)

        # Simplify the final expression
        expr = sp.simplify(expr)
        expr = sp.expand(expr)

    except Exception as e:
        print(f"Error in sympy mapping: {e}")
        # Fallback to original approach if sympy parsing fails
        if mode == 'lj':
            program_str = program_str.replace('X0', 'r**(-13)').replace('X1', 'r**(-7)').replace('X2', 'r')
        else:
            program_str = program_str.replace('X0', '(r-1)')

        ld = {'inv': lambda x: 1/x, 'negpow': lambda x,n: x**(-abs(n)), 'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 'mul': lambda x,y: x*y, 'div': lambda x,y: x/y, 'r': r}
        expr = sp.simplify(sp.sympify(program_str, locals=ld))
        expr = sp.expand(expr)
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

    # Check success criteria - Updated to match requirements: MSE < 0.01 and Coeff_Accuracy < 0.05
    validation_success = mse < 0.01 and coeff_accuracy < 0.05 and est._program.depth_ < 8
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