import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from .models import DiscoveryNet, TrajectoryScaler, BASIS_REGISTRY
from .utils import verify_equivalence

# Basis registries for symbolic distillation
NP_BASIS_REGISTRY = {
    '1': lambda d: np.ones_like(d),
    'r': lambda d: d,
    '1/r': lambda d: 1.0 / d,
    '1/r^2': lambda d: 1.0 / np.power(d, 2),
    '1/r^6': lambda d: 1.0 / np.power(d, 6),
    '1/r^7': lambda d: 1.0 / np.power(d, 7),
    '1/r^12': lambda d: 1.0 / np.power(d, 12),
    '1/r^13': lambda d: 1.0 / np.power(d, 13),
    'exp(-r)': lambda d: np.exp(-d)
}

SP_BASIS_REGISTRY = {
    '1': lambda r: sp.Integer(1),
    'r': lambda r: r,
    '1/r': lambda r: 1/r,
    '1/r^2': lambda r: 1/r**2,
    '1/r^6': lambda r: 1/r**6,
    '1/r^7': lambda r: 1/r**7,
    '1/r^12': lambda r: 1/r**12,
    '1/r^13': lambda r: 1/r**13,
    'exp(-r)': lambda r: sp.exp(-r)
}

# Protected functions for gplearn
def _protected_inv(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 0.01, 1.0/x, 0.0)
inv = make_function(function=_protected_inv, name='inv', arity=1)

def _protected_power(x, y):
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        y_clamped = np.clip(y, -14, 14)
        abs_x = np.where(np.abs(x) < 1e-6, 1e-6, np.abs(x))
        result = np.power(abs_x, y_clamped)
        return np.where(np.isfinite(result), np.clip(result, -1e10, 1e10), 0.0)
power = make_function(function=_protected_power, name='power', arity=2)

def _protected_exp(x):
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.exp(np.clip(x, -20, 20))
        return result
exp = make_function(function=_protected_exp, name='exp', arity=1)

class DiscoveryPipeline:
    def __init__(self, mode='lj', potential=None, device='cpu', seed=42, basis_set=None):
        self.mode = mode
        self.potential = potential # Store the actual potential object
        self.device = device
        self.seed = seed
        self.model = DiscoveryNet(basis_set=basis_set).to(device)
        self.scaler = TrajectoryScaler(mode=mode)
        
    def train_nn(self, p_traj, f_traj, epochs=5000, noise_std=0.0):
        if torch.isnan(p_traj).any() or torch.isnan(f_traj).any():
            print("Warning: NaNs detected in trajectories. Clipping and filling.")
            p_traj = torch.nan_to_num(p_traj, nan=0.0)
            f_traj = torch.nan_to_num(f_traj, nan=0.0)

        if noise_std > 0:
            from scipy.ndimage import gaussian_filter1d
            p_np = p_traj.cpu().numpy()
            p_np = gaussian_filter1d(p_np, sigma=1.0, axis=0)
            p_traj = torch.from_numpy(p_np).to(self.device)

        self.scaler.fit(p_traj, f_traj)
        p_s, f_s = self.scaler.transform(p_traj, f_traj)
        
        # Symmetric Log Transform for high dynamic range
        f_target = torch.sign(f_s) * torch.log1p(torch.abs(f_s))
        
        if torch.isnan(f_target).any():
            f_target = torch.nan_to_num(f_target, nan=0.0)
            
        p_s = p_s.to(self.device)
        f_target = f_target.to(self.device)
        
        base_lr = 2e-3
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=base_lr, weight_decay=1e-4)
        delta = 0.5 if noise_std > 0 else 0.1
        criterion = nn.HuberLoss(delta=delta)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.5)
        
        warmup_epochs = 500
        
        print(f"Training NN for {self.mode} (noise_std={noise_std})...")
        for epoch in range(epochs):
            # LR Warm-up
            if epoch < warmup_epochs:
                lr = base_lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            idxs = torch.randint(0, p_s.shape[0], (1024,))
            p_batch = p_s[idxs]
            f_batch = f_target[idxs]
            
            f_pred = self.model(p_batch)
            loss = criterion(f_pred, f_batch)
            
            if torch.isnan(loss):
                print(f"NaN Loss at epoch {epoch}. Stopping.")
                break
                
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            if epoch >= warmup_epochs:
                scheduler.step(loss)
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.2e} | LR: {optimizer.param_groups[0]['lr']:.2e}")
                
        return loss.item()

    def distill_symbolic(self, population_size=2000, generations=40):
        # Sample the NN across a physical range
        if self.mode == 'lj':
            r_min, r_max = 0.6, 3.5
        elif self.mode == 'morse':
            r_min, r_max = 0.5, 4.0
        else: # spring/gravity
            r_min, r_max = 0.5, 5.0
            
        r_phys = np.linspace(r_min, r_max, 500).reshape(-1, 1).astype(np.float32)
        r_scaled = torch.tensor(r_phys / self.scaler.p_scale, device=self.device)
        
        with torch.no_grad():
            mag_scaled = self.model.predict_mag(r_scaled)
            # Inverse Symmetric Log Transform with clamping to prevent overflow
            # exp(20) is ~4.8e8, which is safe for float32
            mag_s = torch.sign(mag_scaled) * (torch.exp(torch.clamp(torch.abs(mag_scaled), max=20.0)) - 1)
            mag_phys = (mag_s * self.scaler.f_scale).cpu().numpy().ravel()
            
        # Clean up any remaining infinities or NaNs in mag_phys
        mag_phys = np.nan_to_num(mag_phys, nan=0.0, posinf=1e10, neginf=-1e10)

        # Dynamic Input features for SR based on DiscoveryNet basis
        X_feats = []
        for name in self.model.basis_names:
            if name in NP_BASIS_REGISTRY:
                X_feats.append(NP_BASIS_REGISTRY[name](r_phys))
            else:
                raise ValueError(f"Unknown basis function: {name}")
            
        X_sr = np.hstack(X_feats)
        
        # Parsimony adjustment: much higher to favor simple basis combinations
        parsimony = 0.1
        
        print(f"Running Symbolic Regression for {self.mode}...")
        est = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            function_set=('add', 'sub', 'mul', 'div', inv),
            const_range=(-100.0, 100.0),
            parsimony_coefficient=parsimony, 
            stopping_criteria=0.001,
            init_depth=(2, 6),
            max_samples=0.9,
            n_jobs=-1,
            metric='mse',
            random_state=self.seed,
            verbose=1
        )
        
        est.fit(X_sr, mag_phys)
        print(f"Best program: {est._program}")
        
        # Convert to SymPy
        r = sp.Symbol('r')
        locals_dict = {
            'add': lambda x,y: x+y, 'sub': lambda x,y: x-y, 
            'mul': lambda x,y: x*y, 'div': lambda x,y: x/y,
            'inv': lambda x: 1/x, 'power': lambda x,y: sp.Pow(sp.Abs(x), y),
            'exp': lambda x: sp.exp(x)
        }
        
        expr = sp.sympify(str(est._program), locals=locals_dict)
        
        # Mapping back X0, X1, ... to basis functions
        for i, name in enumerate(self.model.basis_names):
            if name in SP_BASIS_REGISTRY:
                val = SP_BASIS_REGISTRY[name](r)
            else:
                raise ValueError(f"Unknown basis function: {name}")
            expr = expr.subs(sp.Symbol(f'X{i}'), val)
            
        expr = sp.simplify(expr)
        return expr

    def refine_constants(self, expr, p_traj, f_traj):
        from scipy.optimize import minimize
        
        # Identify numerical constants (Floats and Integers) and replace with symbols for optimization
        # Exclude small integers that are likely exponents (e.g., 1, 2, -1, -2) to keep the functional form
        all_atoms = list(expr.atoms(sp.Number))
        tune_atoms = [a for a in all_atoms if not (isinstance(a, sp.Integer) and abs(a) <= 2)]
        
        if not tune_atoms:
            return expr
            
        symbols = [sp.Symbol(f'c{i}') for i in range(len(tune_atoms))]
        param_map = {tune_atoms[i]: symbols[i] for i in range(len(tune_atoms))}
        param_expr = expr.subs(param_map)
        
        r_sym = sp.Symbol('r')
        func = sp.lambdify([r_sym] + symbols, param_expr, 'numpy')
        
        p_np = p_traj.cpu().numpy()
        f_np = f_traj.cpu().numpy()
        
        def objective(params):
            # Compute total force
            diff = p_np[:, :, np.newaxis, :] - p_np[:, np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=-1, keepdims=True)
            
            with np.errstate(all='ignore'):
                mag = func(dist, *params)
                if not isinstance(mag, np.ndarray):
                    mag = np.full(dist.shape, mag)
                # Handle possible infinities from bad params
                mag = np.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)
                    
            n = p_np.shape[1]
            mask = (~np.eye(n, dtype=bool))[np.newaxis, :, :, np.newaxis]
            
            pair_forces = mag * (diff / np.clip(dist, 1e-6, None)) * mask
            f_pred = np.sum(pair_forces, axis=2)
            
            return np.mean((f_pred - f_np)**2)
            
        initial_guess = [float(v) for v in tune_atoms]
        # Use a more robust optimizer
        res = minimize(objective, initial_guess, method='Nelder-Mead', tol=1e-3)
        
        final_map = {symbols[i]: res.x[i] for i in range(len(symbols))}
        return sp.simplify(param_expr.subs(final_map))

    def run(self, sim, nn_epochs=5000, noise_std=0.0):
        p_traj, f_traj = sim.generate(steps=2000, noise_std=noise_std)
        final_nn_loss = self.train_nn(p_traj, f_traj, epochs=nn_epochs, noise_std=noise_std)
        discovered_expr = self.distill_symbolic()
        
        print(f"Raw discovered formula: {discovered_expr}")
        refined_expr = self.refine_constants(discovered_expr, p_traj, f_traj)
        print(f"Refined formula: {refined_expr}")
        
        # Use the potential object for verification if available
        success, metrics = verify_equivalence(refined_expr, self.mode, potential=self.potential)
        
        return {
            "mode": self.mode,
            "nn_loss": final_nn_loss,
            "formula": str(refined_expr),
            "raw_formula": str(discovered_expr),
            "mse": metrics.get("mse", 1e6),
            "r2": metrics.get("r2", 0.0),
            "bic": metrics.get("bic", 1e6),
            "success": success
        }
