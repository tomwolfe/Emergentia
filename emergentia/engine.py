import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from .models import DiscoveryNet, TrajectoryScaler
from .utils import verify_equivalence

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
    def __init__(self, mode='lj', potential=None, device='cpu', seed=42):
        self.mode = mode
        self.potential = potential # Store the actual potential object
        self.device = device
        self.seed = seed
        self.model = DiscoveryNet().to(device)
        self.scaler = TrajectoryScaler(mode=mode)
        
    def train_nn(self, p_traj, f_traj, epochs=5000, noise_std=0.0):
        self.scaler.fit(p_traj, f_traj)
        p_s, f_s = self.scaler.transform(p_traj, f_traj)
        
        # Symmetric Log Transform for high dynamic range
        f_target = torch.sign(f_s) * torch.log1p(torch.abs(f_s))
            
        p_s = p_s.to(self.device)
        f_target = f_target.to(self.device)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-3, weight_decay=1e-4)
        # Increase delta for noise resilience if needed, or keep it robust
        delta = 0.5 if noise_std > 0 else 0.1
        criterion = nn.HuberLoss(delta=delta)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=200, factor=0.5)
        
        print(f"Training NN for {self.mode} (noise_std={noise_std})...")
        for epoch in range(epochs):
            idxs = torch.randint(0, p_s.shape[0], (1024,))
            p_batch = p_s[idxs]
            f_batch = f_target[idxs]
            
            f_pred = self.model(p_batch)
            loss = criterion(f_pred, f_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.2e}")
                
        return loss.item()

    def distill_symbolic(self, population_size=2000, generations=40):
        # Sample the NN across a physical range
        if self.mode == 'lj':
            r_min, r_max = 0.6, 3.5
        elif self.mode == 'morse':
            r_min, r_max = 0.5, 4.0
        else: # spring
            r_min, r_max = 0.5, 2.5
            
        r_phys = np.linspace(r_min, r_max, 500).reshape(-1, 1).astype(np.float32)
        r_scaled = torch.tensor(r_phys / self.scaler.p_scale, device=self.device)
        
        with torch.no_grad():
            mag_scaled = self.model.predict_mag(r_scaled)
            # Inverse Symmetric Log Transform
            mag_s = torch.sign(mag_scaled) * (torch.exp(torch.abs(mag_scaled)) - 1)
            mag_phys = (mag_s * self.scaler.f_scale).cpu().numpy().ravel()

        # Input features for SR: [r, 1/r, exp(-r)]
        X_sr = np.hstack([r_phys, 1.0/r_phys, np.exp(-r_phys)])
        
        print(f"Running Symbolic Regression for {self.mode}...")
        est = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            function_set=('add', 'sub', 'mul', 'div', inv, power, exp),
            const_range=(-100.0, 100.0),
            parsimony_coefficient=0.01, 
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
        expr = expr.subs(sp.Symbol('X0'), r).subs(sp.Symbol('X1'), 1/r).subs(sp.Symbol('X2'), sp.exp(-r))
        expr = sp.simplify(expr)
        
        return expr

    def run(self, sim, nn_epochs=5000, noise_std=0.0):
        p_traj, f_traj = sim.generate(steps=2000, noise_std=noise_std)
        final_nn_loss = self.train_nn(p_traj, f_traj, epochs=nn_epochs, noise_std=noise_std)
        discovered_expr = self.distill_symbolic()
        
        # Use the potential object for verification if available
        success, metrics = verify_equivalence(discovered_expr, self.mode, potential=self.potential)
        
        return {
            "mode": self.mode,
            "nn_loss": final_nn_loss,
            "formula": str(discovered_expr),
            "mse": metrics.get("mse", 1e6),
            "r2": metrics.get("r2", 0.0),
            "bic": metrics.get("bic", 1e6),
            "success": success
        }
