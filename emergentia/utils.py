import torch
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

def verify_equivalence(expr, mode, potential=None, domain=None, samples=100):
    """
    Numerically verify if a discovered expression matches the ground truth using curve fitting and statistics.
    """
    r = sp.Symbol('r')
    
    if domain is None:
        if mode == 'lj':
            domain = (0.6, 3.5)
        elif mode == 'morse':
            domain = (0.5, 4.0)
        else:
            domain = (0.5, 2.5)

    try:
        f_discovered = sp.lambdify(r, expr, 'numpy')
        
        # Generate ground truth data
        r_vals = np.linspace(domain[0], domain[1], samples)
        
        if potential is not None:
            # Use the actual potential object if provided
            with torch.no_grad():
                r_tensor = torch.tensor(r_vals, dtype=torch.float32).view(-1, 1)
                y_target = potential.compute_force_magnitude(r_tensor).numpy().ravel()
        else:
            # Fallback to hardcoded targets if potential is not provided
            if mode == 'spring':
                y_target = -10.0 * (r_vals - 1.0)
            elif mode == 'lj':
                y_target = 48.0 * (r_vals**-13) - 24.0 * (r_vals**-7)
            else:
                return False, {"mse": 1e6, "r2": 0.0, "bic": 1e6}

        y_discovered = f_discovered(r_vals)
        if np.isscalar(y_discovered):
            y_discovered = np.full_like(r_vals, y_discovered)
            
        # Statistical Metrics
        mse = np.mean((y_discovered - y_target)**2)
        var_y = np.var(y_target)
        r2 = 1 - (mse / var_y) if var_y > 1e-9 else 0.0
        
        # BIC: n * ln(MSE) + k * ln(n)
        # k is the number of symbols in the expression as a proxy for complexity
        k = len(expr.free_symbols) + len(list(expr.atoms(sp.Number)))
        n = samples
        bic = n * np.log(mse + 1e-12) + k * np.log(n)
        
        success = (r2 > 0.995) and (mse < 1e-2)
        
        return success, {"mse": mse, "r2": r2, "bic": bic}
        
    except Exception as e:
        print(f"Verification error: {e}")
        return False, {"mse": 1e6, "r2": 0.0, "bic": 1e6}

def extract_coefficients(expr, mode):
    """
    Heuristic extraction of physical constants.
    """
    # This remains largely the same but could be improved.
    # For now, I'll keep it simple or remove if not strictly needed.
    r = sp.Symbol('r')
    if mode == 'spring':
        try:
            k = -float(expr.diff(r).subs(r, 1.0))
            return {"k": k}
        except:
            return {"k": 0.0}
    return {}