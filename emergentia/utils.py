import torch
import numpy as np
import sympy as sp

def verify_equivalence(expr, mode, domain=(0.8, 3.0), samples=100, threshold=1e-3):
    """
    Numerically verify if a discovered expression matches the ground truth.
    """
    r = sp.Symbol('r')
    
    if mode == 'spring':
        target_expr = -10.0 * (r - 1.0)
    elif mode == 'lj':
        target_expr = 48.0 * (r**-13) - 24.0 * (r**-7)
    else:
        return False, 1e6

    # Convert SymPy expressions to numeric functions
    try:
        f_discovered = sp.lambdify(r, expr, 'numpy')
        f_target = sp.lambdify(r, target_expr, 'numpy')
        
        r_vals = np.linspace(domain[0], domain[1], samples)
        
        y_discovered = f_discovered(r_vals)
        y_target = f_target(r_vals)
        
        # Handle cases where lambdify returns a constant
        if np.isscalar(y_discovered):
            y_discovered = np.full_like(r_vals, y_discovered)
            
        mse = np.mean((y_discovered - y_target)**2)
        return mse < threshold, mse
        
    except Exception as e:
        return False, 1e6

def extract_coefficients(expr, mode):
    """
    Heuristic extraction of physical constants.
    """
    r = sp.Symbol('r')
    if mode == 'spring':
        # Expect -k * (r - 1) = -k*r + k
        try:
            k = -float(expr.diff(r).subs(r, 1.0))
            return {"k": k}
        except:
            return {"k": 0.0}
    elif mode == 'lj':
        # Expect A/r^13 - B/r^7
        try:
            # We can use least squares to find A and B if we assume the form
            r_vals = np.linspace(1.0, 2.0, 100)
            f_num = sp.lambdify(r, expr, 'numpy')
            y = f_num(r_vals)
            if np.isscalar(y): y = np.full_like(r_vals, y)
            
            X = np.stack([r_vals**-13, -r_vals**-7], axis=1)
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return {"A": coeffs[0], "B": coeffs[1]}
        except:
            return {"A": 0.0, "B": 0.0}
    return {}
