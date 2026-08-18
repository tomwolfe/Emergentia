import torch
import numpy as np
import sympy as sp
from scipy.optimize import curve_fit

def check_functional_form(expr, mode: str) -> bool:
    """Check if the expression contains the expected key terms for the given mode."""
    if expr is None:
        return False
    expr_str = str(expr).lower()
    if mode == "gravity":
        return any(term in expr_str for term in ["1/r**2", "r**(-2)", "r**-2", "1/(r**2)", "1/r^2"])
    elif mode == "spring":
        return "r" in expr_str and "exp" not in expr_str
    elif mode == "lj":
        # Accept both the force form (r^-13, r^-7) and the potential form
        # (r^-12, r^-6) since the discovered expression may be written either way.
        return any(term in expr_str for term in [
            "r**-13", "r**(-13)", "1/r**13", "r**-7", "r**(-7)", "1/r**7",
            "r**-12", "r**(-12)", "1/r**12", "r**-6", "r**(-6)", "1/r**6",
            "r**6", "r**12",  # (a - b*r^6)/r^13 form
        ])
    elif mode in ["morse", "yukawa"]:
        return "exp" in expr_str
    elif mode == "buckingham":
        return "exp" in expr_str or any(term in expr_str for term in ["r**-7", "r**-6", "1/r**7", "1/r**6"])
    elif mode == "electric":
        return any(term in expr_str for term in ["1/r**2", "r**(-2)", "r**-2", "1/(r**2)", "1/r^2"])
    return True


def verify_equivalence(expr, mode, potential=None, domain=None, samples=100,
                       test_r_vals=None, test_y_target=None):
    """
    Numerically verify if a discovered expression matches the ground truth using curve fitting and statistics.
    """
    r = sp.Symbol('r')

    if domain is None:
        if potential is not None and hasattr(potential, 'default_scale'):
            # Verify over the range the simulation actually samples, not the
            # full physical domain. The NN (and thus the discovered law) is
            # only reliable where the trajectory had data.
            domain = (0.8, potential.default_scale)
        elif mode == 'lj':
            domain = (0.6, 3.5)
        elif mode == 'morse':
            domain = (0.5, 4.0)
        elif mode == 'gravity':
            domain = (0.5, 5.0)
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
            elif mode == 'gravity':
                y_target = -1.0 / (r_vals**2)
            else:
                return False, {"mse": 1e6, "r2": 0.0, "bic": 1e6, "functional_form_match": False}

        y_discovered = f_discovered(r_vals)
        if np.isscalar(y_discovered):
            y_discovered = np.full_like(r_vals, y_discovered)
            
        # Statistical Metrics
        mse = np.mean((y_discovered - y_target)**2)
        var_y = np.var(y_target)
        r2 = 1 - (mse / var_y) if var_y > 1e-9 else 0.0
        
        # BIC: n * ln(MSE) + k * ln(n)
        k = len(expr.free_symbols) + len(list(expr.atoms(sp.Number)))
        n = samples
        bic = n * np.log(mse + 1e-12) + k * np.log(n)
        
        result = {"mse": mse, "r2": r2, "bic": bic}

        # Compute test metrics if test data provided
        if test_r_vals is not None and test_y_target is not None:
            y_test_discovered = f_discovered(test_r_vals)
            if np.isscalar(y_test_discovered):
                y_test_discovered = np.full_like(test_r_vals, y_test_discovered)
            test_mse = np.mean((y_test_discovered - test_y_target)**2)
            test_var = np.var(test_y_target)
            test_r2 = 1 - (test_mse / test_var) if test_var > 1e-9 else 0.0
            result["test_mse"] = test_mse
            result["test_r2"] = test_r2

        # Functional form check
        functional_form_match = check_functional_form(expr, mode)
        result["functional_form_match"] = functional_form_match

        success = (r2 > 0.95) and (mse < 0.05)
        
        return success, result
        
    except Exception as e:
        print(f"Verification error: {e}")
        return False, {"mse": 1e6, "r2": 0.0, "bic": 1e6, "functional_form_match": False}

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