import numpy as np
import torch
import sympy as sp
import re
from gplearn.functions import make_function

def _safe_sqrt(x):
    """Safe square root function that handles negative inputs and very small values."""
    return np.sqrt(np.abs(x) + 1e-12)

def _safe_log(x):
    """Safe logarithm function that handles zero and negative inputs with better epsilon."""
    return np.log(np.abs(x) + 1e-12)

def _safe_div(x, y):
    """Safe division function that handles division by zero and extreme values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(y) < 1e-12, 0.0, x / np.where(y >= 0, y + 1e-12, y - 1e-12))
    return np.clip(result, -1e12, 1e12)

def _safe_inv(x):
    """Safe inverse function."""
    return _safe_div(1.0, x)

def _safe_square(x):
    """Safe square function that prevents overflow."""
    x_clipped = np.clip(x, -1e6, 1e6)
    return np.clip(x_clipped**2, -1e12, 1e12)

def _safe_inv_square(x):
    """Safe inverse square function that prevents overflow/underflow."""
    x_squared = np.clip(x**2, 1e-12, 1e12)
    result = 1.0 / (x_squared + 1e-9)
    return np.clip(result, -1e12, 1e12)

def create_safe_functions():
    """Create safe mathematical functions for symbolic regression."""
    safe_sqrt_func = make_function(function=_safe_sqrt, name='sqrt', arity=1)
    safe_log_func = make_function(function=_safe_log, name='log', arity=1)
    safe_div_func = make_function(function=_safe_div, name='div', arity=2)
    safe_inv_func = make_function(function=_safe_inv, name='inv', arity=1)
    safe_square_func = make_function(function=_safe_square, name='square', arity=1)
    safe_inv_square_func = make_function(function=_safe_inv_square, name='inv_square', arity=1)
    
    return safe_sqrt_func, safe_log_func, safe_div_func, safe_inv_func, safe_square_func, safe_inv_square_func

def gp_to_sympy(expr_str, n_features=None):
    """
    Unified GP expression to SymPy conversion.
    """
    local_dict = {
        'add': lambda x,y: x+y,
        'sub': lambda x,y: x-y,
        'mul': lambda x,y: x*y,
        'div': lambda x,y: x/(y+1e-9),
        'sqrt': lambda x: sp.sqrt(sp.Abs(x)),
        'log': lambda x: sp.log(sp.Abs(x)+1e-9),
        'abs': sp.Abs,
        'neg': lambda x: -x,
        'inv': lambda x: 1.0/(x+1e-9),
        'square': lambda x: sp.Pow(x, 2),
        'inv_square': lambda x: 1.0/(sp.Pow(x, 2)+1e-9)
    }
    
    # If n_features is provided, explicitly map X0, X1, ... to symbols
    if n_features is not None:
        for i in range(n_features):
            local_dict[f'X{i}'] = sp.Symbol(f'x{i}')
    else:
        # Dynamic mapping for any Xi found in the string
        feature_matches = re.findall(r'X(\d+)', expr_str)
        for idx in feature_matches:
            local_dict[f'X{idx}'] = sp.Symbol(f'x{idx}')

    return sp.sympify(expr_str, locals=local_dict)

def extract_feature_indices(expr_str):
    """Extract unique feature indices from an expression string."""
    feature_matches = re.findall(r'X(\d+)', expr_str)
    return sorted(list(set(int(idx) for idx in feature_matches)))

def evaluate_sympy(expr, X, feature_names=None):
    """Evaluate a SymPy expression on a numpy array X."""
    if feature_names is None:
        symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))
    else:
        symbols = [sp.Symbol(name) for name in feature_names]
        
    func = sp.lambdify(symbols, expr, modules=['numpy'])
    
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Map columns of X to symbols
    # This assumes symbols are x0, x1, ... corresponding to columns 0, 1, ...
    # Or matches feature_names
    args = []
    for sym in symbols:
        match = re.search(r'(\d+)', str(sym))
        idx = int(match.group(1)) if match else 0
        args.append(X[:, idx] if idx < X.shape[1] else np.zeros(X.shape[0]))
        
    return func(*args)
