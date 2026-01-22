"""
Pure functions extracted from symbolic processing modules to improve testability.
This addresses the need to extract pure functions from symbolic processing modules.
"""

import numpy as np
import sympy as sp
from gplearn.functions import make_function


def _safe_sqrt(x):
    """Safe square root function that handles negative inputs and very small values."""
    return np.sqrt(np.abs(x) + 1e-12)


def _safe_log(x):
    """Safe logarithm function that handles zero and negative inputs with better epsilon."""
    return np.log(np.abs(x) + 1e-12)


def _safe_div(x, y):
    """Safe division function that handles division by zero and extreme values."""
    return np.where(np.abs(y) < 1e-12, 0.0, x / (y + np.sign(y) * 1e-12))


def create_safe_functions():
    """Create safe mathematical functions for symbolic regression."""
    safe_sqrt_func = make_function(function=_safe_sqrt, name='sqrt', arity=1)
    safe_log_func = make_function(function=_safe_log, name='log', arity=1)
    safe_div_func = make_function(function=_safe_div, name='div', arity=2)
    safe_inv_func = make_function(function=lambda x: _safe_div(1.0, x), name='inv', arity=1)
    
    return safe_sqrt_func, safe_log_func, safe_div_func, safe_inv_func


def extract_feature_indices_from_expression(expr_str, n_features):
    """
    Extract feature indices from a symbolic expression string.
    
    Args:
        expr_str: String representation of the expression
        n_features: Total number of features
    
    Returns:
        List of feature indices used in the expression
    """
    import re
    # Look for X followed by digits (e.g., X0, X1, X12, etc.)
    feature_matches = re.findall(r'X(\d+)', expr_str)
    indices = [int(idx) for idx in feature_matches if int(idx) < n_features]
    return sorted(list(set(indices)))  # Return unique indices in sorted order


def parse_sympy_expression(expr_str, n_features):
    """
    Parse a symbolic expression string into a SymPy expression.
    
    Args:
        expr_str: String representation of the expression
        n_features: Total number of features
    
    Returns:
        Parsed SymPy expression
    """
    # Create local dictionary for gplearn-style functions
    local_dict = {
        'add': lambda x, y: x + y,
        'sub': lambda x, y: x - y,
        'mul': lambda x, y: x * y,
        'div': lambda x, y: x / (y + 1e-9),
        'sqrt': lambda x: sp.sqrt(sp.Abs(x)),
        'log': lambda x: sp.log(sp.Abs(x) + 1e-9),
        'abs': sp.Abs,
        'neg': lambda x: -x,
        'inv': lambda x: 1.0 / (x + 1e-9),
        'sin': sp.sin,
        'cos': sp.cos,
        'exp': sp.exp,
    }
    
    # Add feature variables (X0, X1, ..., Xn)
    for i in range(n_features):
        local_dict[f'X{i}'] = sp.Symbol(f'x{i}')
    
    return sp.sympify(expr_str, locals=local_dict)


def extract_numeric_constants(expr):
    """
    Extract numeric constants from a SymPy expression.
    
    Args:
        expr: SymPy expression
    
    Returns:
        List of numeric constants in the expression
    """
    all_atoms = expr.atoms(sp.Number)
    # Filter constants that are likely parameters and not indices/small integers
    constants = sorted([float(a) for a in all_atoms if not a.is_Integer or abs(a) > 5])
    return constants


def create_parametrized_expression(expr, constants):
    """
    Create a parametrized version of an expression where constants are replaced with symbols.
    
    Args:
        expr: Original SymPy expression
        constants: List of constants to parametrize
    
    Returns:
        Parametrized SymPy expression
    """
    const_vars = [sp.Symbol(f'c{i}') for i in range(len(constants))]
    subs_map = {sp.Float(c): cv for c, cv in zip(constants, const_vars)}
    return expr.subs(subs_map)


def evaluate_expression_at_points(expr, points, feature_names=None):
    """
    Evaluate a SymPy expression at given points.
    
    Args:
        expr: SymPy expression to evaluate
        points: Array of points where to evaluate the expression
        feature_names: Names of the features (variables) in the expression
    
    Returns:
        Array of evaluated values
    """
    if feature_names is None:
        # Auto-detect feature names from the expression
        symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))
        feature_names = [str(s) for s in symbols]
    
    # Create lambda function for evaluation
    f_lamb = sp.lambdify(feature_names, expr, modules=['numpy'])
    
    # Evaluate at points
    return f_lamb(*[points[:, i] for i in range(points.shape[1])])


def simplify_with_common_patterns(expr):
    """
    Apply common simplification patterns to a SymPy expression.
    
    Args:
        expr: SymPy expression to simplify
    
    Returns:
        Simplified SymPy expression
    """
    # Apply standard SymPy simplification
    simplified = sp.simplify(expr)
    
    # Additional custom simplifications could be added here
    # For example, recognizing common physics patterns
    
    return simplified


def expression_to_string(expr):
    """
    Convert a SymPy expression to a readable string.
    
    Args:
        expr: SymPy expression
    
    Returns:
        String representation of the expression
    """
    return str(expr)


def count_expression_nodes(expr):
    """
    Count the number of nodes in a SymPy expression tree.
    
    Args:
        expr: SymPy expression
    
    Returns:
        Number of nodes in the expression tree
    """
    if expr.is_Atom:
        return 1
    else:
        return 1 + sum(count_expression_nodes(arg) for arg in expr.args)


def get_expression_complexity(expr):
    """
    Calculate a complexity score for a SymPy expression.
    
    Args:
        expr: SymPy expression
    
    Returns:
        Complexity score (currently just the number of nodes)
    """
    return count_expression_nodes(expr)


def validate_expression_stability(expr, X_sample):
    """
    Validate that an expression is numerically stable for a sample input.
    
    Args:
        expr: SymPy expression to validate
        X_sample: Sample input data
    
    Returns:
        Boolean indicating if the expression is stable
    """
    try:
        # Convert to lambda function
        symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))
        symbol_names = [str(s) for s in symbols]
        
        f_lamb = sp.lambdify(symbol_names, expr, modules=['numpy'])
        
        # Evaluate on sample
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)
        
        result = f_lamb(*[X_sample[:, i] for i in range(X_sample.shape[1])])
        
        # Check for NaNs or Infs
        return np.all(np.isfinite(result))
    except:
        return False