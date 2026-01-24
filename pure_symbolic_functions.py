"""
Pure functions extracted from symbolic processing modules to improve testability.
Consolidated to use symbolic_utils.
"""

import numpy as np
import sympy as sp
from symbolic_utils import create_safe_functions, gp_to_sympy, extract_feature_indices

# Re-export for backward compatibility
safe_sqrt_func, safe_log_func, safe_div_func, safe_inv_func, safe_square_func, safe_inv_square_func = create_safe_functions()

def create_safe_functions_legacy():
    return safe_sqrt_func, safe_log_func, safe_div_func, safe_inv_func

def extract_feature_indices_from_expression(expr_str, n_features):
    indices = extract_feature_indices(expr_str)
    return [idx for idx in indices if idx < n_features]

def parse_sympy_expression(expr_str, n_features):
    return gp_to_sympy(expr_str, n_features=n_features)

def extract_numeric_constants(expr):
    """
    Extract numeric constants from a SymPy expression.
    """
    all_atoms = expr.atoms(sp.Number)
    # Filter constants that are likely parameters and not indices/small integers
    constants = sorted([float(a) for a in all_atoms if not a.is_Integer or abs(a) > 5])
    return constants

def create_parametrized_expression(expr, constants):
    """
    Create a parametrized version of an expression where constants are replaced with symbols.
    """
    const_vars = [sp.Symbol(f'c{i}') for i in range(len(constants))]
    subs_map = {sp.Float(c): cv for c, cv in zip(constants, const_vars)}
    return expr.subs(subs_map)

def evaluate_expression_at_points(expr, points, feature_names=None):
    from symbolic_utils import evaluate_sympy
    return evaluate_sympy(expr, points, feature_names=feature_names)

def simplify_with_common_patterns(expr):
    return sp.simplify(expr)

def expression_to_string(expr):
    return str(expr)

def count_expression_nodes(expr):
    if expr.is_Atom:
        return 1
    else:
        return 1 + sum(count_expression_nodes(arg) for arg in expr.args)

def get_expression_complexity(expr):
    return count_expression_nodes(expr)

def validate_expression_stability(expr, X_sample):
    from symbolic_utils import evaluate_sympy
    try:
        result = evaluate_sympy(expr, X_sample)
        return np.all(np.isfinite(result))
    except:
        return False
