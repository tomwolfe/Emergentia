import sympy as sp
from typing import Dict, List, Tuple
import numpy as np


class DimensionalInconsistencyError(Exception):
    """Raised when an expression has dimensionally inconsistent terms in an Add node."""
    pass


class DimensionalSignature:
    """
    Simple dimensional signature for Length (L), Time (T), Mass (M), Charge (Q).
    This replaces a full unit system (like Pint) for lightweight physical validation.
    """

    def __init__(self):
        self.dimensions = {
            "L": 0,  # Length
            "T": 0,  # Time
            "M": 0,  # Mass
            "Q": 0,  # Charge
        }

    def add_power(self, dim: str, exponent: float):
        if dim in self.dimensions:
            self.dimensions[dim] += int(exponent)

    def get_signature(self) -> Tuple[float, float, float, float]:
        return (
            self.dimensions["L"],
            self.dimensions["T"],
            self.dimensions["M"],
            self.dimensions["Q"],
        )

    def is_valid(self) -> bool:
        for dim in self.dimensions.values():
            if not isinstance(dim, (int, float)) or dim != round(dim, 6):
                return False
        return True


class UnitChecker:
    """
    Validates symbolic expressions for physical consistency using dimensional analysis.
    Assigns dimensions to terminal variables (r, 1/r, etc.) and evaluates expression consistency.
    """

    DimensionalSignature = DimensionalSignature

    VARIABLE_DIMENSIONS = {
        "r": (1, 0, 0, 0),  # Position variable - length dimension
        "r2": (2, 0, 0, 0),
        "1/r": (-1, 0, 0, 0),  # Inverse of position - inverse length dimension
        "1/r^2": (-2, 0, 0, 0),
        "1/r^3": (-3, 0, 0, 0),
        "r_inv": (-1, 0, 0, 0),
        "inv": (-1, 0, 0, 0),
        "x": (1, 0, 0, 0),
        "y": (1, 0, 0, 0),
        "dx": (1, 0, 0, 0),
        "dy": (1, 0, 0, 0),
        "x_inv": (-1, 0, 0, 0),
        "y_inv": (-1, 0, 0, 0),
        "time": (0, -1, 0, 0),
        "dt": (0, -1, 0, 0),
        "t": (0, -1, 0, 0),
        "mass": (0, 0, 1, 0),
        "m": (0, 0, 1, 0),
        "mass_inv": (0, 0, -1, 0),
        "inv_m": (0, 0, -1, 0),
        "charge": (0, 0, 0, 1),
        "q": (0, 0, 0, 1),
        "charge_inv": (0, 0, 0, -1),
        "inv_q": (0, 0, 0, -1),
        "k": (1, 3, -2, 0),  # Spring constant has dimension of energy per length cubed
        "G": (-1, 3, -2, 0),  # Gravitational constant
        "epsilon0": (-1, -3, 2, 2),  # Vacuum permittivity
        "mu0": (1, 1, -2, -2),  # Vacuum permeability
    }

    OPERATOR_DIMENSIONS = {
        "add": (0, 0, 0, 0),
        "sub": (0, 0, 0, 0),
        "mul": (0, 0, 0, 0),
        "div": (0, 0, 0, 0),
        "inv": (0, 0, 0, 0),
        "power": (0, 0, 0, 0),
        "exp": (0, 0, 0, 0),
        "log": (0, 0, 0, 0),
        "sqrt": (0, 0, 0, 0),
    }

    CONSTANT_DIMENSIONS = {
        "1": (0, 0, 0, 0),
        "pi": (0, 0, 0, 0),
        "e": (0, 0, 0, 0),
        "exp(1)": (0, 0, 0, 0),
    }

    def __init__(self, mode: str = "generic"):
        self.mode = mode
        self._setup_mode_specific_dimensions()

    def _setup_mode_specific_dimensions(self):
        """Set up dimensions specific to different physics modes."""
        if self.mode == "gravity":
            self.VARIABLE_DIMENSIONS["G"] = (-1, 3, -2, 0)
            self.VARIABLE_DIMENSIONS["inv_G"] = (1, -3, 2, 0)

        elif self.mode == "electric":
            self.VARIABLE_DIMENSIONS["k_e"] = (1, 3, -2, -2)
            self.VARIABLE_DIMENSIONS["inv_k_e"] = (-1, -3, 2, 2)

    def _get_dimensional_signature(
        self, expr: sp.Expr
    ) -> Tuple[float, float, float, float]:
        """
        Compute the dimensional signature of a symbolic expression using
        bottom-up type inference: every node returns exactly ONE signature
        that is the function of its children's signatures.

        Args:
            expr: A SymPy expression

        Returns:
            A tuple (L, T, M, Q) representing the dimensional signature.
            Returns (nan, nan, nan, nan) if the expression is dimensionally
            inconsistent (e.g., r + t, exp(r)) or if an exponent is non-rational.
        """
        nan_sig = (float("nan"), float("nan"), float("nan"), float("nan"))

        if not isinstance(expr, sp.Expr):
            return (0.0, 0.0, 0.0, 0.0)

        if expr.is_Number or expr.is_NumberSymbol:
            return (0.0, 0.0, 0.0, 0.0)

        if expr.is_Symbol:
            name = str(expr)
            if name in self.VARIABLE_DIMENSIONS:
                dim = self.VARIABLE_DIMENSIONS[name]
                if isinstance(dim, tuple) and len(dim) >= 4:
                    return (float(dim[0]), float(dim[1]), float(dim[2]), float(dim[3]))
            return (0.0, 0.0, 0.0, 0.0)

        if expr.is_Pow:
            base_sig = self._get_dimensional_signature(expr.args[0])
            if any(isinstance(d, float) and d != d for d in base_sig):
                return nan_sig
            exponent = expr.args[1]
            if exponent.is_number:
                try:
                    exp_val = float(exponent.evalf())
                    if abs(exp_val - round(exp_val, 6)) < 1e-6:
                        rounded = round(exp_val, 6)
                        return tuple(d * rounded for d in base_sig)
                    else:
                        return nan_sig
                except Exception:
                    return nan_sig
            else:
                return nan_sig

        if expr.is_Mul:
            result = (0.0, 0.0, 0.0, 0.0)
            for arg in expr.args:
                arg_sig = self._get_dimensional_signature(arg)
                if any(isinstance(d, float) and d != d for d in arg_sig):
                    return nan_sig
                result = tuple(r + a for r, a in zip(result, arg_sig))
            return result

        if expr.is_Add:
            arg_sigs = [self._get_dimensional_signature(arg) for arg in expr.args]
            for sig in arg_sigs:
                if any(isinstance(d, float) and d != d for d in sig):
                    return nan_sig
            first = arg_sigs[0]
            for sig in arg_sigs[1:]:
                if any(abs(f - s) > 1e-6 for f, s in zip(first, sig)):
                    return nan_sig
            return first

        if expr.is_Function:
            func_name = str(expr.func).lower()
            trig_exp_funcs = {
                "exp", "log", "sin", "cos", "tan",
                "asin", "acos", "atan", "sinh", "cosh", "tanh",
            }
            if func_name in trig_exp_funcs:
                for arg in expr.args:
                    arg_sig = self._get_dimensional_signature(arg)
                    if any(isinstance(d, float) and d != d for d in arg_sig):
                        return nan_sig
                    if any(abs(d) > 1e-6 for d in arg_sig):
                        return nan_sig
                return (0.0, 0.0, 0.0, 0.0)
            else:
                import warnings
                warnings.warn(
                    f"Unknown function '{func_name}' in expression. "
                    f"Assuming dimensionless output.",
                    RuntimeWarning,
                )
                return (0.0, 0.0, 0.0, 0.0)

        return (0.0, 0.0, 0.0, 0.0)

    def check_consistency(
        self, expr: sp.Expr
    ) -> Tuple[bool, float, Tuple[float, float, float, float], str]:
        """
        Check if an expression is dimensionally consistent.

        Args:
            expr: A SymPy expression to validate

        Returns:
            Tuple of (is_consistent, metric, signature, message)
            - is_consistent: Whether the expression is dimensionally consistent
            - metric: A score from 0-1 (1 = perfectly consistent, 0 = completely inconsistent)
            - signature: The dimensional signature (L, T, M, Q)
            - message: A human-readable description of the validation
        """
        try:
            signature = self._get_dimensional_signature(expr)

            has_nan = any(isinstance(d, float) and d != d for d in signature)

            if has_nan:
                return (
                    False,
                    0.0,
                    signature,
                    "Dimensionally inconsistent (Add terms have mismatched "
                    "dimensions or transcendentals have dimensional arguments)",
                )

            L, T, M, Q = signature
            return (
                True,
                1.0,
                signature,
                f"Dimensionally consistent (L={L:.2f}, T={T:.2f}, M={M:.2f}, Q={Q:.2f})",
            )

        except Exception as e:
            return (False, 0.0, (0, 0, 0, 0), f"Error during validation: {str(e)}")

    def validate_expression(self, expr: sp.Expr, mode: str = None) -> Dict:
        """
        Validate an expression against specific mode constraints.

        Args:
            expr: A SymPy expression to validate
            mode: Optional mode to validate against

        Returns:
            Dictionary with validation results
        """
        if mode:
            checker = UnitChecker(mode=mode)
        else:
            checker = self

        is_consistent, metric, signature, message = checker.check_consistency(expr)

        return {
            "is_valid": is_consistent,
            "metric": metric,
            "signature": signature,
            "message": message,
            "dimensions": {
                "Length": signature[0],
                "Time": signature[1],
                "Mass": signature[2],
                "Charge": signature[3],
            },
            "mode": mode if mode else "generic",
        }

    def filter_inconsistent_candidates(
        self, candidates: List[sp.Expr], mode: str = None
    ) -> List[sp.Expr]:
        """
        Filter a list of symbolic candidates, removing physically inconsistent ones.

        Args:
            candidates: List of SymPy expressions
            mode: Optional mode to validate against

        Returns:
            Filtered list of dimensionally consistent candidates
        """
        if mode:
            checker = UnitChecker(mode=mode)
        else:
            checker = self

        valid_candidates = []
        invalid_count = 0

        for i, expr in enumerate(candidates):
            try:
                is_valid, metric, signature, message = checker.check_consistency(expr)
                if is_valid and not any(
                    isinstance(d, float) and d != d for d in signature
                ):
                    valid_candidates.append(expr)
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1

        print(
            f"Unit Checker: Filtered out {invalid_count} inconsistent candidates out of {len(candidates)}"
        )

        return valid_candidates


def is_dimensionally_consistent(expr: sp.Expr, mode: str = None) -> bool:
    """Convenience function to check if an expression is dimensionally consistent."""
    checker = UnitChecker(mode=mode)
    is_consistent, _, _, _ = checker.check_consistency(expr)
    return is_consistent
