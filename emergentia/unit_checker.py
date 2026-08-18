import sympy as sp
import math
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
        "G": (3, -2, -1, 0),  # Gravitational constant - dimensions M^-1 L^3 T^-2
        "epsilon0": (-1, -3, 2, 2),  # Vacuum permittivity
        "mu0": (1, 1, -2, -2),  # Vacuum permeability
        # Lennard-Jones constants
        "epsilon": (1, -2, -2, 0),  # Energy: M·L²·T⁻²
        "sigma": (0, 0, 0, 0),  # Distance, effectively dimensionless in LJ when paired with epsilon
        # Morse constants
        "De": (1, -2, -2, 0),  # Dissociation energy
        "a": (0, 0, 0, 0),  # Width parameter
        "re": (1, 0, 0, 0),  # Equilibrium position (length)
        # Buckingham constants
        "A": (1, -1, -2, 0),  # Repulsive coefficient
        "B": (0, 0, 0, 0),  # Exponential width parameter
        "C": (1, -5, -2, 0),  # Attractive coefficient (r⁻⁷ term)
        # Spring constants
        "k_spring": (1, -2, -2, 0),  # Spring constant dimension
        "r0": (1, 0, 0, 0),  # Equilibrium length
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

    # Expected output dimensions (force = M·L·T⁻²) per physics mode
    EXPECTED_OUTPUT_DIMENSIONS = {
        "spring": (1, -2, 1, 0),
        "gravity": (1, -2, 1, 0),
        "lj": (1, -2, 1, 0),
        "morse": (1, -2, 1, 0),
        "buckingham": (1, -2, 1, 0),
        "yukawa": (1, -2, 1, 0),
        "mixed": (1, -2, 1, 0),
        "electric": (1, -2, 1, 0),
        "generic": None,  # skip output check
    }

    def __init__(self, mode: str = "generic", reduced_units: bool = True):
        self.mode = mode
        self.reduced_units = reduced_units
        self._setup_mode_specific_dimensions()

    def _setup_mode_specific_dimensions(self):
        """Set up dimensions specific to different physics modes."""
        if self.mode == "gravity":
            self.VARIABLE_DIMENSIONS["G"] = (3, -2, -1, 0)
            self.VARIABLE_DIMENSIONS["inv_G"] = (-3, 2, 1, 0)

        elif self.mode == "electric":
            self.VARIABLE_DIMENSIONS["k_e"] = (1, 3, -2, -2)
            self.VARIABLE_DIMENSIONS["inv_k_e"] = (-1, -3, 2, 2)

    def _get_atom_dimensions(self, atom: sp.Atom) -> Tuple[float, float, float, float]:
        """
        Get dimensions for an atom in the expression tree.

        Args:
            atom: A SymPy atom (constant, symbol, etc.)

        Returns:
            A tuple (L, T, M, Q) representing the dimensional signature
        """
        if self.reduced_units:
            return (0, 0, 0, 0)

        if atom.is_Number or atom.is_NumberSymbol:
            return (0, 0, 0, 0)

        elif atom.is_Symbol:
            name = str(atom).lower()
            # Try lowercase first, then original case
            dim = self.VARIABLE_DIMENSIONS.get(name)
            if dim is None:
                dim = self.VARIABLE_DIMENSIONS.get(str(atom))
            if isinstance(dim, tuple) and len(dim) >= 4:
                return (dim[0], dim[1], dim[2], dim[3])
            return (0, 0, 0, 0)

        elif hasattr(atom, "func") and hasattr(atom.func, "name"):
            name = atom.func.name.lower()
            if name in self.OPERATOR_DIMENSIONS:
                return self.OPERATOR_DIMENSIONS[name]

        return (0, 0, 0, 0)

    def _get_dimensional_signature(
        self, expr: sp.Expr
    ) -> Tuple[float, float, float, float]:
        """
        Compute the dimensional signature of a symbolic expression.

        Args:
            expr: A SymPy expression

        Returns:
            A tuple (L, T, M, Q) representing the dimensional signature.
            Returns (nan, nan, nan, nan) if the expression is dimensionally inconsistent.
        """
        if not isinstance(expr, sp.Expr):
            return (0, 0, 0, 0)

        NAN_SIG = (float("nan"),) * 4

        def _sig(node) -> Tuple[float, float, float, float]:
            """Recursively compute the dimensional signature of a node."""
            if node.is_Number or node.is_NumberSymbol:
                return (0.0, 0.0, 0.0, 0.0)

            if node.is_Symbol:
                return self._get_atom_dimensions(node)

            if node.is_Pow:
                if len(node.args) == 2:
                    base, exponent = node.args
                    base_dim = _sig(base)
                    if any(math.isnan(d) for d in base_dim):
                        return NAN_SIG
                    try:
                        exp_val = float(exponent.evalf())
                        if abs(exp_val - round(exp_val, 6)) < 1e-6:
                            rounded = round(exp_val, 6)
                            return tuple(base_dim[i] * rounded for i in range(4))
                    except (TypeError, ValueError, AttributeError):
                        pass
                    return NAN_SIG
                return _sig(node.args[0]) if node.args else (0.0, 0.0, 0.0, 0.0)

            if node.is_Mul:
                result = (0.0, 0.0, 0.0, 0.0)
                for arg in node.args:
                    arg_dim = _sig(arg)
                    if any(math.isnan(d) for d in arg_dim):
                        return NAN_SIG
                    result = tuple(result[i] + arg_dim[i] for i in range(4))
                return result

            if node.is_Add:
                arg_dims = [_sig(a) for a in node.args]
                if any(any(math.isnan(d) for d in ad) for ad in arg_dims):
                    return NAN_SIG
                first = arg_dims[0]
                for d in arg_dims[1:]:
                    if not (
                        abs(first[0] - d[0]) < 1e-6
                        and abs(first[1] - d[1]) < 1e-6
                        and abs(first[2] - d[2]) < 1e-6
                        and abs(first[3] - d[3]) < 1e-6
                    ):
                        return NAN_SIG
                return first

            if node.is_Function:
                func_name = str(node.func).lower()
                trig_exp_funcs = {
                    "exp", "log", "sin", "cos", "tan",
                    "asin", "acos", "atan", "sinh", "cosh", "tanh",
                }
                if func_name in trig_exp_funcs:
                    for arg in node.args:
                        arg_dim = _sig(arg)
                        if any(math.isnan(d) for d in arg_dim):
                            return NAN_SIG
                        if not (
                            abs(arg_dim[0]) < 1e-6
                            and abs(arg_dim[1]) < 1e-6
                            and abs(arg_dim[2]) < 1e-6
                            and abs(arg_dim[3]) < 1e-6
                        ):
                            return NAN_SIG
                    return (0.0, 0.0, 0.0, 0.0)
                if func_name == "sqrt":
                    for arg in node.args:
                        arg_dim = _sig(arg)
                        if any(math.isnan(d) for d in arg_dim):
                            return NAN_SIG
                    arg_dim = _sig(node.args[0])
                    return tuple(arg_dim[i] * 0.5 for i in range(4))
                return (0.0, 0.0, 0.0, 0.0)

            return (0.0, 0.0, 0.0, 0.0)

        return _sig(expr)

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
            L, T, M, Q = signature

            # Check that all dimensions are well-behaved
            dimensions_ok = True
            for dim in signature:
                if not (isinstance(dim, (int, float)) and dim == round(dim, 6)):
                    dimensions_ok = False

            # Check for NaN/zoo in the expression structure
            has_nan = expr.has(sp.nan) or expr.has(sp.zoo)

            if self.reduced_units:
                # In reduced units mode, all terminal symbols are dimensionless.
                # Only flag structural issues (NaN/zoo, non-numeric signatures).
                is_consistent = dimensions_ok and not has_nan
                metric = 1.0 if is_consistent else 0.0
                if is_consistent:
                    message = "Dimensionally consistent (reduced units mode)"
                elif has_nan:
                    message = "Expression contains structural issues (NaN/zoo)"
                else:
                    message = "Dimensional signature contains non-numeric values"
                return (is_consistent, metric, signature, message)

            # Check against expected output dimensions for the mode
            expected_dims = self.EXPECTED_OUTPUT_DIMENSIONS.get(self.mode)
            mode_ok = True
            if expected_dims is not None:
                exp_L, exp_T, exp_M, exp_Q = expected_dims
                mode_ok = (
                    abs(L - exp_L) < 1e-6
                    and abs(T - exp_T) < 1e-6
                    and abs(M - exp_M) < 1e-6
                    and abs(Q - exp_Q) < 1e-6
                )

            # Compute a consistency score
            # Score is higher when dimensions are well-defined and match expected output
            metric = 0.0

            if dimensions_ok and mode_ok:
                metric = 1.0
                message = f"Dimensionally consistent (L={L:.2f}, T={T:.2f}, M={M:.2f}, Q={Q:.2f})"
            elif dimensions_ok and not mode_ok:
                metric = 0.3
                message = f"Dimensionally valid but output dimension mismatch: expected {expected_dims}, got {signature}"
            else:
                message = "Dimensional signature contains non-numeric values"

            is_consistent = dimensions_ok and mode_ok

            return (is_consistent, metric, signature, message)

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
            checker = UnitChecker(mode=mode, reduced_units=self.reduced_units)
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
            candidates: List of symbolic expressions
            mode: Optional mode to validate against

        Returns:
            Filtered list of dimensionally consistent candidates
        """
        if mode:
            checker = UnitChecker(mode=mode, reduced_units=self.reduced_units)
        else:
            checker = self

        valid_candidates = []
        invalid_count = 0

        for i, expr in enumerate(candidates):
            try:
                is_valid, metric, signature, message = checker.check_consistency(expr)
                if is_valid and not any(
                    math.isnan(d) for d in signature
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


def is_dimensionally_consistent(expr: sp.Expr, mode: str = None, reduced_units: bool = True) -> bool:
    """Convenience function to check if an expression is dimensionally consistent."""
    checker = UnitChecker(mode=mode, reduced_units=reduced_units)
    is_consistent, _, _, _ = checker.check_consistency(expr)
    return is_consistent