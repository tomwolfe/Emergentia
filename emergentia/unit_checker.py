import sympy as sp
from typing import Dict, List, Tuple
import numpy as np


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

    def _get_atom_dimensions(self, atom: sp.Atom) -> Tuple[float, float, float, float]:
        """
        Get dimensions for an atom in the expression tree.

        Args:
            atom: A SymPy atom (constant, symbol, etc.)

        Returns:
            A tuple (L, T, M, Q) representing the dimensional signature
        """
        if atom.is_Number or atom.is_NumberSymbol:
            return (0, 0, 0, 0)

        elif atom.is_Symbol:
            name = str(atom).lower()
            if name in self.VARIABLE_DIMENSIONS:
                dim = self.VARIABLE_DIMENSIONS[name]
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
            A tuple (L, T, M, Q) representing the dimensional signature
        """
        if not isinstance(expr, sp.Expr):
            return (0, 0, 0, 0)

        total_atoms = sum(1 for _ in expr.atoms())

        if total_atoms == 0:
            return (0, 0, 0, 0)

        L, T, M, Q = 0.0, 0.0, 0.0, 0.0

        def _traverse(node):
            nonlocal L, T, M, Q

            if node.is_Number or node.is_NumberSymbol:
                return

            elif node.is_Symbol:
                dim = self._get_atom_dimensions(node)
                L += dim[0]
                T += dim[1]
                M += dim[2]
                Q += dim[3]

            elif node.is_Function:
                dim = self.OPERATOR_DIMENSIONS.get("exp", (0, 0, 0, 0))
                L += dim[0]
                T += dim[1]
                M += dim[2]
                Q += dim[3]

                # Recurse into arguments
                for arg in node.args:
                    _traverse(arg)

            elif node.is_Pow:
                if len(node.args) == 2:
                    base, exponent = node.args
                    base_dim = self._get_dimensional_signature(base)
                    if hasattr(exponent, "evalf") and hasattr(exponent, "is_Number"):
                        try:
                            exp_val = float(exponent.evalf())
                            if abs(exp_val - round(exp_val, 6)) < 1e-6:
                                L += base_dim[0] * round(exp_val, 6)
                                T += base_dim[1] * round(exp_val, 6)
                                M += base_dim[2] * round(exp_val, 6)
                                Q += base_dim[3] * round(exp_val, 6)
                        except Exception:
                            pass

                    # Don't recurse into base - we already added its dimensions above

            elif node.is_Add:
                if len(node.args) == 2:
                    # For addition of two expressions, check if dimensions are consistent
                    arg1_dim = self._get_dimensional_signature(node.args[0])
                    arg2_dim = self._get_dimensional_signature(node.args[1])
                    # Check if all dimensions are approximately equal
                    dim_match = (
                        abs(arg1_dim[0] - arg2_dim[0]) < 1e-6
                        and abs(arg1_dim[1] - arg2_dim[1]) < 1e-6
                        and abs(arg1_dim[2] - arg2_dim[2]) < 1e-6
                        and abs(arg1_dim[3] - arg2_dim[3]) < 1e-6
                    )
                    # If dimensions don't match, the expression is not dimensionally consistent
                    if not dim_match:
                        # Mark this as invalid by returning early
                        L = float("nan")
                        T = float("nan")
                        M = float("nan")
                        Q = float("nan")
                        return
                for arg in node.args:
                    _traverse(arg)

            elif node.is_Mul:
                if len(node.args) == 2:
                    # For multiplication of two expressions, add their dimensions
                    arg1_dim = self._get_dimensional_signature(node.args[0])
                    arg2_dim = self._get_dimensional_signature(node.args[1])
                    L += arg1_dim[0] + arg2_dim[0]
                    T += arg1_dim[1] + arg2_dim[1]
                    M += arg1_dim[2] + arg2_dim[2]
                    Q += arg1_dim[3] + arg2_dim[3]
                else:
                    for arg in node.args:
                        _traverse(arg)

        _traverse(expr)
        return (L, T, M, Q)

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

            # Compute a consistency score
            # Score is higher when dimensions are well-defined
            metric = 0.0

            if dimensions_ok:
                metric = 1.0
                message = f"Dimensionally consistent (L={L:.2f}, T={T:.2f}, M={M:.2f}, Q={Q:.2f})"
            else:
                message = "Dimensional signature contains non-numeric values"

            is_consistent = dimensions_ok and (
                L == round(L, 6)
                and T == round(T, 6)
                and M == round(M, 6)
                and Q == round(Q, 6)
            )

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
            is_valid, metric, signature, message = checker.check_consistency(expr)
            if is_valid and metric > 0.8:
                valid_candidates.append(expr)
            else:
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
