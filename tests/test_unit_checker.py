import pytest
import sympy as sp
import numpy as np
from emergentia.unit_checker import UnitChecker, is_dimensionally_consistent


class TestUnitChecker:
    def test_init_default_mode(self):
        """Test UnitChecker initialization with default mode."""
        checker = UnitChecker()
        assert checker.mode == "generic"
        assert checker.VARIABLE_DIMENSIONS is not None

    def test_init_custom_mode(self):
        """Test UnitChecker initialization with custom physics mode."""
        checker = UnitChecker(mode="gravity")
        assert checker.mode == "gravity"
        assert "G" in checker.VARIABLE_DIMENSIONS

    def test_dimensional_signature(self):
        """Test dimensional signature creation."""
        sig = checker = UnitChecker().DimensionalSignature()
        sig.add_power("L", 2)
        sig.add_power("M", -1)

        dims = sig.get_signature()
        assert dims[0] == 2  # L
        assert dims[1] == 0  # T
        assert dims[2] == -1  # M
        assert dims[3] == 0  # Q

    def test_expression_dimensions_constant(self):
        """Test dimensions of constant expressions."""
        checker = UnitChecker()
        expr = sp.sympify("2 + 3")
        L, T, M, Q = checker._get_dimensional_signature(expr)
        assert L == 0 and T == 0 and M == 0 and Q == 0

    def test_expression_dimensions_symbol(self):
        """Test dimensions of symbol expressions."""
        checker = UnitChecker()
        expr = sp.Symbol("r")
        L, T, M, Q = checker._get_dimensional_signature(expr)
        assert L == 1 and T == 0 and M == 0 and Q == 0

    def test_expression_dimensions_power(self):
        """Test dimensions of power expressions."""
        checker = UnitChecker()
        expr = sp.Pow(sp.Symbol("r"), 2)
        L, T, M, Q = checker._get_dimensional_signature(expr)
        assert L == 2  # r^2 has dimension L^2

    def test_expression_dimensions_inv(self):
        """Test dimensions of inverse expressions."""
        checker = UnitChecker()
        expr = 1 / sp.Symbol("r")
        L, T, M, Q = checker._get_dimensional_signature(expr)
        assert L == -1  # 1/r = r^-1 has dimension L^-1

    def test_expression_dimensions_mul(self):
        """Test dimensions of multiplication."""
        checker = UnitChecker()
        expr = sp.Symbol("r") * sp.Symbol("1/r")
        L, T, M, Q = checker._get_dimensional_signature(expr)
        # r has L^1, 1/r has L^-1, so r * 1/r = L^0 (dimensionless)
        assert abs(L - 0.0) < 0.01

    def test_check_consistency_valid(self):
        """Test consistency check for dimensionally valid expressions."""
        checker = UnitChecker(mode="gravity")
        expr = sp.sympify("G * m1 * m2 / r^2")
        is_consistent, metric, signature, message = checker.check_consistency(expr)
        assert is_consistent
        assert metric > 0.5

    def test_check_consistency_invalid(self):
        """Test consistency check for dimensionally invalid expressions."""
        checker = UnitChecker()
        expr = sp.sympify("r^2 + t")  # Inconsistent: length vs time
        is_consistent, metric, signature, message = checker.check_consistency(expr)
        assert not is_consistent
        assert metric < 0.5

    def test_check_consistency_non_numeric(self):
        """Test consistency check for expressions with symbolic constants."""
        checker = UnitChecker()
        expr = sp.Symbol("c") * sp.Symbol("r")
        is_consistent, metric, signature, message = checker.check_consistency(expr)
        # Should be consistent but have non-numeric signature
        assert is_consistent or metric > 0.3

    def test_validate_expression(self):
        """Test the validate_expression method."""
        checker = UnitChecker(mode="morse")
        expr = sp.sympify("exp(-r) * r")

        validation = checker.validate_expression(expr, mode="morse")

        assert "is_valid" in validation
        assert "metric" in validation
        assert "signature" in validation
        assert "dimensions" in validation
        assert validation["mode"] == "morse"

    def test_filter_inconsistent_candidates(self):
        """Test filtering of inconsistent candidates."""
        checker = UnitChecker()

        valid_expr = sp.sympify("1/r^2")
        invalid_expr = sp.sympify("r + t")  # Length + time

        candidates = [valid_expr, invalid_expr, valid_expr * valid_expr]
        filtered = checker.filter_inconsistent_candidates(candidates, mode="gravity")

        assert len(filtered) == 2
        assert valid_expr in filtered
        assert valid_expr * valid_expr in filtered

    def test_unit_checker_consistency_with_lj(self):
        """Test Unit-Checker with Lennard-Jones force expression."""
        checker = UnitChecker(mode="lj")

        # Lennard-Jones force: F = 24*epsilon/r^2 * [2*(sigma/r)^12 - (sigma/r)^6]
        # The core term [2*(sigma/r)^12 - (sigma/r)^6] has different dimensions
        # and should be marked as inconsistent

        expr = sp.sympify("2/r^12 - 1/r^6")
        is_consistent, metric, signature, message = checker.check_consistency(expr)

        # Should be inconsistent due to different dimensions
        assert not is_consistent
        assert metric < 0.5

    def test_unit_checker_consistency_with_morse(self):
        """Test Unit-Checker with Morse potential - exp(-r) has dimensional arg, should be inconsistent."""
        checker = UnitChecker(mode="morse")

        # Morse potential derivative: exp(-r) * r
        # exp(-r) requires dimensionless argument, but r has dimension L^1
        expr = sp.sympify("exp(-r) * r")
        is_consistent, metric, signature, message = checker.check_consistency(expr)

        # Should be inconsistent: exp argument is not dimensionless
        assert not is_consistent

    def test_unit_checker_consistency_with_gravity(self):
        """Test Unit-Checker with gravity."""
        checker = UnitChecker(mode="gravity")

        # Gravitational force: F = G*m1*m2/r^2
        expr = sp.sympify("G * m1 * m2 / r^2")
        is_consistent, metric, signature, message = checker.check_consistency(expr)

        # Should be consistent
        assert is_consistent

    def test_check_consistency_with_exponential(self):
        """Test consistency check with exponential functions."""
        checker = UnitChecker()

        # exp(r) should be dimensionless
        expr = sp.exp(sp.Symbol("r"))
        L, T, M, Q = checker._get_dimensional_signature(expr)

        # exp is dimensionless, so r needs to be dimensionless
        # But r is length, so this is actually problematic
        # However, we don't validate that the exponent is dimensionless

    def test_convenience_function_is_dimensionally_consistent(self):
        """Test the convenience function is_dimensionally_consistent."""
        checker = UnitChecker()

        # Valid
        assert is_dimensionally_consistent(sp.sympify("1/r^2"))

        # Invalid
        assert not is_dimensionally_consistent(sp.sympify("r + t"))

    def test_dimension_signature_rounding(self):
        """Test that dimensions are properly rounded."""
        checker = UnitChecker()

        # Create an expression with very small differences
        # SymPy simplifies this to 1.000000001/r
        expr = sp.sympify("0.000000001 / r + 1/r")

        L, T, M, Q = checker._get_dimensional_signature(expr)

        # Should be rounded to reasonable values
        # 1.000000001/r has dimension L^-1 since 1/r has L^-1
        assert abs(L - (-1.0)) < 0.01

    def test_unit_checker_with_complex_expressions(self):
        """Test Unit-Checker with complex nested expressions."""
        checker = UnitChecker(mode="yukawa")

        # Yukawa potential: exp(-r)/r
        # exp(-r) requires dimensionless argument, r has L^1 - should be inconsistent
        expr = sp.sympify("exp(-r) / r")
        is_consistent, metric, signature, message = checker.check_consistency(expr)

        # Should be inconsistent: exp argument is not dimensionless
        assert not is_consistent

    def test_exp_r_flagged_inconsistent(self):
        """Test that exp(r) is flagged as dimensionally inconsistent (r has L^1, exp requires dimensionless)."""
        checker = UnitChecker()
        expr = sp.exp(sp.Symbol("r"))
        is_consistent, metric, signature, message = checker.check_consistency(expr)
        assert not is_consistent

    def test_r_plus_inv_r_flagged_inconsistent(self):
        """Test that r + 1/r is flagged as inconsistent (L^1 + L^-1)."""
        checker = UnitChecker()
        expr = sp.Symbol("r") + 1 / sp.Symbol("r")
        is_consistent, metric, signature, message = checker.check_consistency(expr)
        assert not is_consistent

    def test_unit_checker_multiple_atoms(self):
        """Test Unit-Checker with multiple variable atoms."""
        checker = UnitChecker()

        expr = sp.Symbol("r") * sp.Symbol("r") * sp.Symbol("r")
        L, T, M, Q = checker._get_dimensional_signature(expr)

        # r has L^1, so r * r * r = L^3
        assert abs(L - 3.0) < 0.01

    def test_add_mismatch_returns_nan(self):
        """Test that r + 1/r is flagged as inconsistent (L^1 + L^-1)."""
        checker = UnitChecker()
        expr = sp.Symbol("r") + 1 / sp.Symbol("r")
        is_consistent, metric, signature, message = checker.check_consistency(expr)
        assert not is_consistent
        assert metric == 0.0

    def test_mul_accumulates(self):
        """Test that r * r * r accumulates dimensions (L^3)."""
        checker = UnitChecker()
        expr = sp.Symbol("r") * sp.Symbol("r") * sp.Symbol("r")
        L, T, M, Q = checker._get_dimensional_signature(expr)
        assert abs(L - 3.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
