"""Mutation teeth for the mixed-gap closure (09-gate survivors).

Pins:
  - DiscoveryPipeline._generalization_penalty exact value, shape-mismatch,
    non-finite, and raising-function branches.
  - _select_generalizing picks the template over a singular GP and the GP
    over a singular template.
  - validate_conservativeness True path (kills return-True flip) and the
    imaginary-unit rejection.
  - PhysicsPriorProvider._parse fail-closed None on unparseable input.
"""

import numpy as np
import pytest
import sympy as sp

from emergentia.engine import DiscoveryPipeline
from emergentia.physics_priors import PhysicsPriorProvider


def _const_fn(c):
    def fn(x):
        return np.full_like(np.asarray(x, dtype=float), c, dtype=float)
    return fn


def test_generalization_penalty_exact_constant():
    # mean(4) + max(2)^2 = 8. Kills arithmetic mutants on either the
    # max-abs term or the **2 exponent.
    p = DiscoveryPipeline._generalization_penalty(_const_fn(2.0))
    assert p == pytest.approx(8.0, rel=1e-9)


def test_generalization_penalty_shape_mismatch_is_inf():
    p = DiscoveryPipeline._generalization_penalty(lambda x: np.array([1.0]))
    assert p == float("inf")


def test_generalization_penalty_nonfinite_is_inf():
    def fn(x):
        y = np.ones_like(np.asarray(x, dtype=float))
        y[::2] = np.nan
        return y
    assert DiscoveryPipeline._generalization_penalty(fn) == float("inf")


def test_generalization_penalty_raising_fn_is_inf():
    def fn(x):
        raise ValueError("boom")
    assert DiscoveryPipeline._generalization_penalty(fn) == float("inf")


def test_select_generalizing_prefers_nonsingular_template():
    r = sp.Symbol("r")
    gp_expr = 1 / (r - 5)  # pole inside the extended range
    template_expr = 2 * r + 1
    assert DiscoveryPipeline._select_generalizing(r, gp_expr, template_expr) == template_expr


def test_select_generalizing_keeps_clean_gp_over_singular_template():
    r = sp.Symbol("r")
    gp_expr = 2 * r + 1
    template_expr = 1 / (r - 5)
    assert DiscoveryPipeline._select_generalizing(r, gp_expr, template_expr) == gp_expr


def test_validate_conservativeness_true_path():
    pipe = DiscoveryPipeline.__new__(DiscoveryPipeline)
    r = sp.Symbol("r")
    assert pipe.validate_conservativeness(r**2) is True


def test_validate_conservativeness_rejects_imaginary():
    pipe = DiscoveryPipeline.__new__(DiscoveryPipeline)
    r = sp.Symbol("r")
    assert pipe.validate_conservativeness(sp.I * r) is False


def test_parse_fail_closed_returns_none():
    provider = PhysicsPriorProvider()
    assert provider._parse("not a valid expression @@") is None
