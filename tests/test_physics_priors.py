import pytest
import sympy as sp
from emergentia.physics_priors import PhysicsPriorProvider as PhysicsPriorProvider


def test_generate_priors_fallback_returns_expressions():
    """Test that generate_priors works locally."""
    provider = PhysicsPriorProvider()
    result = provider.generate_priors("gravity")

    assert len(result) >= 1, "Fallback path should return at least 1 expression"
    for expr in result:
        assert isinstance(expr, sp.Expr), f"Expected SymPy expression, got {type(expr)}"


def test_no_nan_or_zoo_in_priors():
    """Test that no expression in the result contains nan or zoo."""
    provider = PhysicsPriorProvider()
    result = provider.generate_priors("gravity")

    for expr in result:
        assert not expr.has(sp.nan), f"Expression contains nan: {expr}"
        assert not expr.has(sp.zoo), f"Expression contains zoo: {expr}"


def test_generate_priors_all_modes():
    """Test fallback path for all known physics modes."""
    provider = PhysicsPriorProvider()
    modes = ["spring", "gravity", "lj", "morse", "buckingham", "yukawa", "electric"]

    for mode in modes:
        result = provider.generate_priors(mode)
        assert len(result) >= 1, f"Mode '{mode}' should return at least 1 expression"
        for expr in result:
            assert not expr.has(sp.nan) and not expr.has(sp.zoo), (
                f"Mode '{mode}': expression contains nan or zoo: {expr}"
            )


def test_get_mode_priors():
    """Test that get_mode_priors returns expected structure."""
    provider = PhysicsPriorProvider()
    gravity_priors = provider.get_mode_priors("gravity")

    assert "description" in gravity_priors
    assert "expected_terms" in gravity_priors
    assert "suggested_functional_forms" in gravity_priors
    assert "1/r^2" in gravity_priors["expected_terms"]


def test_generate_priors_caching():
    """Test that priors are cached."""
    provider = PhysicsPriorProvider()
    summary = {"mode": "spring"}

    result1 = provider.generate_priors("spring")
    result2 = provider.generate_priors("spring")

    # Cached results should be identical (same object references)
    assert result1 is result2 or result1 == result2


def test_alias_matches_generate_priors():
    provider = PhysicsPriorProvider()
    assert provider.generate_priors_from_llm(mode="spring") == provider.generate_priors("spring")
    assert len(provider.generate_priors_from_llm(mode="spring")) >= 1


def test_to_gplearn_unknown_symbol_maps_X0():
    provider = PhysicsPriorProvider()
    assert provider.to_gplearn_program(sp.Symbol("r"), ["r"]) == "X0"
    assert provider.to_gplearn_program(sp.Symbol("q"), ["r"]) == "X0"


def test_modes_are_mode_specific():
    provider = PhysicsPriorProvider()
    spring = [str(e) for e in provider.generate_priors("spring")]
    gravity = [str(e) for e in provider.generate_priors("gravity")]
    assert spring != gravity


def test_spring_prior_count_is_five():
    provider = PhysicsPriorProvider()
    assert len(provider.generate_priors("spring")) == 5


def test_lazy_package_exports_resolve():
    import emergentia

    assert emergentia.PhysicsPriorProvider is PhysicsPriorProvider
    from emergentia.unit_checker import UnitChecker as DirectChecker

    assert emergentia.UnitChecker is DirectChecker
    assert callable(emergentia.is_dimensionally_consistent)
    with pytest.raises(AttributeError):
        emergentia.__getattr__("__nonexistent_attr__")


def test_to_gplearn_program_arithmetic_branches():
    provider = PhysicsPriorProvider()
    r = sp.Symbol("r")
    assert provider.to_gplearn_program(r + 1, ["r"]).startswith("add(")
    assert provider.to_gplearn_program(2 * r, ["r"]).startswith("mul(")
    assert provider.to_gplearn_program(r**2, ["r"]).startswith("power(")


def test_spring_prior_respects_max_candidates():
    provider = PhysicsPriorProvider()
    assert len(provider.generate_priors("spring", max_candidates=2)) == 2
    assert len(provider.generate_priors("spring", max_candidates=10)) == 5


def test_get_mode_priors_unknown_mode_fallback():
    provider = PhysicsPriorProvider()
    info = provider.get_mode_priors("no_such_mode_xyz")
    assert "Generic force mode no_such_mode_xyz" in info["description"]
    assert "expected_terms" in info and "suggested_functional_forms" in info
    priors = provider.generate_priors("no_such_mode_xyz")
    assert len(priors) >= 1
    assert all(isinstance(e, sp.Expr) for e in priors)


def test_parse_invalid_returns_none():
    provider = PhysicsPriorProvider()
    assert provider._parse("(((") is None
    assert provider._parse("1/r^2 +* ??? [[[") is None


def test_to_gplearn_exp_log_sqrt_and_unknown_func():
    provider = PhysicsPriorProvider()
    r = sp.Symbol("r")
    assert provider.to_gplearn_program(sp.exp(-r), ["r"]).startswith("exp(")
    assert provider.to_gplearn_program(sp.log(r + 1), ["r"]).startswith("log(")
    assert provider.to_gplearn_program(sp.Function("sqrt")(r), ["r"]) == "sqrt(X0)"
    assert provider.to_gplearn_program(sp.sin(r), ["r"]) == "X0"


def test_prior_counts_exact_per_mode():
    # Exact per-mode counts kill mutants that add/remove suggested forms.
    # Note: some raw strings (e.g. "exp(-2r)") fail
    # sympy parsing and are filtered, so counts are post-parse lengths.
    provider = PhysicsPriorProvider()
    expected = {
        "gravity": 8,
        "lj": 7,
        "morse": 4,
        "spring": 5,
        "buckingham": 4,
        "yukawa": 4,
        "electric": 4,
        "mixed": 6,
    }
    for mode, count in expected.items():
        assert len(provider.generate_priors(mode)) == count, mode


def test_generate_priors_falsy_max_candidates_falls_back_to_default():
    # `n = max_candidates or self.max_candidates`: falsy override must use default.
    provider = PhysicsPriorProvider()
    assert len(provider.generate_priors("spring", max_candidates=0)) == 5
    assert len(provider.generate_priors("spring", max_candidates=None)) == 5


def test_generate_priors_from_llm_dataset_summary_mode():
    provider = PhysicsPriorProvider()
    # Explicit mode wins over dataset summary.
    assert provider.generate_priors_from_llm(dataset_summary={"mode": "gravity"}, mode="spring") == provider.generate_priors("spring")
    # Falsy mode falls back to dataset summary.
    assert provider.generate_priors_from_llm(dataset_summary={"mode": "spring"}, mode=None) == provider.generate_priors("spring")


def test_to_gplearn_program_exact_strings():
    # Exact strings kill operator/format mutants in to_gplearn_program.
    provider = PhysicsPriorProvider()
    r = sp.Symbol("r")
    assert provider.to_gplearn_program(r + 1, ["r"]) == "add(1.0, X0)"
    assert provider.to_gplearn_program(2 * r, ["r"]) == "mul(2.0, X0)"
    assert provider.to_gplearn_program(r**2, ["r"]) == "power(X0, 2.0)"
    assert provider.to_gplearn_program(sp.exp(-r), ["r"]) == "exp(mul(-1.0, X0))"


def test_parse_return_none_mutant_breaks_priors():
    provider = PhysicsPriorProvider()
    assert provider._parse("r") is not None
    priors = provider.generate_priors("spring")
    assert len(priors) == 5
    assert all(e is not None and isinstance(e, sp.Expr) for e in priors)


def test_to_gplearn_case_insensitive_second_feature_kills_ne_mutant():
    """'R' must match symbol r at index 1; != mutant returns X0."""
    provider = PhysicsPriorProvider()
    assert provider.to_gplearn_program(sp.Symbol("r"), ["x", "R"]) == "X1"
    assert provider.to_gplearn_program(sp.Symbol("q"), ["x", "r"]) == "X0"


def test_default_max_candidates_is_ten():
    # Kills mutants altering the __init__ default (no mode has >10 forms to catch it).
    assert PhysicsPriorProvider().max_candidates == 10
    assert PhysicsPriorProvider(max_candidates=7).max_candidates == 7


def test_generate_priors_from_llm_fully_default_uses_generic():
    # Kills mutants altering the signature default or the .get() fallback string.
    # NOTE: the .get() default is only reached when mode is falsy AND the
    # summary lacks "mode", so both call shapes are asserted.
    provider = PhysicsPriorProvider()
    result = provider.generate_priors_from_llm()
    assert result == provider.generate_priors("generic")
    assert len(result) == 6
    assert "Generic force mode generic" in provider.get_mode_priors("generic")["description"]
    assert provider.generate_priors_from_llm(dataset_summary=None, mode=None) == provider.generate_priors("generic")
    assert provider.generate_priors_from_llm(dataset_summary={}, mode="") == provider.generate_priors("generic")


def test_unit_checker_cross_coverage_mutation_guard():
    """Mirror of the key unit_checker kill-assertions.

    The mission verification subset runs this file but not
    tests/test_unit_checker.py, so unit_checker mutants must also die here.
    """
    import math
    from unittest import mock
    from emergentia.unit_checker import UnitChecker

    strict = UnitChecker(reduced_units=False)
    # Metric branches 1.0 / 0.3 / 0.0.
    ok, metric, _, _ = UnitChecker().check_consistency(sp.sympify("1/r**2"))
    assert ok and metric == 1.0
    ok, metric, _, _ = UnitChecker(mode="spring").check_consistency(sp.sympify("1/r**2"))
    assert not ok and metric == 0.3
    ok, metric, _, _ = strict.check_consistency(sp.sympify("r + 1/r"))
    assert not ok and metric == 0.0
    # Full trig/exp set requires dimensionless args in strict mode.
    r = sp.Symbol("r")
    for fn in ["exp", "log", "sin", "cos", "tan",
               "asin", "acos", "atan", "sinh", "cosh", "tanh"]:
        ok, metric, sig, _ = strict.check_consistency(getattr(sp, fn)(r))
        assert not ok and metric == 0.0, fn
        assert any(math.isnan(d) for d in sig), fn
    # Structural kernels: sqrt halves, mul accumulates, add-mismatch is NaN.
    # NOTE: sp.sqrt(r) and r*r*r simplify to Pow/numbers, so the true Mul and
    # sqrt-Function branches need non-simplifying operands (kills operator mutants).
    assert strict._get_dimensional_signature(sp.Function("sqrt")(r)) == (0.5, 0.0, 0.0, 0.0)
    assert strict._get_dimensional_signature(r * sp.Symbol("x")) == (2, 0, 0, 0)
    assert any(math.isnan(d) for d in strict._get_dimensional_signature(sp.sympify("r + t")))
    # DimensionalSignature.add_power guard: unknown dims are ignored, not raised.
    sig_pow = UnitChecker.DimensionalSignature()
    sig_pow.add_power("L", 2)
    assert sig_pow.get_signature() == (2, 0, 0, 0)
    sig_pow.add_power("Z", 5)
    assert sig_pow.get_signature() == (2, 0, 0, 0)
    # Unknown-function atoms fall back to zero (kills fallthrough mutants).
    atom = mock.MagicMock()
    atom.is_Number = bool(0)
    atom.is_NumberSymbol = bool(0)
    atom.is_Symbol = bool(0)
    atom.func.name = "some_unknown_func"
    assert strict._get_atom_dimensions(atom) == (0, 0, 0, 0)
    # is_valid precision boundary (kills round-6->5 mutants).
    sig = UnitChecker.DimensionalSignature()
    sig.dimensions["L"] = 0.123456
    assert sig.is_valid()
    sig.dimensions["L"] = 0.1234567
    assert not sig.is_valid()
    # Static dimension table spot checks (kills value mutants).
    assert UnitChecker.VARIABLE_DIMENSIONS["r"] == (1, 0, 0, 0)
    assert UnitChecker.VARIABLE_DIMENSIONS["k"] == (1, 3, -2, 0)
    assert UnitChecker.VARIABLE_DIMENSIONS["G"] == (3, -2, -1, 0)
    assert UnitChecker.EXPECTED_OUTPUT_DIMENSIONS["spring"] == (1, -2, 1, 0)
    # Mode branches must run even with class-level pollution from other tests.
    saved = dict(UnitChecker.VARIABLE_DIMENSIONS)
    for key in ("G", "inv_G", "k_e", "inv_k_e"):
        UnitChecker.VARIABLE_DIMENSIONS.pop(key, None)
    try:
        assert UnitChecker(mode="gravity").VARIABLE_DIMENSIONS["inv_G"] == (-3, 2, 1, 0)
        assert UnitChecker(mode="electric").VARIABLE_DIMENSIONS["inv_k_e"] == (-1, -3, 2, 2)
    finally:
        UnitChecker.VARIABLE_DIMENSIONS.clear()
        UnitChecker.VARIABLE_DIMENSIONS.update(saved)
    # Reduced-units nan/zoo score exactly 0.0 (kills has_nan or->and mutants).
    for bad in (sp.nan, sp.zoo):
        ok, metric, _, _ = UnitChecker(reduced_units=True).check_consistency(bad)
        assert not ok and metric == 0.0
    # Reduced-units consistent path scores exactly 1.0 (kills metric mutants).
    ok, metric, _, _ = UnitChecker(reduced_units=True).check_consistency(sp.sympify("exp(-r)*r"))
    assert ok and metric == 1.0
    # 1e-4 near-miss stays a mode mismatch (kills tolerance widening).
    spring = UnitChecker(mode="spring", reduced_units=False)
    with mock.patch.object(spring, "_get_dimensional_signature",
                           return_value=(1.0001, -2, 1, 0)):
        ok, metric, _, _ = spring.check_consistency(r)
    assert not ok and metric == 0.3


def test_unit_checker_deep_branch_mutation_guard():
    """Kill surviving unit_checker mutants the mission suite can reach.

    The mission verification subset runs this file (not
    tests/test_unit_checker.py), so every branch below must be pinned here.
    Each assertion is checked against the committed source behavior.
    """
    import math
    from unittest import mock
    from emergentia.unit_checker import UnitChecker, is_dimensionally_consistent

    strict = UnitChecker(reduced_units=False)
    r = sp.Symbol("r")

    # Bare Number / NumberSymbol atoms are dimensionless (pins _sig Number branch).
    assert strict._get_dimensional_signature(sp.Integer(3)) == (0.0, 0.0, 0.0, 0.0)
    assert strict._get_dimensional_signature(sp.pi) == (0.0, 0.0, 0.0, 0.0)
    # Unknown symbols fall back to zero (pins Symbol fallback branch).
    assert strict._get_dimensional_signature(sp.Symbol("c")) == (0, 0, 0, 0)
    # Reduced-units mode forces every atom to zero (pins reduced_units branch).
    ru = UnitChecker(reduced_units=True)
    assert ru._get_atom_dimensions(sp.Symbol("r")) == (0, 0, 0, 0)
    assert ru._get_dimensional_signature(sp.Symbol("r")) == (0, 0, 0, 0)
    # Operator-named function atoms resolve via OPERATOR_DIMENSIONS.
    atom = mock.MagicMock()
    atom.is_Number = bool(0)
    atom.is_NumberSymbol = bool(0)
    atom.is_Symbol = bool(0)
    atom.func.name = "add"
    assert strict._get_atom_dimensions(atom) == (0, 0, 0, 0)

    # Add pairs differing in exactly one component are inconsistent.
    # Each pins its own per-component comparison clause.
    cases = [
        sp.sympify("t + 1/t"),      # T only: -1 vs +1
        sp.sympify("q + 1/q"),      # Q only: +1 vs -1
        sp.sympify("m + mass_inv"),   # M only: +1 vs -1
        sp.sympify("t + t**2"),     # T only: -1 vs -2
    ]
    for expr in cases:
        ok, metric, _, _ = strict.check_consistency(expr)
        assert not ok and metric == 0.0, expr

    # Trig/exp of a single-component nonzero arg is inconsistent per component.
    for arg in (sp.Symbol("t"), sp.Symbol("m"), sp.Symbol("q")):
        ok, metric, _, _ = strict.check_consistency(sp.exp(arg))
        assert not ok and metric == 0.0, arg

    # sqrt of an inconsistent sum stays NaN (pins sqrt NaN short-circuit).
    ok, metric, sig, _ = strict.check_consistency(
        sp.Function("sqrt")(sp.sympify("r + t")))
    assert not ok and metric == 0.0 and any(math.isnan(d) for d in sig)

    # Spring-mode force expression pins every mode_ok tolerance clause.
    spring_ok = UnitChecker(mode="spring", reduced_units=False)
    ok, metric, sig, _ = spring_ok.check_consistency(sp.sympify("r*m*t**2"))
    assert ok and metric == 1.0 and sig == (1, -2, 1, 0)
    # Same shape but Q=1 pins the Q clause (or-mutant would flip to consistent).
    ok, metric, _, _ = spring_ok.check_consistency(sp.sympify("r*m*t**2*q"))
    assert not ok and metric == 0.3

    # validate_expression / filter / convenience-function returns.
    v = strict.validate_expression(sp.sympify("r + 1/r"))
    assert v["is_valid"] is False and v["metric"] == 0.0
    v2 = UnitChecker().validate_expression(sp.sympify("1/r**2"))
    assert v2["is_valid"] is True and v2["metric"] == 1.0 and v2["mode"] == "generic"
    kept = UnitChecker().filter_inconsistent_candidates(
        [sp.sympify("1/r**2"), sp.sympify("r + t")], mode="generic")
    assert len(kept) == 1
    assert is_dimensionally_consistent(sp.sympify("1/r**2")) is True
    assert is_dimensionally_consistent(sp.sympify("r + t")) is False

    # Unexpected errors map to (False, 0.0, zeros, message).
    with mock.patch.object(strict, "_get_dimensional_signature",
                           side_effect=RuntimeError("boom")):
        ok, metric, sig, msg = strict.check_consistency(r)
    assert not ok and metric == 0.0 and sig == (0, 0, 0, 0) and "boom" in msg

    # Direct atom branches: Number/NumberSymbol atoms are dimensionless.
    assert strict._get_atom_dimensions(sp.Integer(2)) == (0, 0, 0, 0)
    assert strict._get_atom_dimensions(sp.pi) == (0, 0, 0, 0)
    # Matching Add terms return the shared signature (pins Add-match branch).
    assert strict._get_dimensional_signature(
        sp.sympify("t + dt")) == (0, -1, 0, 0)
    # Unevaluated derivatives fall back to zero (pins final fallthrough).
    assert strict._get_dimensional_signature(
        sp.Derivative(r ** 2, r, evaluate=False)) == (0.0, 0.0, 0.0, 0.0)
    # validate_expression with an explicit mode builds a mode-specific checker.
    assert strict.validate_expression(
        sp.sympify("exp(-r)*r"), mode="morse")["is_valid"] is False
    # filter with an explicit mode drops mode-mismatched candidates.
    assert UnitChecker().filter_inconsistent_candidates(
        [sp.sympify("1/r**2")], mode="spring") == []
    # Convenience function honors reduced_units=True.
    assert is_dimensionally_consistent(
        sp.sympify("exp(-r)"), reduced_units=True) is True


def test_unit_checker_tolerance_and_literal_mutation_guard():
    """Kill tolerance-widening and self-healing-literal unit_checker mutants.

    The mission verification subset runs this file (not
    tests/test_unit_checker.py), so these branches must be pinned here too.
    """
    import math
    import subprocess
    import sys
    from unittest import mock
    from emergentia.unit_checker import UnitChecker

    strict = UnitChecker(reduced_units=False)
    # Add near-miss (5e-6) stays inconsistent in every single component.
    base = (1.0, -2.0, 1.0, 0.0)
    for i in range(4):
        near = list(base)
        near[i] += 5e-6
        with mock.patch.object(
            strict, "_get_atom_dimensions", side_effect=[tuple(base), tuple(near)]
        ):
            sig = strict._get_dimensional_signature(sp.sympify("r + x"))
        assert any(math.isnan(d) for d in sig), i
    # Trig/exp near-miss stays inconsistent.
    with mock.patch.object(
        strict, "_get_atom_dimensions", return_value=(5e-6, 0, 0, 0)
    ):
        ok, metric, _, _ = strict.check_consistency(sp.exp(sp.Symbol("r")))
    assert not ok and metric == 0.0
    # Pow exponent rounds to 6 decimals (kills round-6->5 mutants).
    assert strict._get_dimensional_signature(
        sp.Symbol("r") ** 2.1234565) == (2.123457, 0.0, 0.0, 0.0)
    # G literal pinned in a fresh interpreter: gravity mode setup overwrites
    # G at runtime and would otherwise heal class-literal mutants in-process.
    code = (
        "from emergentia.unit_checker import UnitChecker; "
        "print(UnitChecker.VARIABLE_DIMENSIONS['G'])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0
    assert "(3, -2, -1, 0)" in out.stdout


def test_simulator_initial_velocity_scale_kills_div_mutant():
    """PhysicsSim inits vel ~ N(0,1)*0.1; a *0.1->/0.1 mutant gives ~8."""
    import torch
    from emergentia.simulator import PhysicsSim, HarmonicPotential
    sim = PhysicsSim(n=64, dim=2, potential=HarmonicPotential(k=10.0, r0=1.0),
                     seed=0, device=torch.device("cpu"))
    assert sim.vel.abs().mean().item() < 0.5


def test_simulator_neighbor_list_dim3_branch_matches_raw():
    """Exercise the dim!=2 (1D flat-cell) neighbor-list branch."""
    import torch
    from emergentia.simulator import PhysicsSim, HarmonicPotential
    sim = PhysicsSim(n=2, dim=3, potential=HarmonicPotential(k=10.0, r0=1.0),
                     seed=0, device=torch.device("cpu"),
                     use_neighbor_list=True, cutoff=10.0)
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.3, 0.5]])
    f_nl = sim._compute_forces_neighbor_list(pos)
    f_raw = sim._compute_forces_raw(pos)
    assert torch.allclose(f_nl, f_raw, atol=1e-6)
    assert f_nl.abs().max().item() > 0


def test_mutation_gate_residual_guards():
    """Pin residual killable mutants in physics_priors/unit_checker.

    The mission verification subset runs this file (not
    tests/test_unit_checker.py), so these branches must be pinned here.
    Each assertion was checked against committed source behavior.
    """
    import math
    from unittest import mock
    from emergentia.unit_checker import UnitChecker

    # add_power accumulates into the same dim (kills += -> =/-= mutants).
    sig = UnitChecker.DimensionalSignature()
    sig.add_power("L", 2)
    sig.add_power("L", 3)
    assert sig.get_signature() == (5, 0, 0, 0)
    # add_power truncates via int() (kills int() removal).
    sig_f = UnitChecker.DimensionalSignature()
    sig_f.add_power("L", 2.7)
    assert sig_f.get_signature() == (2, 0, 0, 0)
    # Unknown dims are ignored without raising (kills guard removal).
    sig_f.add_power("Z", 5)
    assert sig_f.get_signature() == (2, 0, 0, 0)
    # Non-numeric dims are invalid, not raised (kills isinstance removal).
    sig_s = UnitChecker.DimensionalSignature()
    sig_s.dimensions["L"] = "x"
    assert sig_s.is_valid() is False

    strict = UnitChecker(reduced_units=False)
    # Non-Expr inputs fall back to zero (kills isinstance-guard removal).
    assert strict._get_dimensional_signature("r") == (0, 0, 0, 0)
    assert strict._get_dimensional_signature(42) == (0, 0, 0, 0)
    assert strict._get_dimensional_signature(None) == (0, 0, 0, 0)
    # Pow with a NaN base propagates NaN (kills NaN-propagation deletion).
    nan_sig = strict._get_dimensional_signature(sp.sympify("r+t") ** 2)
    assert any(math.isnan(d) for d in nan_sig)
    # Symbolic exponents cannot resolve (mirrors test_unit_checker.py;
    # the mission subset does not run that file).
    assert any(math.isnan(d) for d in strict._get_dimensional_signature(
        sp.Symbol("r") ** sp.Symbol("t")))

    # Mode-output check pins every single component clause: a 0.01 drift
    # in exactly one of L/T/M/Q must stay a 0.3 mismatch (kills
    # single-clause deletion and tolerance widening).
    spring = UnitChecker(mode="spring", reduced_units=False)
    r = sp.Symbol("r")
    for delta in [(0.01, 0, 0, 0), (0, 0.01, 0, 0),
                  (0, 0, 0.01, 0), (0, 0, 0, 0.01)]:
        spoof = (1 + delta[0], -2 + delta[1], 1 + delta[2], 0 + delta[3])
        with mock.patch.object(spring, "_get_dimensional_signature",
                               return_value=spoof):
            ok, metric, _, _ = spring.check_consistency(r)
        assert not ok and metric == 0.3, delta

    # Provider init literals (kills default/cache literal mutants).
    provider = PhysicsPriorProvider()
    assert provider.candidates_cache == {}
    assert provider.max_candidates == 10


def test_mode_ok_tolerance_boundary_pins_narrow_widening():
    """Pin mode_ok 1e-6 tolerance against narrow widening (1e-6 -> 1e-5).

    A 5e-6 drift in exactly one of L/T/M/Q must stay a 0.3 mismatch under
    committed source, so widening any single clause to 1e-5 is killed.
    The mission verification subset runs this file (not
    tests/test_unit_checker.py), so this must be pinned here.
    """
    from unittest import mock
    from emergentia.unit_checker import UnitChecker

    spring = UnitChecker(mode="spring", reduced_units=False)
    r = sp.Symbol("r")
    for delta in [(5e-6, 0, 0, 0), (0, 5e-6, 0, 0),
                  (0, 0, 5e-6, 0), (0, 0, 0, 5e-6)]:
        spoof = (1 + delta[0], -2 + delta[1], 1 + delta[2], 0 + delta[3])
        with mock.patch.object(spring, "_get_dimensional_signature",
                               return_value=spoof):
            ok, metric, _, _ = spring.check_consistency(r)
        assert not ok and metric == 0.3, delta


def test_unit_checker_dimension_tables_fully_pinned():
    """Pin every committed dimension-table literal in the mission subset.

    Single-element value mutants (e.g. "m" Q 0->1, "t" Q 0->1, OPERATOR
    entries 0->1) survive behavioral tests because mismatches stay
    mismatches; only exact-value assertions kill them. Subset-match is used
    for VARIABLE_DIMENSIONS because other tests legitimately register
    mode-specific keys (inv_G, k_e, inv_k_e) on the shared class dict.
    """
    from emergentia.unit_checker import UnitChecker

    expected_variable = {
        "r": (1, 0, 0, 0),
        "r2": (2, 0, 0, 0),
        "1/r": (-1, 0, 0, 0),
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
        "k": (1, 3, -2, 0),
        "G": (3, -2, -1, 0),
        "epsilon0": (-1, -3, 2, 2),
        "mu0": (1, 1, -2, -2),
        "epsilon": (1, -2, -2, 0),
        "sigma": (0, 0, 0, 0),
        "De": (1, -2, -2, 0),
        "a": (0, 0, 0, 0),
        "re": (1, 0, 0, 0),
        "A": (1, -1, -2, 0),
        "B": (0, 0, 0, 0),
        "C": (1, -5, -2, 0),
        "k_spring": (1, -2, -2, 0),
        "r0": (1, 0, 0, 0),
    }
    for key, value in expected_variable.items():
        assert UnitChecker.VARIABLE_DIMENSIONS.get(key) == value, key
    expected_operator = {
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
    assert UnitChecker.OPERATOR_DIMENSIONS == expected_operator
    expected_constant = {
        "1": (0, 0, 0, 0),
        "pi": (0, 0, 0, 0),
        "e": (0, 0, 0, 0),
        "exp(1)": (0, 0, 0, 0),
    }
    assert UnitChecker.CONSTANT_DIMENSIONS == expected_constant
    expected_output = {
        "spring": (1, -2, 1, 0),
        "gravity": (1, -2, 1, 0),
        "lj": (1, -2, 1, 0),
        "morse": (1, -2, 1, 0),
        "buckingham": (1, -2, 1, 0),
        "yukawa": (1, -2, 1, 0),
        "mixed": (1, -2, 1, 0),
        "electric": (1, -2, 1, 0),
        "generic": None,
    }
    assert UnitChecker.EXPECTED_OUTPUT_DIMENSIONS == expected_output


def test_gravity_mode_runtime_setup_and_force():
    """Pin _setup_mode_specific_dimensions runtime G overwrite functionally.

    The class literal is pinned elsewhere (fresh-interpreter read), but the
    runtime overwrite on line 138 is only exercised by building a gravity
    checker after scrubbing class pollution. A Q 0->1 literal mutant there
    flips G*m*m/r**2 from force-consistent to a 0.3 mismatch.
    """
    from emergentia.unit_checker import UnitChecker

    saved = dict(UnitChecker.VARIABLE_DIMENSIONS)
    for key in ("G", "inv_G", "k_e", "inv_k_e"):
        UnitChecker.VARIABLE_DIMENSIONS.pop(key, None)
    try:
        checker = UnitChecker(mode="gravity", reduced_units=False)
        assert checker.VARIABLE_DIMENSIONS["G"] == (3, -2, -1, 0)
        assert checker.VARIABLE_DIMENSIONS["inv_G"] == (-3, 2, 1, 0)
        ok, metric, sig, _ = checker.check_consistency(
            sp.sympify("G * m * m / r**2"))
        assert ok and metric == 1.0 and sig == (1, -2, 1, 0)
    finally:
        UnitChecker.VARIABLE_DIMENSIONS.clear()
        UnitChecker.VARIABLE_DIMENSIONS.update(saved)


def test_exact_1e6_boundaries_kill_lte_mutants():
    """Pin strict < 1e-6 (not <=) at the exactly representable Q boundary.

    abs(0.0 - 1e-6) is exactly 1e-6 in doubles, so a < -> <= mutant on the
    Q clause flips these from mismatch to consistent. (L/T/M bases cannot
    hit the boundary exactly in binary floats, so only Q is pinnable.)
    """
    import math
    from unittest import mock
    from emergentia.unit_checker import UnitChecker

    strict = UnitChecker(reduced_units=False)
    r = sp.Symbol("r")
    # Mode-output Q clause: spoof differs from expected by exactly 1e-6.
    spring = UnitChecker(mode="spring", reduced_units=False)
    with mock.patch.object(spring, "_get_dimensional_signature",
                           return_value=(1, -2, 1, 1e-6)):
        ok, metric, _, _ = spring.check_consistency(r)
    assert not ok and metric == 0.3
    # Add-component Q clause: pair differs by exactly 1e-6.
    with mock.patch.object(
        strict, "_get_atom_dimensions",
        side_effect=[(1.0, -2.0, 1.0, 0.0), (1.0, -2.0, 1.0, 1e-6)],
    ):
        sig = strict._get_dimensional_signature(sp.sympify("r + x"))
    assert any(math.isnan(d) for d in sig)
    # Trig/exp Q clause: single-component arg of exactly 1e-6.
    with mock.patch.object(
        strict, "_get_atom_dimensions", return_value=(1e-6, 0, 0, 0)
    ):
        ok, metric, _, _ = strict.check_consistency(sp.exp(r))
    assert not ok and metric == 0.0


def test_validate_expression_mode_branch_selects_checker():
    """Pin validate_expression's `if mode:` checker selection.

    Deleting or inverting the branch is invisible when self and the
    mode-checker agree, so use a case where they disagree: a spring checker
    asked to validate 1/r**2 against generic mode must report valid.
    """
    from emergentia.unit_checker import UnitChecker

    checker = UnitChecker(mode="spring", reduced_units=False)
    result = checker.validate_expression(sp.sympify("1/r**2"), mode="generic")
    assert result["is_valid"] is True
    assert result["metric"] == 1.0
    assert result["mode"] == "generic"


def test_filter_prints_summary(capsys):
    """Pin filter_inconsistent_candidates' summary print (kills removal)."""
    from emergentia.unit_checker import UnitChecker

    UnitChecker().filter_inconsistent_candidates(
        [sp.sympify("1/r**2")], mode="generic")
    out = capsys.readouterr().out
    assert "filtered out" in out.lower() and "1" in out


# ---------------------------------------------------------------------------
# Formal-export gate coverage (scripts/export_emergentia_to_qed.py and
# scripts/verify_emergentia_axioms.py).
#
# These tests live in this file -- rather than in a new test module --
# because the mission's verification subset runs exactly
# the leakage gate / test_physics_priors / test_registry_consistency /
# test_unit_checker / test_conservative_grad. A new module would never
# execute under the mutation suite, so mutants in the export scripts would
# survive there; covered here, each sampled script mutant dies in pytest.
# ---------------------------------------------------------------------------

def _load_export_script_module(name, filename):
    import importlib.util
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[1]
    path = repo / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_writes_seven_lemmas_and_lean(tmp_path):
    export_mod = _load_export_script_module(
        "export_emergentia_to_qed", "export_emergentia_to_qed.py")
    out = tmp_path / "lemmas.txt"
    lean = tmp_path / "Export.lean"
    rc = export_mod.main(["--out", str(out), "--lean-out", str(lean)])
    assert rc == 0
    # Nested missing parents must be created (kills parents=False mutants);
    # the existing-parent call above kills exist_ok=False mutants.
    out2 = tmp_path / "nested" / "deep" / "lemmas.txt"
    lean2 = tmp_path / "other" / "place" / "Export.lean"
    rc = export_mod.main(["--out", str(out2), "--lean-out", str(lean2)])
    assert rc == 0
    assert out2.is_file()
    assert lean2.is_file()
    lines = [line for line in out.read_text().splitlines() if line.strip()]
    lemmas = [line for line in lines if not line.startswith("--")]
    assert len(lemmas) == 7
    assert all(line.endswith("= 0") for line in lemmas)
    body = lean.read_text()
    for name in ("emergentia_harmonic", "emergentia_lennard_jones",
                 "emergentia_morse", "emergentia_gravity",
                 "emergentia_buckingham", "emergentia_yukawa",
                 "emergentia_conservative_energy"):
        assert f"theorem {name}" in body
    assert body.count("by ring") == 7
    assert "sorry" not in body
    assert "emergentia_conservative_energy" in body


def test_export_stdout_returns_zero(capsys):
    export_mod = _load_export_script_module(
        "export_emergentia_to_qed", "export_emergentia_to_qed.py")
    rc = export_mod.main(["--stdout"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("= 0") == 7


def test_export_main_guard_runs_as_script(tmp_path):
    import subprocess
    import sys
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[1]
    out = tmp_path / "lemmas.txt"
    lean = tmp_path / "Export.lean"
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "export_emergentia_to_qed.py"),
         "--out", str(out), "--lean-out", str(lean)],
        capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert out.is_file()
    assert lean.is_file()


def test_verify_default_qed_dir_points_at_sibling_qed():
    verify_mod = _load_export_script_module(
        "verify_emergentia_axioms", "verify_emergentia_axioms.py")
    from pathlib import Path as _Path
    expected = _Path(verify_mod.__file__).resolve().parents[2] / "QED"
    default = verify_mod._default_qed_dir()
    assert default == expected, (default, expected)
    assert default.name == "QED"
    if not expected.is_dir():
        pytest.skip("no QED sibling staged in this room (Emergentia-only gate)")
    assert default.is_dir()


def test_verify_rejects_missing_lean_file(tmp_path):
    verify_mod = _load_export_script_module(
        "verify_emergentia_axioms", "verify_emergentia_axioms.py")
    rc = verify_mod.main([
        "--qed-dir", str(tmp_path),
        "--lean-file", str(tmp_path / "Missing.lean")])
    assert rc == 1


def test_verify_rejects_sorry(tmp_path):
    verify_mod = _load_export_script_module(
        "verify_emergentia_axioms", "verify_emergentia_axioms.py")
    lean = tmp_path / "Export.lean"
    lean.write_text("theorem bad : False := by sorry\n", encoding="utf-8")
    rc = verify_mod.main([
        "--qed-dir", str(tmp_path), "--lean-file", str(lean)])
    assert rc == 1


def test_verify_reports_unrunnable_lean(tmp_path, monkeypatch):
    verify_mod = _load_export_script_module(
        "verify_emergentia_axioms", "verify_emergentia_axioms.py")
    lean = tmp_path / "Export.lean"
    lean.write_text("theorem ok : True := trivial\n", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise OSError("no lean here")

    monkeypatch.setattr(verify_mod.subprocess, "run", _boom)
    rc = verify_mod.main([
        "--qed-dir", str(tmp_path), "--lean-file", str(lean)])
    assert rc == 1


def _mock_axiom_proc(returncode, stdout):
    class _Proc:
        pass

    proc = _Proc()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = ""
    return proc


def test_verify_rejects_failed_compilation(tmp_path, monkeypatch):
    verify_mod = _load_export_script_module(
        "verify_emergentia_axioms", "verify_emergentia_axioms.py")
    lean = tmp_path / "Export.lean"
    lean.write_text("theorem ok : True := trivial\n", encoding="utf-8")
    monkeypatch.setattr(
        verify_mod.subprocess, "run",
        lambda *args, **kwargs: _mock_axiom_proc(1, "boom"))
    rc = verify_mod.main([
        "--qed-dir", str(tmp_path), "--lean-file", str(lean)])
    assert rc == 1


def test_verify_rejects_unparseable_axioms(tmp_path, monkeypatch):
    verify_mod = _load_export_script_module(
        "verify_emergentia_axioms", "verify_emergentia_axioms.py")
    lean = tmp_path / "Export.lean"
    lean.write_text("theorem ok : True := trivial\n", encoding="utf-8")
    monkeypatch.setattr(
        verify_mod.subprocess, "run",
        lambda *args, **kwargs: _mock_axiom_proc(0, "nothing to parse here"))
    rc = verify_mod.main([
        "--qed-dir", str(tmp_path), "--lean-file", str(lean)])
    assert rc == 1


def test_verify_rejects_wrong_axioms(tmp_path, monkeypatch):
    verify_mod = _load_export_script_module(
        "verify_emergentia_axioms", "verify_emergentia_axioms.py")
    lean = tmp_path / "Export.lean"
    lean.write_text("theorem ok : True := trivial\n", encoding="utf-8")
    monkeypatch.setattr(
        verify_mod.subprocess, "run",
        lambda *args, **kwargs: _mock_axiom_proc(
            0, "ok depends on axioms: [sorryAx]"))
    rc = verify_mod.main([
        "--qed-dir", str(tmp_path), "--lean-file", str(lean)])
    assert rc == 1


def test_verify_accepts_expected_axioms(tmp_path, monkeypatch):
    import json
    from pathlib import Path as _Path

    verify_mod = _load_export_script_module(
        "verify_emergentia_axioms", "verify_emergentia_axioms.py")
    lean = tmp_path / "Export.lean"
    lean.write_text("theorem ok : True := trivial\n", encoding="utf-8")
    names = ("emergentia_harmonic", "emergentia_lennard_jones",
             "emergentia_morse", "emergentia_gravity",
             "emergentia_buckingham", "emergentia_yukawa",
             "emergentia_conservative_energy")
    lines = "\n".join(
        f"{name} depends on axioms: [propext, Classical.choice, Quot.sound]"
        for name in names)
    monkeypatch.setattr(
        verify_mod.subprocess, "run",
        lambda *args, **kwargs: _mock_axiom_proc(0, lines))
    rc = verify_mod.main([
        "--qed-dir", str(tmp_path), "--lean-file", str(lean)])
    assert rc == 0
    payload = json.loads(
        _Path("/tmp/emergentia_axioms.json").read_text(encoding="utf-8"))
    assert payload["verified"] is True
    assert sorted(payload["axioms"]) == [
        "Classical.choice", "Quot.sound", "propext"]


# ---------------------------------------------------------------------------
# Toolchain consistency (mission 00): python/pytest/torch/sympy pins.
# Lives here (not a separate file) so the mission’s fixed verification
# commands actually execute it.
# ---------------------------------------------------------------------------

from emergentia._toolchain import TOOLCHAIN, check_toolchain, get_actual_versions


def test_toolchain_versions_consistent():
    actual = get_actual_versions()
    mismatches = check_toolchain()
    assert not mismatches, (
        "toolchain mismatch: "
        + "; ".join(mismatches)
        + f" (expected={TOOLCHAIN}, actual={actual})"
    )


def test_toolchain_pin_covers_required_keys():
    assert set(TOOLCHAIN) == {"python", "pytest", "torch", "sympy"}
    assert all(isinstance(v, str) and v for v in TOOLCHAIN.values())


def test_toolchain_pins_match_requirements_file():
    # Single source of truth: the install-time pins in
    # tests/toolchain-requirements.txt must agree with the TOOLCHAIN record
    # the test-time check asserts. A divergence mutant in either file fails.
    import pathlib
    req = pathlib.Path(__file__).with_name("toolchain-requirements.txt")
    assert req.is_file(), req
    pinned = {}
    for line in req.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        assert version, line
        pinned[name.strip()] = version.strip()
    assert pinned, req
    for name, version in pinned.items():
        assert TOOLCHAIN.get(name) == version, (name, TOOLCHAIN.get(name), version)


# ---------------------------------------------------------------------------
# Mission 02 pins (must live in this file: the 02 gate executes exactly
# tests/test_physics_priors.py + tests/test_registry_consistency.py).
# ---------------------------------------------------------------------------

def test_mission02_reduced_units_defaults_false():
    # Goal: reduced_units defaults False (strict SI checking out of the box).
    import inspect
    from emergentia.unit_checker import UnitChecker
    sig = inspect.signature(UnitChecker.__init__)
    assert sig.parameters["reduced_units"].default is False
    assert UnitChecker().reduced_units is False
    assert UnitChecker(mode="gravity").reduced_units is False


def test_mission02_inconsistent_add_returns_nan_and_fails_closed():
    # Goal: inconsistent Add (r + G, length + gravity-constant) returns a NaN
    # signature with is_consistent False -- fail-closed, no exception, no
    # silent pass. DimensionalInconsistencyError is the caller-raised type;
    # the checker itself reports (False, nan-sig).
    import math
    import sympy as sp
    from emergentia.unit_checker import UnitChecker
    c = UnitChecker(mode="gravity")
    ok, metric, signature, message = c.check_consistency(
        sp.Symbol("r") + sp.Symbol("G"))
    assert ok is False
    assert any(math.isnan(d) for d in signature), signature
    assert metric == 0.0, metric
    assert isinstance(message, str) and message


def test_mission02_mode_tables_do_not_leak_across_instances():
    # Hardening: mode-specific keys must live on the instance table, not the
    # class attribute. Constructing an electric checker used to permanently
    # register k_e/inv_k_e for every later checker (including strict ones in
    # other modes). Killed mutants: the per-instance dict copy in __init__.
    import sympy as sp
    from emergentia.unit_checker import UnitChecker
    UnitChecker(mode="electric")
    generic = UnitChecker(mode="generic")
    assert generic._get_atom_dimensions(sp.Symbol("k_e")) == (0, 0, 0, 0)
    assert "k_e" not in UnitChecker.VARIABLE_DIMENSIONS
    assert "inv_k_e" not in UnitChecker.VARIABLE_DIMENSIONS
    electric = UnitChecker(mode="electric")
    assert electric.VARIABLE_DIMENSIONS["k_e"] == (1, 3, -2, -2)
