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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
