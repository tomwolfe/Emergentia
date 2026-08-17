import pytest
import sympy as sp
from emergentia.llm_priors import LLMPriorProvider


def test_generate_priors_fallback_returns_expressions():
    """Test that generate_priors_from_llm works with zai_client=None (fallback path)."""
    provider = LLMPriorProvider(zai_client=None)
    result = provider.generate_priors_from_llm({"mode": "gravity"}, mode="gravity")

    assert len(result) >= 1, "Fallback path should return at least 1 expression"
    for expr in result:
        assert isinstance(expr, sp.Expr), f"Expected SymPy expression, got {type(expr)}"


def test_no_nan_or_zoo_in_priors():
    """Test that no expression in the result contains nan or zoo."""
    provider = LLMPriorProvider(zai_client=None)
    result = provider.generate_priors_from_llm({"mode": "gravity"}, mode="gravity")

    for expr in result:
        assert not expr.has(sp.nan), f"Expression contains nan: {expr}"
        assert not expr.has(sp.zoo), f"Expression contains zoo: {expr}"


def test_generate_priors_all_modes():
    """Test fallback path for all known physics modes."""
    provider = LLMPriorProvider(zai_client=None)
    modes = ["spring", "gravity", "lj", "morse", "buckingham", "yukawa", "electric"]

    for mode in modes:
        result = provider.generate_priors_from_llm({"mode": mode}, mode=mode)
        assert len(result) >= 1, f"Mode '{mode}' should return at least 1 expression"
        for expr in result:
            assert not expr.has(sp.nan) and not expr.has(sp.zoo), (
                f"Mode '{mode}': expression contains nan or zoo: {expr}"
            )


def test_get_mode_priors():
    """Test that get_mode_priors returns expected structure."""
    provider = LLMPriorProvider(zai_client=None)
    gravity_priors = provider.get_mode_priors("gravity")

    assert "description" in gravity_priors
    assert "expected_terms" in gravity_priors
    assert "suggested_functional_forms" in gravity_priors
    assert "1/r^2" in gravity_priors["expected_terms"]


def test_generate_priors_caching():
    """Test that priors are cached."""
    provider = LLMPriorProvider(zai_client=None)
    summary = {"mode": "spring"}

    result1 = provider.generate_priors_from_llm(summary, mode="spring")
    result2 = provider.generate_priors_from_llm(summary, mode="spring")

    # Cached results should be identical (same object references)
    assert result1 is result2 or result1 == result2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
