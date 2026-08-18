import pytest
import torch
import numpy as np
from emergentia.simulator import (
    PhysicsSim,
    HarmonicPotential,
    GravityPotential,
    LennardJonesPotential,
)
from emergentia.engine import DiscoveryPipeline


@pytest.mark.parametrize("mode,potential_cls,kwargs", [
    ("spring", HarmonicPotential, {}),
    ("gravity", GravityPotential, {}),
    ("lj", LennardJonesPotential, {}),
])
def test_pipeline_recovers_law(mode, potential_cls, kwargs):
    """Full pipeline should achieve R² > 0.90 on clean data."""
    torch.manual_seed(42)
    np.random.seed(42)

    pot = potential_cls(**kwargs)
    sim = PhysicsSim(n=3, dim=2, potential=pot, seed=42)
    pipeline = DiscoveryPipeline(mode=mode, potential=pot, seed=42)
    result = pipeline.run(sim, nn_epochs=1000, noise_std=0.0)

    assert result["r2"] > 0.90, f"{mode}: R²={result['r2']}"
    assert result["formula"] is not None
    assert "X" not in result["formula"], (
        f"Formula should not contain unmapped feature indices: {result['formula']}"
    )


def test_regression_gravity_formula():
    """Regression test: gravity discovery should find 1/r^2 dependence."""
    torch.manual_seed(42)
    np.random.seed(42)

    pot = GravityPotential(G=1.0)
    sim = PhysicsSim(n=3, dim=2, potential=pot, seed=42)
    pipeline = DiscoveryPipeline(mode="gravity", potential=pot, seed=42)
    result = pipeline.run(sim, nn_epochs=1000, noise_std=0.0)

    assert result["r2"] > 0.90
    # The discovered formula should involve 1/r**2 (gravity's characteristic term)
    formula = result["formula"]
    assert (
        "1/r**2" in formula
        or "r**(-2)" in formula
        or "/r**2" in formula
        or "inv" in formula.lower()
    ), (
        f"Gravity formula should contain 1/r^2: {formula}"
    )


def test_regression_spring_formula():
    """Regression test: spring discovery should find linear r dependence."""
    torch.manual_seed(42)
    np.random.seed(42)

    pot = HarmonicPotential(k=10.0, r0=1.0)
    sim = PhysicsSim(n=3, dim=2, potential=pot, seed=42)
    pipeline = DiscoveryPipeline(mode="spring", potential=pot, seed=42)
    result = pipeline.run(sim, nn_epochs=1000, noise_std=0.0)

    assert result["r2"] > 0.90
    # The discovered formula should contain r or (r - constant)
    formula = result["formula"]
    assert "r" in formula, f"Spring formula should contain r: {formula}"


def test_pipeline_with_noise_01():
    """Pipeline should achieve R² > 0.85 with noise_std=0.01."""
    torch.manual_seed(42)
    np.random.seed(42)

    pot = GravityPotential(G=1.0)
    sim = PhysicsSim(n=3, dim=2, potential=pot, seed=42)
    pipeline = DiscoveryPipeline(mode="gravity", potential=pot, seed=42)
    result = pipeline.run(sim, nn_epochs=1000, noise_std=0.01)

    assert result["r2"] > 0.85, f"Expected R² > 0.85 with noise, got {result['r2']}"


def test_pipeline_results_structure():
    """Verify the result dictionary has all expected fields."""
    torch.manual_seed(42)
    np.random.seed(42)

    pot = GravityPotential(G=1.0)
    sim = PhysicsSim(n=3, dim=2, potential=pot, seed=42)
    pipeline = DiscoveryPipeline(mode="gravity", potential=pot, seed=42)
    result = pipeline.run(sim, nn_epochs=500, noise_std=0.0)

    expected_keys = {
        "mode", "formula", "mse", "r2", "bic", "success",
        "test_r2", "test_mse", "bandwidth", "noise_std",
    }
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
