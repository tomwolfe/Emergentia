"""Mixed Harmonic+Gravity potential must clear the R^2 > 0.85 discovery bar.

Regression test for the known mixed-potential gap (was 0.8364 at
noise_std=0.01, below the 0.85 bar). Uses the conservative-field path with
two-stage training.
"""

import torch

from emergentia.engine import DiscoveryPipeline
from emergentia.simulator import CompositePotential, GravityPotential, HarmonicPotential, PhysicsSim


def test_mixed_r2_085():
    device = torch.device("cpu")
    mixed_potential = CompositePotential([HarmonicPotential(), GravityPotential()])
    sim = PhysicsSim(n=3, dim=2, potential=mixed_potential, seed=42, device=device)
    pipe = DiscoveryPipeline(
        mode="mixed",
        potential=mixed_potential,
        device=device,
        seed=42,
        use_conservative_field=True,
    )
    result = pipe.run(sim, nn_epochs=1000, noise_std=0.01, sr_generations=10, sr_population=500)
    assert result["r2"] > 0.85, f"Expected mixed R^2 > 0.85, got {result['r2']}"
    print(f"Mixed R^2 = {result['r2']:.4f}")
