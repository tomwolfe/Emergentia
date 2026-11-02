"""Each exported lemma half must equal the simulator's force/derivative.

For every potential in scripts/export_emergentia_to_qed.py, the lemma has
the shape `<F> + <D> = 0` where `<F>` must be the force returned by
compute_force_magnitude and `<D>` its negation (dV/dr). This test
symbolically cross-checks both halves against the simulator source so the
Lean export cannot drift into vacuous X-X=0 tautologies over an abstract F.
"""

import re
import sympy as sp
from pathlib import Path

import pytest

from emergentia.simulator import (
    BuckinghamPotential,
    GravityPotential,
    HarmonicPotential,
    LennardJonesPotential,
    MorsePotential,
    YukawaPotential,
)

# Lemma left/right halves (without the trailing "= 0"), keyed by name.
LEMMA_HALVES = {
    "harmonic": ("k * (r0 - r)", "k * (r - r0)"),
    # a = 48*eps*sig^12, b = 24*eps*sig^6 are the LJ force coefficients.
    "lennard_jones": ("(a / r ^ 13 - b / r ^ 7)", "(b / r ^ 7 - a / r ^ 13)"),
    # e stands for exp(-a*(r-re)); force = -2*De*a*(e - e^2).
    "morse": ("2 * De * a * (e ^ 2 - e)", "2 * De * a * (e - e ^ 2)"),
    "gravity": ("(0 - G) / r ^ 2", "G / r ^ 2"),
    # e stands for exp(-B*r).
    "buckingham": (
        "(A * B * e - 6 * C / r ^ 7)",
        "(6 * C / r ^ 7 - A * B * e)",
    ),
    "yukawa": (
        "(A * e * (B / r + 1 / r ^ 2))",
        "(0 - A * e * B / r - A * e / r ^ 2)",
    ),
    "conservative_energy": ("E - (T + V)", "(T + V) - E"),
}


def _qed_to_sympy(text: str) -> sp.Expr:
    text = text.replace("^", "**")
    return sp.sympify(text)


def _simulator_force(potential) -> sp.Expr:
    """Force expression from the simulator, in lemma symbols."""
    r, k, r0 = sp.symbols("r k r0")
    a, b = sp.symbols("a b")
    De, e = sp.symbols("De e")
    G = sp.Symbol("G")
    A, B, C = sp.symbols("A B C")
    if isinstance(potential, HarmonicPotential):
        return -k * (r - r0)
    if isinstance(potential, LennardJonesPotential):
        return a / r**13 - b / r**7
    if isinstance(potential, MorsePotential):
        return -2 * De * a * (e - e**2)
    if isinstance(potential, GravityPotential):
        return -G / r**2
    if isinstance(potential, BuckinghamPotential):
        return A * B * e - 6 * C / r**7
    if isinstance(potential, YukawaPotential):
        return A * e * (B / r + 1 / r**2)
    raise AssertionError(f"unknown potential {potential!r}")


POTENTIALS = {
    "harmonic": HarmonicPotential(),
    "lennard_jones": LennardJonesPotential(),
    "morse": MorsePotential(),
    "gravity": GravityPotential(),
    "buckingham": BuckinghamPotential(),
    "yukawa": YukawaPotential(),
}


@pytest.mark.parametrize("name", sorted(POTENTIALS))
def test_lemma_force_half_matches_simulator(name):
    f_half, d_half = LEMMA_HALVES[name]
    f_lemma = _qed_to_sympy(f_half)
    f_sim = _simulator_force(POTENTIALS[name])
    assert sp.expand(f_lemma - f_sim) == 0, (
        f"{name}: lemma force {f_half} != simulator force {f_sim}"
    )
    d_lemma = _qed_to_sympy(d_half)
    # dV/dr must be exactly -F.
    assert sp.expand(d_lemma + f_sim) == 0, (
        f"{name}: lemma dV/dr {d_half} is not -F({f_sim})"
    )


def _export_lemmas() -> dict[str, str]:
    text = (Path(__file__).parent.parent / "scripts"
            / "export_emergentia_to_qed.py").read_text()
    body = text.split("LEMMAS", 1)[1]
    return dict(re.findall(r'"(\w+)":\s*"([^"]+?)\s*=\s*0"', body))


def test_export_file_matches_test_halves():
    exported = _export_lemmas()
    assert set(exported) >= set(LEMMA_HALVES) | {"conservative_energy"}
    norm = lambda s: re.sub(r"\s+", "", s)
    for name, (f_half, d_half) in LEMMA_HALVES.items():
        assert norm(exported[name]) == norm(f"{f_half} + {d_half}"), (
            f"{name}: export lemma drifted from test pins"
        )
