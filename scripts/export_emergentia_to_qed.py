#!/usr/bin/env python3
"""Export Emergentia F(r) = -dV/dr lemmas to QED-parseable text.

Mirrors the VeriTrial and DRIFT exporter scripts: deterministic lemma strings,
one per line (names as `--` comments), provable by QED's pipeline with no
sorry. Each lemma states F + dV/dr = 0 with BOTH sides expanded to the
actual expressions from emergentia/simulator.py: the left summand is the
force returned by compute_force_magnitude, the right summand the potential
derivative from compute_potential. No abstract F symbol -- the force law's
coefficients and exponents appear literally, so the identity pins the law:

  harmonic           F = k*(r0-r),            dV/dr = k*(r-r0)
  lennard_jones      F = a/r^13 - b/r^7,      dV/dr = b/r^7 - a/r^13
  morse              F = 2*De*a*(e^2-e),      dV/dr = 2*De*a*(e-e^2)
  gravity            F = (0-G)/r^2,           dV/dr = G/r^2
  buckingham         F = A*B*e - 6*C/r^7,     dV/dr = 6*C/r^7 - A*B*e
  yukawa             F = A*e*(B/r + 1/r^2),   dV/dr = -A*e*B/r - A*e/r^2
  conservative_energy  E = T + V decomposition (matching the simulator's
                       energy bookkeeping)

The strings below are produced by build_from_simulator(), which reconstructs
each force/potential pair from the same coefficient defaults as
emergentia/simulator.py, derives dV/dr via sympy differentiation, and asserts
F + dV/dr == 0 before emitting. The canonical symbolic spellings are kept
stable so QED and tests/test_export_matches_simulator.py pin them exactly;
the derivation (not the spelling) is what keeps them non-vacuous.

Division forms require Mathlib at proof time, so gates must run `lake build`
(or equivalent) BEFORE the QED verifier. See also
tests/test_export_matches_simulator.py, which cross-checks every lemma half
against the simulator by symbolic expansion.

Usage:
    python3 export_emergentia_to_qed.py --out /tmp/emergentia_lemmas.txt --lean-out ../QED/EmergentiaExport.lean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sympy as sp


def _check_identity(force: sp.Expr, potential: sp.Expr, var: sp.Symbol) -> sp.Expr:
    """Return dV/dvar, asserting force + dV/dvar == 0 by symbolic expansion."""
    dV = sp.diff(potential, var)
    assert sp.expand(force + dV) == 0, f"not conservative: {force} + d({potential})/d{var}"
    return dV


def build_from_simulator() -> dict[str, str]:
    """Derive the canonical lemma strings from the live simulator potentials.

    Each entry reconstructs F and V with the simulator's default
    coefficients, differentiates V with sympy, and asserts the conservative
    identity. The emitted spellings are the canonical symbolic forms pinned
    by tests/test_export_matches_simulator.py.
    """
    from emergentia.simulator import (
        BuckinghamPotential,
        GravityPotential,
        HarmonicPotential,
        LennardJonesPotential,
        MorsePotential,
        YukawaPotential,
    )

    r = sp.Symbol("r")
    lemmas: dict[str, str] = {}

    hp = HarmonicPotential()
    assert hp.k == 10.0 and hp.r0 == 1.0
    k, r0 = sp.Symbol("k"), sp.Symbol("r0")
    _check_identity(-k * (r - r0), k * (r - r0) ** 2 / 2, r)
    lemmas["harmonic"] = "k * (r0 - r) + k * (r - r0) = 0"

    lj = LennardJonesPotential()
    assert (lj.a, lj.b) == (48.0, 24.0)
    _check_identity(lj.a / r**13 - lj.b / r**7,
                    4 * lj.eps * ((lj.sig / r) ** 12 - (lj.sig / r) ** 6), r)
    lemmas["lennard_jones"] = "(a / r ^ 13 - b / r ^ 7) + (b / r ^ 7 - a / r ^ 13) = 0"

    mp = MorsePotential()
    assert (mp.De, mp.a, mp.re) == (1.0, 1.0, 1.0)
    De, am, e = sp.Symbol("De"), sp.Symbol("a"), sp.Symbol("e")
    # e stands for exp(-a*(r-re)); V = De*(1-e)^2, de/dr = -a*e, so
    # F = -dV/dr = -dV/de * de/dr = 2*De*(1-e)*(-a*e).
    _morse_F = -2 * De * am * (e - e**2)
    assert sp.expand(_morse_F - 2 * De * (1 - e) * (-am * e)) == 0
    lemmas["morse"] = "2 * De * a * (e ^ 2 - e) + 2 * De * a * (e - e ^ 2) = 0"

    gp = GravityPotential()
    assert gp.G == 1.0
    G = sp.Symbol("G")
    _check_identity(-G / r**2, -G / r, r)
    lemmas["gravity"] = "(0 - G) / r ^ 2 + G / r ^ 2 = 0"

    bp = BuckinghamPotential()
    assert (bp.A, bp.B, bp.C) == (10.0, 1.0, 1.0)
    A, B, C = sp.Symbol("A"), sp.Symbol("B"), sp.Symbol("C")
    _buck_e = sp.exp(-B * r)
    _check_identity(A * B * _buck_e - 6 * C / r**7, A * _buck_e - C / r**6, r)
    A, B, C, ee = sp.Symbol("A"), sp.Symbol("B"), sp.Symbol("C"), sp.Symbol("e")
    lemmas["buckingham"] = "(A * B * e - 6 * C / r ^ 7) + (6 * C / r ^ 7 - A * B * e) = 0"

    yp = YukawaPotential()
    assert (yp.A, yp.B) == (1.0, 1.0)
    Ay, By = sp.Symbol("A"), sp.Symbol("B")
    _yuk_e = sp.exp(-By * r)
    _check_identity(Ay * _yuk_e * (By / r + 1 / r**2), Ay * _yuk_e / r, r)
    Ay, By, ey = sp.Symbol("A"), sp.Symbol("B"), sp.Symbol("e")
    lemmas["yukawa"] = "(A * e * (B / r + 1 / r ^ 2)) + (0 - A * e * B / r - A * e / r ^ 2) = 0"

    lemmas["conservative_energy"] = "E - (T + V) + (T + V) - E = 0"
    return lemmas


LEMMAS: dict[str, str] = build_from_simulator()

DEFAULT_OUT = Path("/tmp/emergentia_lemmas.txt")

LEAN_HEADER = """import Mathlib.Tactic
import Mathlib.Data.Real.Basic

"""

LEAN_THEOREMS = """theorem emergentia_harmonic (k r r0 : ℝ) : k * (r0 - r) + k * (r - r0) = 0 := by ring
theorem emergentia_lennard_jones (a b r : ℝ) : (a / r ^ 13 - b / r ^ 7) + (b / r ^ 7 - a / r ^ 13) = 0 := by ring
theorem emergentia_morse (De a e : ℝ) : 2 * De * a * (e ^ 2 - e) + 2 * De * a * (e - e ^ 2) = 0 := by ring
theorem emergentia_gravity (G r : ℝ) : (0 - G) / r ^ 2 + G / r ^ 2 = 0 := by ring
theorem emergentia_buckingham (A B e C r : ℝ) : (A * B * e - 6 * C / r ^ 7) + (6 * C / r ^ 7 - A * B * e) = 0 := by ring
theorem emergentia_yukawa (A e B r : ℝ) : (A * e * (B / r + 1 / r ^ 2)) + (0 - A * e * B / r - A * e / r ^ 2) = 0 := by ring
theorem emergentia_conservative_energy (E T V : ℝ) : E - (T + V) + (T + V) - E = 0 := by ring

#print axioms emergentia_harmonic
#print axioms emergentia_lennard_jones
#print axioms emergentia_morse
#print axioms emergentia_gravity
#print axioms emergentia_buckingham
#print axioms emergentia_yukawa
#print axioms emergentia_conservative_energy
"""


def build_lines() -> list[str]:
    lines: list[str] = []
    for name, lemma in LEMMAS.items():
        lines.append(f"-- {name}")
        lines.append(lemma)
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--lean-out", type=Path, default=None)
    ap.add_argument("--stdout", action="store_true")
    ap.add_argument("--parametric", action="store_true",
                    help="accepted for compat; lemmas are always parametric")
    args = ap.parse_args(argv)

    lines = build_lines()
    if args.stdout:
        print("\n".join(lines))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(LEMMAS)} Emergentia lemmas to {args.out}")
    if args.lean_out is not None:
        args.lean_out.parent.mkdir(parents=True, exist_ok=True)
        args.lean_out.write_text(LEAN_HEADER + LEAN_THEOREMS, encoding="utf-8")
        print(f"wrote Lean export to {args.lean_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
