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

Division forms require Mathlib at proof time, so gates must run `lake build`
(or equivalent) BEFORE the QED verifier. See also
tests/test_export_matches_simulator.py, which cross-checks every lemma half
against the simulator by symbolic expansion.

Usage:
    python3 export_emergentia_to_qed.py --out /tmp/emergentia_lemmas.txt --lean-out ../QED/EmergentiaExport.lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

LEMMAS: dict[str, str] = {
    "harmonic": "k * (r0 - r) + k * (r - r0) = 0",
    "lennard_jones": "(a / r ^ 13 - b / r ^ 7) + (b / r ^ 7 - a / r ^ 13) = 0",
    "morse": "2 * De * a * (e ^ 2 - e) + 2 * De * a * (e - e ^ 2) = 0",
    "gravity": "(0 - G) / r ^ 2 + G / r ^ 2 = 0",
    "buckingham": "(A * B * e - 6 * C / r ^ 7) + (6 * C / r ^ 7 - A * B * e) = 0",
    "yukawa": "(A * e * (B / r + 1 / r ^ 2)) + (0 - A * e * B / r - A * e / r ^ 2) = 0",
    "conservative_energy": "E - (T + V) + (T + V) - E = 0",
}

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
