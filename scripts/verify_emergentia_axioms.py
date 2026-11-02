#!/usr/bin/env python3
"""Certify the axiom set of the seven exported Emergentia lemmas.

Mirrors DRIFT/scripts/verify_drift_axioms.py: compiles the Lean export
with the pinned toolchain, parses every `#print axioms` line and requires
exactly [propext, Classical.choice, Quot.sound]. Writes
/tmp/emergentia_axioms.json for the quint evidence collector.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

EXPECTED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}

NAMES = (
    "emergentia_harmonic",
    "emergentia_lennard_jones",
    "emergentia_morse",
    "emergentia_gravity",
    "emergentia_buckingham",
    "emergentia_yukawa",
    "emergentia_conservative_energy",
)


def _lake_lean_cmd(qed: Path) -> list[str]:
    return ["lake", "env", "lean"]


def _default_qed_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "QED"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qed-dir", type=Path,
                    default=_default_qed_dir())
    ap.add_argument("--lean-file", type=Path, default=None,
                    help="Lean export to check (default: <qed-dir>/EmergentiaExport.lean)")
    args = ap.parse_args(argv)
    qed = args.qed_dir.resolve()
    lean_file = args.lean_file.resolve() if args.lean_file else qed / "EmergentiaExport.lean"
    if not lean_file.is_file():
        print(f"Emergentia Lean export not found at {lean_file}", file=sys.stderr)
        return 1
    body = lean_file.read_text(encoding="utf-8")
    if "sorry" in body:
        print("EMERGENTIA axiom check: sorry in Lean export", file=sys.stderr)
        return 1
    tmp = qed / "_emergentia_axiom_check.lean"
    tmp.write_text(body, encoding="utf-8")
    try:
        proc = subprocess.run(
            [*_lake_lean_cmd(qed), str(tmp)], cwd=str(qed), capture_output=True,
            text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"EMERGENTIA axiom check failed to run: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print("EMERGENTIA axiom check: Lean compilation failed", file=sys.stderr)
        print(output[-2000:], file=sys.stderr)
        return 1
    groups = re.findall(r"depends on axioms:?\s*\[(.*?)\]", output)
    if not groups:
        print("EMERGENTIA axiom check: could not parse #print axioms", file=sys.stderr)
        return 1
    axioms = sorted({a.strip().strip("'") for grp in groups
                     for a in grp.split(",") if a.strip()})
    if set(axioms) != EXPECTED_AXIOMS:
        print(f"EMERGENTIA axiom check: unexpected axioms {axioms}", file=sys.stderr)
        return 1
    payload = {"lemmas": list(NAMES), "axioms": axioms, "verified": True}
    Path("/tmp/emergentia_axioms.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"EMERGENTIA axioms certified: {axioms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
