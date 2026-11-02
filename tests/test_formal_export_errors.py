"""Error-path and CLI-branch coverage for the formal-export scripts.

Genuine gap-closing tests: the green ladder only exercises the success path
of scripts/export_emergentia_to_qed.py and scripts/verify_emergentia_axioms.py,
so mutants in their error branches (missing parent dirs, sorry rejection,
toolchain failures, unparseable/mismatched axioms) would survive. These tests
pin those branches with a stubbed ``lake`` (fast, hermetic, no network).
"""
import os
import stat
import subprocess
import sys
from pathlib import Path

ROOM = Path(__file__).resolve().parents[1]
EXPORT = ROOM / "scripts" / "export_emergentia_to_qed.py"
AXIOMS = ROOM / "scripts" / "verify_emergentia_axioms.py"

STANDARD_AXIOM_LINES = [
    "emergentia_harmonic depends on axioms: [propext, Classical.choice, Quot.sound]",
    "emergentia_lennard_jones depends on axioms: [propext, Classical.choice, Quot.sound]",
    "emergentia_morse depends on axioms: [propext, Classical.choice, Quot.sound]",
    "emergentia_gravity depends on axioms: [propext, Classical.choice, Quot.sound]",
    "emergentia_buckingham depends on axioms: [propext, Classical.choice, Quot.sound]",
    "emergentia_yukawa depends on axioms: [propext, Classical.choice, Quot.sound]",
    "emergentia_conservative_energy depends on axioms: [propext, Classical.choice, Quot.sound]",
]


def _write_stub_lake(bin_dir: Path, output: str) -> None:
    """Install a stub ``lake`` that prints canned ``#print axioms`` output."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    stub = bin_dir / "lake"
    stub.write_text(
        "#!/bin/sh\nprintf '%s' \"$FAKE_LAKE_OUTPUT\"\n", encoding="utf-8"
    )
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    assert stub.is_file()


def _run_export(*args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXPORT), *args],
        cwd=str(ROOM), capture_output=True, text=True, timeout=120, env=env,
    )


def _run_axioms(*args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(AXIOMS), *args],
        cwd=str(ROOM), capture_output=True, text=True, timeout=300, env=env,
    )


def test_export_stdout_lists_seven_lemmas():
    """The --stdout branch prints all seven force-balance lemmas."""
    proc = _run_export("--stdout")
    assert proc.returncode == 0, proc.stderr
    lemmas = [l for l in proc.stdout.splitlines()
              if l.strip() and not l.strip().startswith("--")]
    assert len(lemmas) == 7, proc.stdout
    assert all(l.strip().endswith("= 0") for l in lemmas), proc.stdout
    assert "conservative_energy" in proc.stdout


def test_export_creates_nested_lean_out_dirs(tmp_path):
    """Nested --lean-out parents are created (parents=True is load-bearing)."""
    nested = tmp_path / "a" / "b" / "c"
    assert not nested.exists()
    proc = _run_export(
        "--out", str(tmp_path / "lemmas.txt"),
        "--lean-out", str(nested / "EmergentiaExport.lean"),
    )
    assert proc.returncode == 0, proc.stderr
    assert (nested / "EmergentiaExport.lean").is_file()
    assert "emergentia_conservative_energy" in (
        nested / "EmergentiaExport.lean"
    ).read_text(encoding="utf-8")


def test_axioms_rejects_sorry_lean_file(tmp_path):
    """A Lean file containing ``sorry`` is rejected before compilation."""
    lean = tmp_path / "Sorry.lean"
    lean.write_text(
        "import Mathlib.Tactic\n"
        "theorem sorry_present : True := sorry\n"
        "#print axioms sorry_present\n",
        encoding="utf-8",
    )
    proc = _run_axioms("--lean-file", str(lean), "--qed-dir", str(tmp_path))
    assert proc.returncode == 1, "axioms script must reject a Lean file containing sorry"
    assert "sorry" in proc.stderr


def test_axioms_reports_toolchain_failure(tmp_path):
    """A missing ``lake`` binary fails cleanly with no traceback."""
    lean = tmp_path / "Clean.lean"
    lean.write_text(
        "import Mathlib.Tactic\n"
        "theorem clean_true : True := trivial\n"
        "#print axioms clean_true\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PATH"] = str(tmp_path)
    proc = _run_axioms("--lean-file", str(lean), "--qed-dir", str(tmp_path), env=env)
    assert proc.returncode == 1, "axioms script must report a missing lake toolchain without traceback"
    assert "failed to run" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_axioms_rejects_empty_axiom_output(tmp_path):
    """Unparseable ``#print axioms`` output fails with a parse error."""
    _write_stub_lake(tmp_path / "bin", "")
    lean = tmp_path / "Clean.lean"
    lean.write_text(
        "import Mathlib.Tactic\n"
        "theorem clean_true : True := trivial\n"
        "#print axioms clean_true\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PATH"] = str(tmp_path / "bin") + os.pathsep + env["PATH"]
    env["FAKE_LAKE_OUTPUT"] = ""
    proc = _run_axioms("--lean-file", str(lean), "--qed-dir", str(tmp_path), env=env)
    assert proc.returncode == 1, "axioms script must report unparseable axiom output"
    assert "could not parse" in proc.stderr
    assert "unexpected axioms" not in proc.stderr


def test_axioms_rejects_wrong_axiom_set(tmp_path):
    """An axiom set other than [propext, Classical.choice, Quot.sound] fails."""
    _write_stub_lake(tmp_path / "bin", "")
    lean = tmp_path / "Clean.lean"
    lean.write_text(
        "import Mathlib.Tactic\n"
        "theorem clean_true : True := trivial\n"
        "#print axioms clean_true\n",
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PATH"] = str(tmp_path / "bin") + os.pathsep + env["PATH"]
    env["FAKE_LAKE_OUTPUT"] = "clean_true depends on axioms: [propext]\n"
    proc = _run_axioms("--lean-file", str(lean), "--qed-dir", str(tmp_path), env=env)
    assert proc.returncode == 1, "axioms script must reject a non-standard axiom set"
    assert "unexpected axioms" in proc.stderr
