"""Pinned toolchain versions for consistent runs.

Check only; no behavior change. No new dependencies or network calls.
"""

TOOLCHAIN = {
    "python": "3.13.13",
    "pytest": "8.4.2",
    "torch": "2.10.0",
    "sympy": "1.14.0",
}


def get_actual_versions():
    """Return actual {python, pytest, torch, sympy} versions without network calls."""
    import sys
    from importlib import metadata

    try:
        pytest_version = metadata.version("pytest")
    except Exception:
        import pytest

        pytest_version = pytest.__version__

    import sympy
    import torch

    return {
        "python": sys.version.split()[0],
        "pytest": pytest_version,
        "torch": torch.__version__,
        "sympy": sympy.__version__,
    }


def check_toolchain(expected=None):
    """Return list of mismatch messages; empty list means consistent."""
    expected = expected or TOOLCHAIN
    actual = get_actual_versions()
    mismatches = []
    for key in ("python", "pytest", "torch", "sympy"):
        if actual.get(key) != expected.get(key):
            mismatches.append(
                f"{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
            )
    return mismatches
