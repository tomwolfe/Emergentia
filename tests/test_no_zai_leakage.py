"""Zero-leakage gate: no ZAI/GLM network references anywhere in the package."""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
PAT = re.compile(r"zai|glm-4|ZAI_API|ZaiClient|LLMPriorProvider|enable_llm_priors|llm_priors", re.IGNORECASE)
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tether", ".venv", "results"}
SKIP_FILES = {"test_no_zai_leakage.py"}


def _scan_targets():
    for p in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix in {".py", ".md"} and p.name not in SKIP_FILES:
            yield p


def test_no_zai_leakage_in_source():
    offenders = []
    for p in _scan_targets():
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if PAT.search(line):
                offenders.append(f"{p.relative_to(ROOT)}:{i}: {line.strip()}")
    assert not offenders, "ZAI/LLM leakage found:\n" + "\n".join(offenders)


def test_no_zai_sdk_import():
    import emergentia.physics_priors as pp
    import inspect
    src = inspect.getsource(pp)
    assert "zai" not in src.lower(), "physics_priors must not reference zai"
    assert not hasattr(pp, "ZaiClient"), "ZaiClient must not exist"
