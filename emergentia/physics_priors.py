"""Pure local physics priors. No network, no client, no API keys."""
import sympy as sp
from typing import List, Dict, Any

class PhysicsPriorProvider:
    PHYSICS_MODES = {
        "gravity": {
            "description": "Gravitational force between particles",
            "expected_terms": ["1/r^2", "1/r", "r^2", "constant"],
            "suggested_functional_forms": [
                "1/r^2",
                "1/r",
                "r^2",
                "1/r^2 + 1/r",
                "1/r^2 - 1/r",
                "1/r^2 * constant",
                "r^2 / constant",
                "1/r^2 / r",
            ],
        },
        "lj": {
            "description": "Lennard-Jones potential derivative",
            "expected_terms": ["1/r^7", "1/r^13", "1/r", "exp(r)", "exp(-r)"],
            "suggested_functional_forms": [
                "1/r^7",
                "1/r^13",
                "1/r^7 - 1/r^13",
                "1/r^7 * (1/r^13)",
                "1/r^7 - 1/r^13",
                "exp(-r)/r^7",
                "exp(-r)/r^13",
            ],
        },
        "morse": {
            "description": "Morse potential derivative",
            "expected_terms": ["exp(-r)", "exp(2r)", "r", "constant"],
            "suggested_functional_forms": [
                "exp(-r)",
                "exp(-2r)",
                "exp(-r) * r",
                "exp(-r) - 1",
                "exp(-r) + exp(-2r)",
                "r * exp(-r)",
            ],
        },
        "spring": {
            "description": "Harmonic spring force",
            "expected_terms": ["r", "constant", "r^2"],
            "suggested_functional_forms": [
                "r",
                "r^2",
                "r * constant",
                "constant - r",
                "r * (constant - r)",
            ],
        },
        "buckingham": {
            "description": "Buckingham potential derivative",
            "expected_terms": ["1/r^7", "exp(-r)", "r"],
            "suggested_functional_forms": [
                "1/r^7",
                "1/r^7 * exp(-r)",
                "exp(-r)/r^7",
                "1/r^7 + exp(-r)",
            ],
        },
        "yukawa": {
            "description": "Yukawa potential derivative",
            "expected_terms": ["exp(-r)/r", "1/r", "1/r^2"],
            "suggested_functional_forms": [
                "exp(-r)/r",
                "exp(-r)/r^2",
                "exp(-r)/r * 1/r",
                "exp(-r)/r + 1/r^2",
            ],
        },
        "electric": {
            "description": "Coulomb-like electrostatic force",
            "expected_terms": ["1/r^2", "1/r", "charge * constant", "charge^2"],
            "suggested_functional_forms": [
                "1/r^2",
                "1/r^2 * charge",
                "charge^2 / r^2",
                "charge * 1/r^2",
            ],
        },
        "mixed": {
            "description": "Composite potential (combination of forces)",
            "expected_terms": ["combination", "sum", "subtraction", "weighted"],
            "suggested_functional_forms": [
                "combination of terms",
                "weighted sum",
                "superposition",
            ],
        },
    }

    def __init__(self, max_candidates: int = 10):
        self.max_candidates = max_candidates
        self.candidates_cache = {}

    def get_mode_priors(self, mode: str) -> Dict[str, Any]:
        if mode in self.PHYSICS_MODES:
            return self.PHYSICS_MODES[mode]
        return {
            "description": f"Generic force mode {mode}",
            "expected_terms": ["1/r", "1/r^2", "r", "constant"],
            "suggested_functional_forms": ["1/r", "1/r^2", "r", "1/r + 1/r^2", "1/r * r", "r * 1/r^2"],
        }

    def _parse(self, s: str):
        try:
            return sp.sympify(s)
        except Exception:
            return None

    def generate_priors(self, mode: str, max_candidates: int = None) -> List[sp.Expr]:
        n = max_candidates and self.max_candidates
        info = self.get_mode_priors(mode)
        out = []
        for s in info.get("suggested_functional_forms", [])[:n]:
            e = self._parse(s)
            if e is not None:
                out.append(e)
        return out

    # Back-compat alias used by engine/tests
    def generate_priors_from_llm(self, dataset_summary=None, mode: str = "generic"):
        m = mode or (dataset_summary or {}).get("mode", "generic")
        return self.generate_priors(m)

    def to_gplearn_program(self, expr: sp.Expr, feature_names) -> str:
        def _c(e):
            if e.is_Number:
                return str(float(e))
            if e.is_Symbol:
                n = str(e).lower()
                for i, fn in enumerate(feature_names):
                    if fn.lower() == n:
                        return f"X{i}"
                return "X0"
            fn = e.func.__name__
            if fn == 'Add':
                r = _c(e.args[0])
                for a in e.args[1:]:
                    r = f"add({r}, {_c(a)})"
                return r
            if fn == 'Mul':
                r = _c(e.args[0])
                for a in e.args[1:]:
                    r = f"mul({r}, {_c(a)})"
                return r
            if fn == 'Pow':
                return f"power({_c(e.args[0])}, {_c(e.args[1])})"
            ln = fn.lower()
            if ln in ('exp','log','sqrt'):
                return f"{ln}({_c(e.args[0])})"
            return "X0"
        return _c(expr)
