import sympy as sp
import json
import numpy as np
from typing import List, Dict, Any
from emergentia.unit_checker import UnitChecker, is_dimensionally_consistent


class LLMPriorProvider:
    """
    Provides prior knowledge from GLM-4.7-flash to guide symbolic regression.
    Uses the Z.AI SDK to analyze dataset metadata and suggest functional forms.
    """

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

    SYMPY_COMPATIBLE_OPERATORS = {
        "add": lambda x, y: x + y,
        "sub": lambda x, y: x - y,
        "mul": lambda x, y: x * y,
        "div": lambda x, y: x / y,
        "inv": lambda x: 1 / x,
        "power": lambda x, y: x**y,
        "exp": lambda x: sp.exp(x),
        "log": lambda x: sp.log(x),
        "sqrt": lambda x: sp.sqrt(x),
    }

    def __init__(self, zai_client=None, model="glm-4.7-flash", max_candidates=10):
        """
        Initialize the LLM Prior Provider.

        Args:
            zai_client: Optional Z.AI client instance
            model: The model to use (default: 'glm-4.7-flash')
            max_candidates: Maximum number of functional forms to return
        """
        self.model = model
        self.max_candidates = max_candidates
        self.zai_client = zai_client
        self.candidates_cache = {}

        if zai_client is None:
            try:
                from zai import ZaiClient

                self.zai_client = ZaiClient()
                print(f"Initialized Z.AI client with model: {model}")
            except ImportError:
                print(
                    "Warning: zai-sdk not installed. LLM priors will use default physics-based suggestions."
                )
                self.zai_client = None

        self.unit_checker = UnitChecker()

    def get_mode_priors(self, mode: str) -> Dict[str, Any]:
        """
        Get prior knowledge for a specific physics mode.

        Args:
            mode: The physics mode (gravity, lj, morse, etc.)

        Returns:
            Dictionary containing mode-specific priors
        """
        if mode in self.PHYSICS_MODES:
            return self.PHYSICS_MODES[mode]
        else:
            return {
                "description": f"Generic force mode {mode}",
                "expected_terms": ["1/r", "1/r^2", "r", "constant"],
                "suggested_functional_forms": [
                    "1/r",
                    "1/r^2",
                    "r",
                    "1/r + 1/r^2",
                    "1/r * r",
                    "r * 1/r^2",
                ],
            }

    def _parse_sym_expression(self, expression_str: str) -> sp.Expr:
        """
        Parse a string expression into a SymPy expression.

        Args:
            expression_str: String representation of the expression

        Returns:
            SymPy expression
        """
        try:
            expr = sp.sympify(expression_str)
            return expr
        except Exception as e:
            print(f"Error parsing expression '{expression_str}': {e}")
            return None

    def to_gplearn_program(self, expr: sp.Expr, feature_names: List[str]) -> str:
        """
        Convert a SymPy expression into a gplearn-compatible program string.

        Args:
            expr: A SymPy expression
            feature_names: List of feature names (e.g., ['r', '1/r', 'r^2', ...])

        Returns:
            A gplearn-compatible program string (e.g., 'add(mul(X0, X1), X2)')
        """
        # Map feature indices to symbols
        # feature_names = ['r', '1/r', 'r^2', '1/r^2', 'exp(-r)', 'log(r+1)']
        # X0 -> r, X1 -> 1/r, etc.
        sym_map = {}
        for i, name in enumerate(feature_names):
            sym_map[i] = sp.Symbol(f"X{i}")

        def _convert_sym(sym):
            name = str(sym).lower()
            # Check if it's one of our feature names
            for i, fn in enumerate(feature_names):
                if fn.lower() == name:
                    return sp.Symbol(f"X{i}")
            # Check if it's a known variable like r, m, etc.
            if name in ['r', 'r^2', '1/r', 'exp(-r)', 'log(r+1)']:
                for i, fn in enumerate(feature_names):
                    if fn.lower() == name:
                        return sp.Symbol(f"X{i}")
            # Fall back to the symbol itself
            return sym

        def _expr_to_program(expr) -> str:
            """Recursively convert a SymPy expression to a gplearn program string."""
            if expr.is_Number:
                return str(expr)
            elif expr.is_Symbol:
                sym_idx = None
                name = str(expr).lower()
                for i, fn in enumerate(feature_names):
                    if fn.lower() == name:
                        sym_idx = i
                        break
                if sym_idx is not None:
                    return f"X{sym_idx}"
                return f"X0"
            elif expr.func.__name__ == 'Add':
                args = [_expr_to_program(a) for a in expr.args]
                return f"add({', '.join(args)})"
            elif expr.func.__name__ == 'Mul':
                args = [_expr_to_program(a) for a in expr.args]
                return f"mul({', '.join(args)})"
            elif expr.func.__name__ == 'Sub':
                args = [_expr_to_program(a) for a in expr.args]
                return f"sub({', '.join(args)})"
            elif expr.func.__name__ == 'Pow':
                base = _expr_to_program(expr.args[0])
                exp = _expr_to_program(expr.args[1]) if len(expr.args) > 1 else "1"
                return f"power({base}, {exp})"
            elif expr.func.__name__ == 'Div':
                base = _expr_to_program(expr.args[0])
                exp = _expr_to_program(expr.args[1]) if len(expr.args) > 1 else "1"
                return f"div({base}, {exp})"
            elif expr.func.__name__ == 'Exp':
                return f"exp({_expr_to_program(expr.args[0])})"
            elif expr.func.__name__ == 'Log':
                return f"log({_expr_to_program(expr.args[0])})"
            elif expr.func.__name__ == 'Sqrt':
                return f"sqrt({_expr_to_program(expr.args[0])})"
            else:
                return f"X0"

        program_str = _expr_to_program(expr)
        return program_str

    def _generate_symmetric_forms(self, base_expr: sp.Expr) -> List[sp.Expr]:
        """
        Generate symmetric variations of an expression by swapping terms.

        Args:
            base_expr: Base SymPy expression

        Returns:
            List of symmetric variations
        """
        if not isinstance(base_expr, sp.Expr):
            return []

        variations = []
        atoms = list(base_expr.atoms())

        if len(atoms) <= 1:
            return []

        # Generate variations by swapping pairs of terms
        from itertools import permutations

        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                var = base_expr
                # Swap terms (simplified)
                try:
                    var = var.subs(atoms[j], atoms[i]).subs(atoms[i], atoms[j])
                    if var != base_expr:
                        variations.append(var)
                except Exception:
                    pass

        return variations

    def _apply_operators(
        self, base_expr: sp.Expr, num_variations: int = 5
    ) -> List[sp.Expr]:
        """
        Apply various mathematical operators to an expression.

        Args:
            base_expr: Base SymPy expression
            num_variations: Number of operator applications

        Returns:
            List of variations with operators applied
        """
        variations = []
        operators = ["add", "sub", "mul", "div", "inv", "power", "exp", "log", "sqrt"]

        for _ in range(num_variations):
            op = np.random.choice(operators)

            if op == "add" and base_expr != 0:
                variations.append(base_expr + 1)
            elif op == "sub" and base_expr != 0:
                variations.append(base_expr - 1)
            elif op == "mul" and base_expr != 0:
                variations.append(base_expr * 2)
            elif op == "div" and base_expr != 1:
                variations.append(base_expr / 2)
            elif op == "inv" and base_expr != 0:
                variations.append(1 / base_expr)
            elif op == "exp" and base_expr != 0:
                variations.append(sp.exp(base_expr))
            elif op == "log" and base_expr > 0:
                variations.append(sp.log(base_expr))
            elif op == "sqrt" and base_expr >= 0:
                variations.append(sp.sqrt(base_expr))
            elif op == "power" and base_expr > 0:
                variations.append(base_expr**2)

        return variations

    def _generate_sym_expression_from_llm(
        self, llm_response: str, mode: str
    ) -> List[sp.Expr]:
        """
        Parse LLM response and generate SymPy expressions.

        Args:
            llm_response: LLM response text
            mode: Physics mode for dimensionality check

        Returns:
            List of valid SymPy expressions
        """
        try:
            llm_response = llm_response.strip()

            # Handle different response formats
            if "```" in llm_response:
                # Extract code blocks
                import re

                code_blocks = re.findall(
                    r"```python\s*(.*?)\s*```", llm_response, re.DOTALL
                )
                if code_blocks:
                    llm_response = code_blocks[0]

            # Try to parse as a list of expressions
            if "," in llm_response:
                expr_strings = [s.strip() for s in llm_response.split(",")]
            else:
                expr_strings = [llm_response.strip()]

            expressions = []
            valid_count = 0

            for expr_str in expr_strings[: self.max_candidates]:
                expr = self._parse_sym_expression(expr_str)
                if expr is not None:
                    is_valid, metric, signature, message = self.unit_checker.check_consistency(expr)
                    if is_valid:
                        expressions.append(expr)
                        valid_count += 1

            if valid_count == 0:
                # If no valid expressions, return empty list
                return []

            return expressions

        except Exception as e:
            print(f"Error processing LLM response: {e}")
            return []

    def _generate_hybrid_priors(self, mode: str) -> List[sp.Expr]:
        """
        Generate physics-based priors as a fallback.

        Args:
            mode: Physics mode

        Returns:
            List of SymPy expressions
        """
        mode_info = self.get_mode_priors(mode)
        expressions = []

        for expr_str in mode_info.get("suggested_functional_forms", [])[
            : self.max_candidates
        ]:
            expr = self._parse_sym_expression(expr_str)
            if expr is not None:
                expressions.append(expr)

        return expressions

    def generate_priors_from_llm(
        self, dataset_summary: Dict[str, Any], mode: str = None
    ) -> List[sp.Expr]:
        """
        Generate prior expressions from LLM based on dataset metadata.

        Args:
            dataset_summary: Dictionary containing dataset information
                - min_force: Minimum force magnitude
                - max_force: Maximum force magnitude
                - periodicity: Periodic behavior (if any)
                - decay_rate: Decay rate for exponential behavior
                - observed_terms: Known terms in the force
            mode: Optional physics mode

        Returns:
            List of SymPy-compatible prior expressions
        """
        if mode is None:
            mode = dataset_summary.get("mode", "generic")

        # Check cache first
        cache_key = f"{mode}_{json.dumps(dataset_summary)}"
        if cache_key in self.candidates_cache:
            return self.candidates_cache[cache_key]

        print(f"Generating LLM priors for mode: {mode}")

        if self.zai_client is None:
            print("No Z.AI client available, using hybrid priors")
            candidates = self._generate_hybrid_priors(mode)
            self.candidates_cache[cache_key] = candidates
            return candidates

        try:
            mode_description = self.get_mode_priors(mode)["description"]

            # Build prompt
            prompt = f"""
You are an expert in physics-based symbolic regression. Based on the following dataset summary, suggest {self.max_candidates} plausible functional forms for the force function F(r) in the form of SymPy expressions:

Dataset Summary:
- Mode: {mode_description}
- Min Force: {dataset_summary.get("min_force", "N/A")}
- Max Force: {dataset_summary.get("max_force", "N/A")}
- Periodicity: {dataset_summary.get("periodicity", "None")}
- Decay Rate: {dataset_summary.get("decay_rate", "None")}
- Observed Terms: {dataset_summary.get("observed_terms", "N/A")}
- Noise Level: {dataset_summary.get("noise_level", "N/A")}

Please suggest {self.max_candidates} SymPy-compatible functional forms. Format them as a comma-separated list like:
'1/r^2, 1/r, r^2, 1/r^2 + 1/r, exp(-r)/r'

Focus on physically plausible forms that match the expected behavior for force functions (dimensionally consistent, decaying/increasing appropriately).
"""

            print("Calling GLM-4.7-flash with prompt...")

            response = self.zai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a physics-aware symbolic regression assistant. Generate SymPy-compatible mathematical expressions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1000,
                temperature=0.3,
            )

            llm_text = response.choices[0].message.content
            print(f"LLM Response: {llm_text}")

            # Generate expressions from LLM response
            candidates = self._generate_sym_expression_from_llm(llm_text, mode)

            if len(candidates) == 0:
                print("LLM didn't provide valid expressions, using hybrid priors")
                candidates = self._generate_hybrid_priors(mode)

            # Ensure we have exactly max_candidates
            if len(candidates) < self.max_candidates:
                hybrid_candidates = self._generate_hybrid_priors(mode)
                for expr in hybrid_candidates:
                    if expr not in candidates:
                        candidates.append(expr)
                        if len(candidates) >= self.max_candidates:
                            break

            # Store in cache
            self.candidates_cache[cache_key] = candidates

            # Validate and log the priors
            print(f"\nGenerated {len(candidates)} prior expressions:")
            for i, expr in enumerate(candidates[:5]):
                try:
                    is_valid, metric, signature, message = (
                        self.unit_checker.check_consistency(expr)
                    )
                    print(
                        f"  {i + 1}. {expr} - Valid: {is_valid}, Metric: {metric:.2f}"
                    )
                except Exception as e:
                    print(f"  {i + 1}. {expr} - Error: {e}")

            return candidates[: self.max_candidates]

        except Exception as e:
            print(f"Error generating priors from LLM: {e}")
            print("Falling back to hybrid priors")
            candidates = self._generate_hybrid_priors(mode)
            self.candidates_cache[cache_key] = candidates
            return candidates

    def get_prior_equivalents(self, expression: sp.Expr) -> List[sp.Expr]:
        """
        Generate equivalent expressions from the prior set.

        Args:
            expression: Base SymPy expression

        Returns:
            List of equivalent variations
        """
        variations = []

        # Apply symmetry transformations
        symmetric_variants = self._generate_symmetric_forms(expression)
        variations.extend(symmetric_variants)

        # Apply operator variations
        operator_variants = self._apply_operators(expression, num_variations=3)
        variations.extend(operator_variants)

        return variations

    def validate_prior(self, expression: sp.Expr, mode: str = None) -> Dict[str, Any]:
        """
        Validate a prior expression using dimensional analysis.

        Args:
            expression: SymPy expression to validate
            mode: Physics mode for validation

        Returns:
            Validation dictionary
        """
        if mode is None:
            mode = "generic"

        return self.unit_checker.validate_expression(expression, mode)

    def clear_cache(self):
        """Clear the candidate cache."""
        self.candidates_cache = {}
        print("LLM prior cache cleared")


class ZaiClient:
    """
    Wrapper for the Z.AI SDK client for accessing GLM-4.7-flash.
    This is verified in the engine.py integration.
    """

    def __init__(self, api_key: str = None, model: str = "glm-4.7-flash"):
        """
        Initialize the Z.AI client.

        Args:
            api_key: Optional API key
            model: Model to use (default: 'glm-4.7-flash')
        """
        try:
            from zai import ZaiClient as SDKClient

            if api_key:
                self.client = SDKClient(api_key=api_key)
            else:
                self.client = SDKClient()

            self.model = model
            print(f"Initialized Z.AI client with model: {model}")

        except ImportError:
            raise ImportError(
                "zai-sdk is not installed. Install it with: pip install zai-sdk==0.1.0"
            )

    def chat(self):
        """
        Get the chat completions interface.

        Returns:
            Chat completions interface
        """
        return self.client.chat.completions

    def get_version(self):
        """Get the Z.AI SDK version."""
        try:
            import zai

            return zai.__version__
        except ImportError:
            return "unknown"
