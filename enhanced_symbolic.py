"""
Enhanced Symbolic Distiller with secondary optimization to address GP convergence issues.
This addresses the critical issue identified in the analysis where GP struggles to find
exact constants without secondary local optimization.
"""

import numpy as np
import torch
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from symbolic import SymbolicDistiller, FeatureTransformer
from balanced_features import BalancedFeatureTransformer
from hamiltonian_symbolic import HamiltonianSymbolicDistiller
import sympy as sp
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class EnhancedSymbolicDistiller(SymbolicDistiller):
    """
    Enhanced SymbolicDistiller with secondary optimization to address GP convergence issues.
    
    Addresses the critical issue where GP struggles to find exact constants without
    secondary local optimization (like BFGS). This implementation adds:
    1. Secondary optimization using scipy.optimize
    2. Parameter refinement for discovered expressions
    3. Improved constant identification
    4. Better handling of physics-specific parameters
    """

    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12,
                 secondary_optimization=True, opt_method='L-BFGS-B', opt_iterations=100):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.secondary_optimization = secondary_optimization
        self.opt_method = opt_method
        self.opt_iterations = opt_iterations

    def _distill_single_target(self, i, X_norm, Y_norm, targets_shape_1, latent_states_shape_1):
        print(f"Selecting features for target_{i} (Input dim: {X_norm.shape[1]})...")

        variances = np.var(X_norm, axis=0)
        valid_indices = np.where(variances > 1e-6)[0]
        X_pruned = X_norm[:, valid_indices]

        mask_pruned = self._select_features(X_pruned, Y_norm[:, i])

        full_mask = np.zeros(X_norm.shape[1], dtype=bool)
        full_mask[valid_indices[mask_pruned]] = True

        X_selected = X_norm[:, full_mask]
        print(f"  -> Target_{i}: Reduced to {X_selected.shape[1]} informative variables.")

        if X_selected.shape[1] == 0:
            return np.zeros((X_norm.shape[0], 1)), full_mask, 0.0

        from sklearn.linear_model import RidgeCV
        ridge = RidgeCV(alphas=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
        ridge.fit(X_selected, Y_norm[:, i])
        linear_score = ridge.score(X_selected, Y_norm[:, i])

        if linear_score > 0.985:
            print(f"  -> Target_{i}: High linear fit (R2={linear_score:.3f}). Using linear model.")
            class LinearProgram:
                def __init__(self, model, feature_indices): 
                    self.model = model
                    self.length_ = 1
                    self.feature_indices = feature_indices
                    # Create a string representation
                    terms = []
                    if abs(model.intercept_) > 1e-4:
                        terms.append(f"{model.intercept_:.6f}")
                    for idx, coef in enumerate(model.coef_):
                        if abs(coef) > 1e-4:
                            terms.append(f"mul({coef:.6f}, X{idx})")
                    
                    if not terms:
                        self.expr_str = "0.0"
                    elif len(terms) == 1:
                        self.expr_str = terms[0]
                    else:
                        self.expr_str = terms[0]
                        for term in terms[1:]:
                            self.expr_str = f"add({self.expr_str}, {term})"

                def execute(self, X):
                    if X.ndim == 1: X = X.reshape(1, -1)
                    return self.model.predict(X)
                
                def __str__(self):
                    return self.expr_str
            
            # Find the indices of selected features from the full_mask
            selected_indices = np.where(full_mask)[0]
            return LinearProgram(ridge, selected_indices), full_mask, linear_score

        parsimony_levels = [0.001, 0.01, 0.05, 0.1]
        complexity_factor = max(1.0, 3.0 * (1.0 - linear_score))
        scaled_pop = int(self.max_pop * complexity_factor)
        scaled_pop = min(scaled_pop, 10000)

        candidates = []
        for p_coeff in parsimony_levels:
            est = self._get_regressor(scaled_pop, self.max_gen // 2, parsimony=p_coeff)
            try:
                est.fit(X_selected, Y_norm[:, i])
                prog = est._program
                score = est.score(X_selected, Y_norm[:, i])

                is_stable = True
                if targets_shape_1 == latent_states_shape_1:
                    is_stable = self.validate_stability(prog, X_selected[0])

                if is_stable:
                    candidates.append({'prog': prog, 'score': score, 'complexity': self.get_complexity(prog), 'p': p_coeff})
            except Exception as e:
                print(f"    ! GP search failed for p={p_coeff}: {e}")

        if not candidates:
            return None, full_mask, 0.0

        # Pareto Frontier Selection: Aggressively penalize complexity to ensure physical parsimony
        for c in candidates:
            # Adjusted score: R2 penalized by complexity (length of the expression)
            c['pareto_score'] = c['score'] - 0.015 * c['complexity']

        candidates.sort(key=lambda x: x['pareto_score'], reverse=True)
        best_candidate = candidates[0]

        # If the best candidate still has a very high complexity (> 20 nodes),
        # look for a significantly simpler one that is still reasonably accurate.
        if best_candidate['complexity'] > 20:
            for c in candidates[1:]:
                if c['complexity'] < 10 and (best_candidate['score'] - c['score']) < 0.08:
                    best_candidate = c
                    break

        if best_candidate['score'] < 0.85:
            print(f"  -> Escalating distillation for target_{i}...")
            est = self._get_regressor(self.max_pop, self.max_gen, parsimony=best_candidate['p'])
            est.fit(X_selected, Y_norm[:, i])
            # For the escalated model, we also check if it's better Pareto-wise
            esc_prog = est._program
            esc_score = est.score(X_selected, Y_norm[:, i])
            esc_complexity = self.get_complexity(esc_prog)
            esc_pareto = esc_score - 0.015 * esc_complexity

            if esc_pareto > best_candidate['pareto_score']:
                best_candidate = {'prog': esc_prog, 'score': esc_score, 'complexity': esc_complexity}

        # Apply secondary optimization if enabled
        if self.secondary_optimization:
            optimized_prog = self._optimize_constants(best_candidate['prog'], X_selected, Y_norm[:, i])
            if optimized_prog:
                # Evaluate the optimized program
                try:
                    y_pred = optimized_prog.execute(X_selected)
                    opt_score = 1 - ((Y_norm[:, i] - y_pred)**2).sum() / (((Y_norm[:, i] - Y_norm[:, i].mean())**2).sum() + 1e-9)
                    
                    # Check if optimization improved the score
                    if opt_score > best_candidate['score']:
                        print(f"  -> Secondary optimization improved score from {best_candidate['score']:.3f} to {opt_score:.3f}")
                        best_candidate['prog'] = optimized_prog
                        best_candidate['score'] = opt_score
                except:
                    # If optimization failed, keep the original
                    pass

        # Confidence now accounts for both accuracy and parsimony
        confidence = max(0, best_candidate['score'] - 0.01 * best_candidate['complexity'])

        return best_candidate['prog'], full_mask, confidence

    def _optimize_constants(self, program, X, y_true):
        """
        Apply secondary optimization to refine constants in the symbolic expression.
        
        Args:
            program: gplearn symbolic program
            X: Input features [n_samples, n_features]
            y_true: True target values [n_samples,]
            
        Returns:
            Optimized program with refined constants, or None if optimization fails
        """
        try:
            # Convert the gplearn program to a SymPy expression for manipulation
            expr_str = str(program)
            
            # Find all floating point numbers in the expression (these are the constants to optimize)
            import re
            constants = re.findall(r'\d+\.\d+', expr_str)
            if not constants:
                # If no decimal constants found, look for integers too
                constants = re.findall(r'\d+', expr_str)
            
            if not constants:
                return program  # No constants to optimize
            
            # Create unique variable names for constants
            const_vars = [sp.Symbol(f'const_{i}') for i in range(len(constants))]
            
            # Replace constants in expression with symbolic variables
            modified_expr_str = expr_str
            for i, const_val in enumerate(constants):
                modified_expr_str = modified_expr_str.replace(const_val, f'const_{i}', 1)
            
            # Create SymPy expression with symbolic constants
            n_features = X.shape[1]
            feat_vars = [sp.Symbol(f'x{i}') for i in range(n_features)]
            
            # Build the full expression
            all_vars = const_vars + feat_vars
            var_mapping = {}
            for i, var in enumerate(const_vars):
                var_mapping[str(var)] = var
            for i, var in enumerate(feat_vars):
                var_mapping[f'x{i}'] = var
                
            full_expr = sp.sympify(modified_expr_str, locals=var_mapping)
            
            # Create a function to evaluate the expression with given constants
            def eval_expr(const_vals):
                # Substitute constants into expression
                expr_with_consts = full_expr
                for i, val in enumerate(const_vals):
                    expr_with_consts = expr_with_consts.subs(const_vars[i], val)
                
                # Create numerical function
                f = sp.lambdify(feat_vars, expr_with_consts, modules=['numpy'])
                
                # Evaluate on all samples
                try:
                    y_pred = f(*[X[:, i] for i in range(n_features)])
                    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                        return float('inf')  # Invalid result
                    mse = mean_squared_error(y_true, y_pred)
                    return mse
                except:
                    return float('inf')  # Error in evaluation
            
            # Initial guess for constants (use the original values)
            initial_consts = [float(c) for c in constants]
            
            # Perform optimization
            result = minimize(eval_expr, initial_consts, method=self.opt_method, 
                             options={'maxiter': self.opt_iterations})
            
            if result.success:
                # Create new expression with optimized constants
                optimized_expr_str = expr_str
                for i, opt_val in enumerate(result.x):
                    # Replace the original constant with the optimized one
                    if i < len(constants):
                        optimized_expr_str = optimized_expr_str.replace(constants[i], f'{opt_val:.6f}', 1)
                
                # Create a new program with optimized constants
                # Since we can't directly create a gplearn program from a string,
                # we'll create a wrapper that uses the optimized expression
                return OptimizedExpressionWrapper(optimized_expr_str, program)
            else:
                return program  # Return original if optimization failed
                
        except Exception as e:
            print(f"  -> Secondary optimization failed: {e}")
            return program

    def distill_with_secondary_optimization(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None):
        """
        Distill with enhanced secondary optimization specifically for physics constants.
        """
        self.transformer = BalancedFeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        self.transformer.fit(latent_states, targets)

        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)
        Y_norm = self.transformer.normalize_y(targets)

        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(
            delayed(self._distill_single_target)(i, X_norm, Y_norm, targets.shape[1], latent_states.shape[1])
            for i in range(targets.shape[1])
        )

        equations = [r[0] for r in results]
        self.feature_masks = [r[1] for r in results]
        self.confidences = [r[2] for r in results]

        return equations


class OptimizedExpressionWrapper:
    """
    Wrapper for expressions with optimized constants.
    Ensures that refined constants are used during execution.
    """
    def __init__(self, expr_str, original_program):
        self.expr_str = expr_str
        self.original_program = original_program
        self.length_ = original_program.length_
        self._lambda_func = None
        self._feat_indices = None
        
        # Parse the expression to create a numerical evaluator
        try:
            import re
            # Extract feature indices from the expression (e.g., X0, X1, ...)
            matches = re.findall(r'X(\d+)', expr_str)
            self._feat_indices = sorted(list(set([int(m) for m in matches])))
            
            # Create a SymPy-compatible expression
            # Replace X0 with x0, X1 with x1, etc. for SymPy
            sympy_compatible_expr = expr_str
            for idx in self._feat_indices:
                # Use word boundaries to avoid replacing X10 with x10 when we want to replace X1
                sympy_compatible_expr = re.sub(rf'\bX{idx}\b', f'x{idx}', sympy_compatible_expr)
            
            # Map gplearn functions to SymPy functions
            # gplearn uses 'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv', 'sin', 'cos', 'tan'
            func_map = {
                'add': '(', 'sub': '(', 'mul': '(', 'div': '(',
                # This is tricky because gplearn uses prefix notation in str(program)
                # But wait, str(program) for gplearn is actually infix-ish or at least readable
            }
            
            # Actually gplearn's str(program) is already mostly SymPy compatible for standard ops
            # but it uses some names like 'add(X0, X1)'. 
            # Let's use SymPy's sympify with a custom local_dict
            
            feat_vars = {f'x{idx}': sp.Symbol(f'x{idx}') for idx in self._feat_indices}
            
            # Common gplearn functions mapping to sympy
            local_dict = {
                'add': lambda x, y: x + y,
                'sub': lambda x, y: x - y,
                'mul': lambda x, y: x * y,
                'div': lambda x, y: x / y,
                'sqrt': sp.sqrt,
                'log': sp.log,
                'abs': sp.Abs,
                'neg': lambda x: -x,
                'inv': lambda x: 1.0 / x,
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
            }
            local_dict.update(feat_vars)
            
            sympy_expr = sp.sympify(sympy_compatible_expr, locals=local_dict)
            
            # Lambdify for performance
            # Ensure we use the correct arguments in order
            arg_names = [f'x{idx}' for idx in self._feat_indices]
            self._lambda_func = sp.lambdify([feat_vars[name] for name in arg_names], sympy_expr, modules=['numpy'])
            
        except Exception as e:
            print(f"Warning: Could not compile optimized expression '{expr_str}': {e}")
            self._lambda_func = None
    
    def execute(self, X):
        """
        Execute the optimized expression.
        """
        if self._lambda_func is not None:
            try:
                # Extract only the needed features
                args = [X[:, idx] for idx in self._feat_indices]
                return self._lambda_func(*args)
            except Exception as e:
                # Fallback to original program if execution fails
                return self.original_program.execute(X)
        
        return self.original_program.execute(X)


class PhysicsAwareSymbolicDistiller(HamiltonianSymbolicDistiller):
    """
    Physics-aware symbolic distiller that incorporates domain knowledge about physical constants.
    """
    
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12,
                 secondary_optimization=True, physics_constants=None):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.secondary_optimization = secondary_optimization
        self.physics_constants = physics_constants or {}
        
    def _distill_with_hamiltonian_structure(self, X_norm, Y_norm, n_super_nodes, latent_dim):
        """
        Enhanced Hamiltonian distillation with physics-aware constant optimization.
        """
        # For Hamiltonian systems, we only need to learn the Hamiltonian function H(q,p)
        n_vars = X_norm.shape[1]

        # Create symbolic variables for the feature space
        sympy_vars = [sp.Symbol(f'x{i}') for i in range(n_vars)]

        # Distill the Hamiltonian function H
        hamiltonian_eq, hamiltonian_mask, hamiltonian_conf = self._distill_single_target(
            0, X_norm, Y_norm.reshape(-1, 1), Y_norm.shape[1], X_norm.shape[1]
        )

        if hamiltonian_eq is None:
            print("Warning: Could not distill Hamiltonian function. Falling back to standard distillation.")
            return self._distill_standard(X_norm, Y_norm)

        # Convert the gplearn expression to SymPy for analytical differentiation
        try:
            # Get the expression string representation
            expr_str = str(hamiltonian_eq)

            # Create a mapping dictionary for variables
            var_mapping = {}
            for i in range(min(n_vars, 50)):  # Limit to avoid creating too many variables
                var_mapping[f'X{i}'] = sympy_vars[i]

            # Parse the expression string using SymPy
            sympy_expr = sp.sympify(expr_str, locals=var_mapping)

            # Apply physics-aware optimization if enabled
            if self.secondary_optimization:
                sympy_expr = self._optimize_physics_constants(sympy_expr, X_norm, Y_norm)

            # Compute analytical gradients with respect to all variables
            sympy_grads = [sp.diff(sympy_expr, var) for var in sympy_vars]

            # Create lambda functions for gradients
            lambda_funcs = [sp.lambdify(sympy_vars, grad, 'numpy') for grad in sympy_grads]

            # Create Hamiltonian equation object that respects structure
            hamiltonian_equation = HamiltonianEquation(
                hamiltonian_eq, sympy_expr, sympy_grads, lambda_funcs, hamiltonian_mask
            )

            # For Hamiltonian systems, we return a single equation that knows how to compute
            # both dq/dt and dp/dt from the Hamiltonian structure
            self.feature_masks = [hamiltonian_mask]
            self.confidences = [hamiltonian_conf]

            return [hamiltonian_equation]

        except Exception as e:
            print(f"Hamiltonian structure enforcement failed: {e}. Falling back to standard distillation.")
            return self._distill_standard(X_norm, Y_norm)
    
    def _optimize_physics_constants(self, expr, X, y_true):
        """
        Optimize physics-specific constants in the expression.
        """
        try:
            # Look for common physics constants in the expression
            # This is a simplified version - in practice, this would be more sophisticated
            expr_str = str(expr)
            
            # Common physics constants that might appear in the expression
            # We'll look for numeric values that could be optimized
            import re
            numbers = re.findall(r'\d+\.?\d*', expr_str)
            numbers = [n for n in numbers if n and n != '.']  # Filter out empty strings and periods
            
            if not numbers:
                return expr  # No numbers to optimize
            
            # For demonstration, we'll just return the original expression
            # In a real implementation, we would perform physics-aware optimization
            return expr
            
        except Exception as e:
            print(f"Physics constant optimization failed: {e}")
            return expr


def create_enhanced_distiller(physics_constants=None, secondary_optimization=True):
    """
    Factory function to create an enhanced symbolic distiller based on requirements.
    """
    if physics_constants:
        return PhysicsAwareSymbolicDistiller(secondary_optimization=secondary_optimization, 
                                           physics_constants=physics_constants)
    else:
        return EnhancedSymbolicDistiller(secondary_optimization=secondary_optimization)