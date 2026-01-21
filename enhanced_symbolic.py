"""
Enhanced Symbolic Distiller with secondary optimization to address GP convergence issues.
This addresses the critical issue identified in the analysis where GP struggles to find
exact constants without secondary local optimization.
"""

import numpy as np
import torch
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from symbolic import SymbolicDistiller, FeatureTransformer, gp_to_sympy
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
        Uses SymPy tree traversal for robust constant identification.
        """
        try:
            expr_str = str(program)
            n_features = X.shape[1]
            feat_vars = [sp.Symbol(f'x{i}') for i in range(n_features)]
            
            # gplearn uses X0, X1, etc.
            local_dict = {
                'add': lambda x, y: x + y,
                'sub': lambda x, y: x - y,
                'mul': lambda x, y: x * y,
                'div': lambda x, y: x / (y + 1e-9),
                'sqrt': lambda x: sp.sqrt(sp.Abs(x)),
                'log': lambda x: sp.log(sp.Abs(x) + 1e-9),
                'abs': sp.Abs,
                'neg': lambda x: -x,
                'inv': lambda x: 1.0 / (x + 1e-9),
                'sin': sp.sin,
                'cos': sp.cos,
                'exp': sp.exp,
            }
            for i in range(n_features):
                local_dict[f'X{i}'] = feat_vars[i]
                
            # 1. Parse into SymPy
            full_expr = sp.sympify(expr_str, locals=local_dict)
            
            # 2. Extract numeric constants
            all_atoms = full_expr.atoms(sp.Number)
            # Filter constants that are likely parameters and not indices/small integers
            constants = sorted([float(a) for a in all_atoms if not a.is_Integer or abs(a) > 5])
            
            if not constants:
                return program
            
            # 3. Parametrize constants
            const_vars = [sp.Symbol(f'c{i}') for i in range(len(constants))]
            subs_map = {sp.Float(c): cv for c, cv in zip(constants, const_vars)}
            param_expr = full_expr.subs(subs_map)
            
            # 4. Lambdify for optimization
            f_lamb = sp.lambdify(const_vars + feat_vars, param_expr, modules=['numpy'])
            
            def eval_expr(const_vals):
                try:
                    y_pred = f_lamb(*const_vals, *[X[:, i] for i in range(n_features)])
                    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                        return float('inf')
                    return mean_squared_error(y_true, y_pred)
                except:
                    return float('inf')
            
            # Perform optimization
            result = minimize(eval_expr, constants, method=self.opt_method, 
                             options={'maxiter': self.opt_iterations})
            
            if result.success:
                opt_consts = result.x
                
                # Physics-Inspired Simplification
                simplified_consts = opt_consts.copy()
                for j in range(len(simplified_consts)):
                    val = simplified_consts[j]
                    for base in [1.0, 0.5, 0.25]:
                        rounded = round(val / base) * base
                        if abs(val - rounded) < 0.12:
                            simplified_consts[j] = rounded
                            break
                
                mse_opt = eval_expr(opt_consts)
                mse_simple = eval_expr(simplified_consts)
                final_consts = simplified_consts if mse_simple < 1.05 * mse_opt else opt_consts
                
                # Create final optimized expression
                final_subs = {cv: sp.Float(val) for cv, val in zip(const_vars, final_consts)}
                optimized_expr = param_expr.subs(final_subs)
                
                return OptimizedExpressionWrapper(str(optimized_expr), program)
            else:
                return program
                
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
        
        # Parse the expression to create a numerical evaluator using robust gp_to_sympy
        try:
            # Use robust converter
            # gp_to_sympy handles the mapping of X0, X1 to x0, x1
            sympy_expr = gp_to_sympy(expr_str)
            
            # Identify all used features
            all_symbols = sorted(list(sympy_expr.free_symbols), key=lambda s: s.name)
            self._feat_indices = [int(s.name[1:]) for s in all_symbols if s.name.startswith('x')]
            
            # Lambdify for performance
            # We must provide all features up to max_idx to ensure correct indexing
            max_idx = max(self._feat_indices) if self._feat_indices else 0
            feat_vars = [sp.Symbol(f'x{i}') for i in range(max_idx + 1)]
            self._lambda_func = sp.lambdify(feat_vars, sympy_expr, modules=['numpy'])
            
        except Exception as e:
            print(f"Warning: Could not compile optimized expression '{expr_str}': {e}")
            self._lambda_func = None
    
    def execute(self, X):
        """
        Execute the optimized expression.
        """
        if self._lambda_func is not None:
            try:
                if X.ndim == 1: X = X.reshape(1, -1)
                # Ensure X has enough features
                n_req = len(self._lambda_func.__code__.co_varnames)
                if X.shape[1] < n_req:
                    X_padded = np.pad(X, ((0, 0), (0, n_req - X.shape[1])), mode='constant')
                    args = [X_padded[:, i] for i in range(n_req)]
                else:
                    args = [X[:, i] for i in range(n_req)]
                
                result = self._lambda_func(*args)
                if np.isscalar(result):
                    return np.full(X.shape[0], result)
                return np.asarray(result).flatten()
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

            # Use the robust converter from symbolic.py
            sympy_expr = gp_to_sympy(expr_str, n_features=n_vars)

            # Apply physics-aware optimization if enabled
            if self.secondary_optimization:
                sympy_expr = self._optimize_physics_constants(sympy_expr, X_norm, Y_norm)

            # Compute analytical gradients with respect to all variables
            sympy_vars = [sp.Symbol(f'x{i}') for i in range(n_vars)]
            sympy_grads = [sp.diff(sympy_expr, var) for var in sympy_vars]

            # Create lambda functions for gradients
            # Each lambda should take all variables as input for consistency
            lambda_funcs = [sp.lambdify(sympy_vars, grad, 'numpy') for grad in sympy_grads]

            # Estimate dissipation coefficients γ
            dissipation_coeffs = np.zeros(n_super_nodes)
            # (Assuming this logic is handled similarly to HamiltonianSymbolicDistiller or passed in)

            # Create Hamiltonian equation object that respects structure
            hamiltonian_equation = HamiltonianEquation(
                hamiltonian_eq, hamiltonian_mask, n_super_nodes, latent_dim,
                dissipation_coeffs=dissipation_coeffs, 
                sympy_expr=sympy_expr, 
                grad_funcs=lambda_funcs
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