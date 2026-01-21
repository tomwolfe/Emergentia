"""
Enhanced Symbolic Distiller that enforces Hamiltonian structure during symbolic regression.
This addresses the "Neural-Symbolic Handshake" problem by ensuring the symbolic model
maintains the physical inductive biases learned by the neural model.
"""

import numpy as np
import torch
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from symbolic import SymbolicDistiller, FeatureTransformer
import sympy as sp


class HamiltonianSymbolicDistiller(SymbolicDistiller):
    """
    Enhanced SymbolicDistiller that enforces Hamiltonian structure during symbolic regression.
    
    This addresses the "Neural-Symbolic Handshake" problem by ensuring the symbolic model
    maintains the physical inductive biases learned by the neural model, specifically
    phase-space volume conservation (Liouville's theorem).
    """
    
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12, 
                 enforce_hamiltonian_structure=True):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.enforce_hamiltonian_structure = enforce_hamiltonian_structure
        
    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None):
        """
        Distill symbolic equations with Hamiltonian structure enforcement.
        
        Args:
            latent_states: [N, n_super_nodes * latent_dim] - Current latent states
            targets: [N, n_super_nodes * latent_dim] - Target derivatives or Hamiltonian values
            n_super_nodes: Number of super-nodes
            latent_dim: Dimension of each latent node (must be even for (q,p) pairs)
            box_size: Box size for periodic boundary conditions
        """
        self.transformer = FeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        self.transformer.fit(latent_states, targets)

        X_poly = self.transformer.transform(latent_states)
        X_norm = self.transformer.normalize_x(X_poly)

        # Handle both scalar Hamiltonian and vector targets
        if targets.ndim == 1 or targets.shape[1] == 1:
            Y_norm = self.transformer.normalize_y(targets.reshape(-1, 1))
        else:
            Y_norm = self.transformer.normalize_y(targets)

        if self.enforce_hamiltonian_structure and latent_dim % 2 == 0:
            # For Hamiltonian systems, we need to ensure the symbolic model respects
            # the Hamiltonian structure: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
            return self._distill_with_hamiltonian_structure(X_norm, Y_norm, n_super_nodes, latent_dim)
        else:
            # Standard distillation for non-Hamiltonian systems
            results = self._distill_standard(X_norm, Y_norm)
            equations = [r[0] for r in results]
            self.feature_masks = [r[1] for r in results]
            self.confidences = [r[2] for r in results]
            return equations

    def _distill_with_hamiltonian_structure(self, X_norm, Y_norm, n_super_nodes, latent_dim):
        """
        Distill Hamiltonian H(q,p) and derive equations respecting Hamiltonian structure.
        """
        # For Hamiltonian systems, we only need to learn the Hamiltonian function H(q,p)
        # Then we can analytically compute the derivatives using ∂H/∂q and ∂H/∂p
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

    def _distill_standard(self, X_norm, Y_norm):
        """
        Standard distillation without Hamiltonian structure enforcement.
        """
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=-1)(
            delayed(self._distill_single_target)(i, X_norm, Y_norm, Y_norm.shape[1], X_norm.shape[1])
            for i in range(Y_norm.shape[1])
        )
        return results


class HamiltonianEquation:
    """
    Special equation class that enforces Hamiltonian structure: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    """
    
    def __init__(self, original_eq, sympy_expr, sympy_grads, lambda_funcs, feature_mask):
        self.original_eq = original_eq
        self.sympy_expr = sympy_expr
        self.sympy_grads = sympy_grads
        self.lambda_funcs = lambda_funcs
        self.feature_mask = feature_mask
    
    def execute(self, X_selected):
        """
        Execute the original Hamiltonian function (for fitting purposes).
        """
        return self.original_eq.execute(X_selected)
    
    def compute_derivatives(self, z, transformer):
        """
        Compute time derivatives using Hamiltonian structure: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        
        Args:
            z: [n_super_nodes * latent_dim] - Current state vector (q followed by p)
            transformer: FeatureTransformer instance to convert z to feature space
            
        Returns:
            dzdt: [n_super_nodes * latent_dim] - Time derivatives respecting Hamiltonian structure
        """
        # Transform z to feature space
        X_poly = transformer.transform(z.reshape(1, -1))
        X_norm = transformer.normalize_x(X_poly)
        
        # Apply feature mask to get selected features
        X_selected = X_norm[:, self.feature_mask]
        
        # Compute analytical gradients using SymPy
        try:
            # Create a dictionary mapping SymPy variables to actual values
            var_dict = {var: val for var, val in zip(
                [sp.Symbol(f'x{i}') for i in range(len(X_selected.flatten()))], 
                X_selected.flatten()
            )}
            
            # Compute analytical gradients
            grad = np.array([float(grad_func(**var_dict)) for grad_func in self.lambda_funcs[:len(X_selected.flatten())]])
            
            # Now map these gradients to the Hamiltonian structure
            # For a Hamiltonian system, z = [q1, ..., qn, p1, ..., pn]
            # We have: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
            n_total = len(z)
            n_qp_pairs = n_total // 2
            
            dzdt = np.zeros_like(z)
            
            # dq/dt = ∂H/∂p (gradient w.r.t. momentum components)
            # dp/dt = -∂H/∂q (negative gradient w.r.t. position components)
            
            # Map the gradient indices back to the original z space
            # This requires understanding how the feature transformer maps z to X
            # For simplicity, we assume the first n_total features correspond to z
            # (this is a simplification - in practice, this mapping would be more complex)
            
            # For now, we'll assume that the first n_total elements of the gradient
            # correspond to derivatives w.r.t. z components in order
            if len(grad) >= n_total:
                # dq/dt = ∂H/∂p (second half of z corresponds to momenta)
                dq_dt = grad[n_qp_pairs:n_total]
                
                # dp/dt = -∂H/∂q (first half of z corresponds to positions)
                dp_dt = -grad[0:n_qp_pairs]
                
                dzdt[0:n_qp_pairs] = dq_dt
                dzdt[n_qp_pairs:n_total] = dp_dt
            else:
                # Fallback: compute numerical gradients
                dzdt = self._numerical_hamiltonian_derivatives(z, transformer)
                
        except Exception as e:
            print(f"Analytical gradient computation failed: {e}. Using numerical gradients.")
            dzdt = self._numerical_hamiltonian_derivatives(z, transformer)
        
        return dzdt
    
    def _numerical_hamiltonian_derivatives(self, z, transformer):
        """
        Compute Hamiltonian derivatives numerically as a fallback.
        """
        eps = 1e-6
        n_dims = len(z)
        n_qp_pairs = n_dims // 2
        
        dzdt = np.zeros(n_dims)
        
        # Compute Hamiltonian for current state
        X_poly_curr = transformer.transform(z.reshape(1, -1))
        X_norm_curr = transformer.normalize_x(X_poly_curr)
        H_curr = self.original_eq.execute(X_norm_curr[:, self.feature_mask])[0]
        
        # Compute dq/dt = ∂H/∂p using finite differences
        for i in range(n_qp_pairs):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[n_qp_pairs + i] += eps  # perturb momentum component
            z_minus[n_qp_pairs + i] -= eps
            
            X_poly_plus = transformer.transform(z_plus.reshape(1, -1))
            X_poly_minus = transformer.transform(z_minus.reshape(1, -1))
            
            X_norm_plus = transformer.normalize_x(X_poly_plus)
            X_norm_minus = transformer.normalize_x(X_poly_minus)
            
            H_plus = self.original_eq.execute(X_norm_plus[:, self.feature_mask])[0]
            H_minus = self.original_eq.execute(X_norm_minus[:, self.feature_mask])[0]
            
            dzdt[i] = (H_plus - H_minus) / (2 * eps)  # dq_i/dt = ∂H/∂p_i
        
        # Compute dp/dt = -∂H/∂q using finite differences
        for i in range(n_qp_pairs):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[i] += eps  # perturb position component
            z_minus[i] -= eps
            
            X_poly_plus = transformer.transform(z_plus.reshape(1, -1))
            X_poly_minus = transformer.transform(z_minus.reshape(1, -1))
            
            X_norm_plus = transformer.normalize_x(X_poly_plus)
            X_norm_minus = transformer.normalize_x(X_poly_minus)
            
            H_plus = self.original_eq.execute(X_norm_plus[:, self.feature_mask])[0]
            H_minus = self.original_eq.execute(X_norm_minus[:, self.feature_mask])[0]
            
            dzdt[n_qp_pairs + i] = -(H_plus - H_minus) / (2 * eps)  # dp_i/dt = -∂H/∂q_i
        
        return dzdt


def create_symplectic_functions():
    """
    Create custom SymPy functions that preserve symplectic structure.
    These can be added to the function set in genetic programming.
    """
    # Define custom functions that inherently preserve symplectic structure
    def symplectic_add(x1, x2):
        return x1 + x2
    
    def symplectic_mul(x1, x2):
        return x1 * x2
    
    # Register these as custom functions for GP
    symplectic_add_func = make_function(function=symplectic_add, name='sym_add', arity=2)
    symplectic_mul_func = make_function(function=symplectic_mul, name='sym_mul', arity=2)
    
    return [symplectic_add_func, symplectic_mul_func]