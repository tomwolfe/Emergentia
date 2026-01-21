"""
Improved Symbolic Dynamics Module

This module implements enhanced symbolic dynamics for integration with the neural network.
"""

import numpy as np
from scipy.integrate import ode
import torch


class ImprovedSymbolicDynamics:
    """
    Enhanced symbolic dynamics class for integrating discovered equations.
    """
    def __init__(self, distiller, equations, feature_masks, is_hamiltonian, n_super_nodes, latent_dim):
        self.distiller = distiller
        self.equations = equations
        self.feature_masks = feature_masks
        self.is_hamiltonian = is_hamiltonian
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        
        # NEW: Precompute transformation parameters for efficiency
        if hasattr(distiller, 'transformer'):
            self.transformer = distiller.transformer
        else:
            self.transformer = None

    def __call__(self, t, z):
        """
        Callable method for integration with scipy.integrate.odeint
        """
        # Ensure z is a numpy array
        if torch.is_tensor(z):
            z = z.detach().cpu().numpy()

        # Handle the case where z is a scalar (single value)
        if np.isscalar(z):
            z = np.array([z])

        if z.ndim == 1:
            z = z.reshape(1, -1)

        # NEW: Enhanced error handling and numerical stability
        try:
            # Clamp input to prevent numerical explosion
            z_clamped = np.clip(z, -1e3, 1e3)

            if self.is_hamiltonian:
                # For Hamiltonian systems, we compute dH/dz
                result = self._compute_hamiltonian_derivative(z_clamped)
            else:
                # For general systems, evaluate each equation
                result = self._compute_general_derivative(z_clamped)

            # Return the proper shape for odeint
            if result.shape[0] == 1:
                return result[0]  # Return 1D array if single sample
            else:
                return result
        except Exception as e:
            print(f"Error in symbolic dynamics evaluation: {e}")
            # Return a safe fallback (zeros) to prevent integration failure
            if z.ndim == 1:
                return np.zeros_like(z)
            else:
                return np.zeros(z.shape[1])  # Return 1D array for odeint

    def _compute_hamiltonian_derivative(self, z):
        """
        Compute Hamiltonian derivatives: dz/dt = J * ∇H(z) where J is the symplectic matrix
        For canonical coordinates (q,p): dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        """
        if self.equations[0] is None:
            # If no equation discovered, return zeros
            if z.ndim == 1:
                return np.zeros_like(z)
            else:
                return np.zeros(z.shape[1])  # Return 1D array for odeint

        # Transform z to features
        if self.transformer:
            X = self.transformer.transform(z)
            X_norm = self.transformer.normalize_x(X)
        else:
            X_norm = z

        # NEW: Proper handling of dimensions for numerical differentiation
        if z.ndim == 1:
            # Single sample case - reshape for processing
            z_single = z
            z_reshaped = z.reshape(1, -1)
        else:
            z_reshaped = z
            z_single = z[0]  # Take first sample if multiple provided

        # NEW: Numerical differentiation for gradient computation
        eps = 1e-8
        grad_H = np.zeros_like(z_reshaped)

        for i in range(z_reshaped.shape[1]):  # Iterate over each dimension
            z_plus = z_reshaped.copy()
            z_minus = z_reshaped.copy()
            z_plus[0, i] += eps
            z_minus[0, i] -= eps

            # Transform and normalize both perturbed states
            if self.transformer:
                X_plus = self.transformer.transform(z_plus)
                X_minus = self.transformer.transform(z_minus)
                X_plus_norm = self.transformer.normalize_x(X_plus)
                X_minus_norm = self.transformer.normalize_x(X_minus)
            else:
                X_plus_norm = z_plus
                X_minus_norm = z_minus

            # Evaluate Hamiltonian at both points
            try:
                H_plus = self.equations[0].execute(X_plus_norm)
                H_minus = self.equations[0].execute(X_minus_norm)
                # Handle cases where execute returns arrays or scalars
                if hasattr(H_plus, '__len__') and len(H_plus) > 0:
                    H_plus = H_plus[0] if isinstance(H_plus, (list, np.ndarray)) else H_plus
                if hasattr(H_minus, '__len__') and len(H_minus) > 0:
                    H_minus = H_minus[0] if isinstance(H_minus, (list, np.ndarray)) else H_minus
                grad_H[0, i] = (float(H_plus) - float(H_minus)) / (2 * eps)
            except:
                grad_H[0, i] = 0.0  # Fallback to zero gradient

        # For Hamiltonian systems: [dq/dt, dp/dt] = [dH/dp, -dH/dq]
        # Assuming z = [q, p] where q and p have equal dimensions
        half_dim = z_reshaped.shape[1] // 2
        dq_dt = grad_H[:, half_dim:2*half_dim]  # dH/dp
        dp_dt = -grad_H[:, :half_dim]           # -dH/dq

        result = np.hstack([dq_dt, dp_dt])

        # Return the proper shape for odeint
        if z.ndim == 1:
            return result[0]  # Return 1D array if single sample
        else:
            return result

    def _compute_general_derivative(self, z):
        """
        Compute general derivatives dz/dt = f(z) for each dimension
        """
        # Handle reshaping for processing
        if z.ndim == 1:
            z_reshaped = z.reshape(1, -1)
        else:
            z_reshaped = z

        if self.transformer:
            X = self.transformer.transform(z_reshaped)
            X_norm = self.transformer.normalize_x(X)
        else:
            X_norm = z_reshaped

        derivatives = np.zeros_like(z_reshaped)

        for i, eq in enumerate(self.equations):
            if eq is not None and i < derivatives.shape[1]:
                try:
                    # NEW: Apply feature mask if available
                    if i < len(self.feature_masks) and self.feature_masks[i] is not None:
                        # Use only the relevant features for this equation
                        relevant_X = X_norm[:, self.feature_masks[i]]
                        if relevant_X.size > 0:
                            result = eq.execute(relevant_X)
                            # Handle different return types
                            if hasattr(result, '__len__') and len(result) > 0:
                                derivatives[0, i] = result[0] if isinstance(result, (list, np.ndarray)) else result
                            else:
                                derivatives[0, i] = result
                        else:
                            derivatives[0, i] = 0.0
                    else:
                        result = eq.execute(X_norm)
                        # Handle different return types
                        if hasattr(result, '__len__') and len(result) > 0:
                            derivatives[0, i] = result[0] if isinstance(result, (list, np.ndarray)) else result
                        else:
                            derivatives[0, i] = result
                except Exception as e:
                    print(f"Error evaluating equation {i}: {e}")
                    derivatives[0, i] = 0.0  # Fallback to zero
            else:
                derivatives[0, i] = 0.0  # Default to zero if no equation

        # NEW: Apply numerical stability constraints
        derivatives = np.clip(derivatives, -1e2, 1e2)

        # Return proper shape for odeint
        if z.ndim == 1:
            return derivatives[0]  # Return 1D array if single sample
        else:
            return derivatives