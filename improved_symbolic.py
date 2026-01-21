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

    def _safe_extract_value(self, value):
        """
        Safely extract a scalar value from various return types.
        """
        if isinstance(value, (list, tuple)):
            if len(value) > 0:
                return float(value[0]) if np.isscalar(value[0]) else float(value[0])
            else:
                return 0.0
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
            elif value.size == 1:
                return float(value.flat[0])
            else:
                # If it's a multi-element array, return the first element
                return float(value.flat[0])
        elif np.isscalar(value):
            return float(value)
        else:
            # Fallback to 0.0 if we can't handle the type
            return 0.0

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

        # Store the original shape for later use
        z_original_shape = z.shape

        # Ensure z is 1D for processing (odeint expects 1D arrays)
        if z.ndim > 1:
            z = z.flatten()

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

            # Ensure result has the same shape as input z for odeint
            # Since we ensured z is 1D, result should also be 1D
            if hasattr(result, 'shape'):
                # If result is 2D but should be 1D, flatten it
                if len(result.shape) > 1 and np.prod(result.shape) == len(z):
                    result = result.flatten()
                elif result.shape != z.shape:
                    # If shapes don't match, try to reshape or return zeros
                    if result.size == z.size:
                        result = result.reshape(z.shape)
                    else:
                        print(f"Shape mismatch: expected {z.shape}, got {result.shape}. Returning zeros.")
                        return np.zeros_like(z)
            else:
                # If result doesn't have a shape attribute (scalar), convert to array
                if np.isscalar(result):
                    result = np.full(z.shape, result)
                else:
                    result = np.asarray(result)

                    # If result is still not the right shape, try to fix it
                    if result.shape != z.shape:
                        if result.size == z.size:
                            result = result.reshape(z.shape)
                        else:
                            print(f"Result shape mismatch: expected {z.shape}, got {result.shape}. Returning zeros.")
                            return np.zeros_like(z)

            # Ensure result is 1D array for odeint
            if result.ndim > 1:
                result = result.flatten()

            # Make sure the result has the same length as input z
            if result.shape[0] != z.shape[0]:
                print(f"Size mismatch: expected {z.shape[0]}, got {result.shape[0]}. Returning zeros.")
                return np.zeros_like(z)

            return result
        except Exception as e:
            print(f"Error in symbolic dynamics evaluation: {e}")
            # Return a safe fallback (zeros) with same shape as input to prevent integration failure
            return np.zeros_like(z)

    def _compute_hamiltonian_derivative(self, z):
        """
        Compute Hamiltonian derivatives: dz/dt = J * ∇H(z) where J is the symplectic matrix
        For canonical coordinates (q,p): dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        """
        if self.equations[0] is None:
            # If no equation discovered, return zeros
            return np.zeros_like(z)

        # Ensure z is 1D for processing
        if z.ndim == 1:
            z_reshaped = z
        else:
            z_reshaped = z.flatten()

        # Transform z to features
        if self.transformer:
            # Handle both 1D and 2D inputs for transformer
            X = self.transformer.transform(z_reshaped.reshape(1, -1))
            X_full = self.transformer.normalize_x(X)
        else:
            X_full = z_reshaped.reshape(1, -1)

        # NEW: Numerical differentiation for gradient computation
        eps = 1e-8
        grad_H = np.zeros_like(z_reshaped)

        for i in range(len(z_reshaped)):  # Iterate over each dimension
            z_plus = z_reshaped.copy()
            z_minus = z_reshaped.copy()
            z_plus[i] += eps
            z_minus[i] -= eps

            # Transform and normalize both perturbed states
            if self.transformer:
                X_plus = self.transformer.transform(z_plus.reshape(1, -1))
                X_minus = self.transformer.transform(z_minus.reshape(1, -1))
                X_plus_full = self.transformer.normalize_x(X_plus)
                X_minus_full = self.transformer.normalize_x(X_minus)
            else:
                X_plus_full = z_plus.reshape(1, -1)
                X_minus_full = z_minus.reshape(1, -1)

            # Evaluate Hamiltonian at both points
            # Need to use the correct feature mask for this equation
            feature_mask = self.feature_masks[0] if self.feature_masks and len(self.feature_masks) > 0 else None

            try:
                if feature_mask is not None and np.any(feature_mask):
                    # Use only the selected features
                    H_plus = self.equations[0].execute(X_plus_full[:, feature_mask])
                    H_minus = self.equations[0].execute(X_minus_full[:, feature_mask])
                else:
                    # Use all features if no mask
                    H_plus = self.equations[0].execute(X_plus_full)
                    H_minus = self.equations[0].execute(X_minus_full)

                # Handle cases where execute returns arrays or scalars
                # NEW: More robust handling to prevent tuple index out of range
                H_plus = self._safe_extract_value(H_plus)
                H_minus = self._safe_extract_value(H_minus)

                grad_H[i] = (H_plus - H_minus) / (2 * eps)
            except Exception as e:
                print(f"Error in numerical differentiation for dimension {i}: {e}")
                grad_H[i] = 0.0  # Fallback to zero gradient

        # For Hamiltonian systems: [dq/dt, dp/dt] = [dH/dp, -dH/dq]
        # Assuming z = [q, p] where q and p have equal dimensions
        half_dim = len(z_reshaped) // 2

        # Check if we have enough dimensions to split into q and p
        if len(z_reshaped) % 2 != 0:
            # Odd number of dimensions - can't split evenly
            # Return zeros with correct shape
            return np.zeros_like(z)

        # grad_H is [∂H/∂q₁, ∂H/∂q₂, ..., ∂H/∂p₁, ∂H/∂p₂, ...]
        # So we split it into ∂H/∂q and ∂H/∂p parts
        dH_dq = grad_H[:half_dim]    # Partial derivatives w.r.t. positions
        dH_dp = grad_H[half_dim:]    # Partial derivatives w.r.t. momenta

        # Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        dq_dt = dH_dp                   # Rate of change of position = ∂H/∂p
        dp_dt = -dH_dq                  # Rate of change of momentum = -∂H/∂q

        # Combine: [dq/dt, dp/dt]
        result = np.hstack([dq_dt, dp_dt])

        # Ensure result has the correct shape for odeint - same as input z
        if result.shape != z.shape:
            if z.shape == () and result.shape == (1,):  # scalar input, 1-element array output
                return result[0]
            elif z.ndim == 1 and result.ndim == 1 and result.shape[0] == z.shape[0]:
                return result
            else:
                print(f"Hamiltonian derivative shape mismatch: expected {z.shape}, got {result.shape}. Returning zeros.")
                return np.zeros_like(z)

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

        # Ensure derivatives array has the correct shape
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
                            # Use safe extraction method
                            derivatives[0, i] = self._safe_extract_value(result)
                        else:
                            derivatives[0, i] = 0.0
                    else:
                        result = eq.execute(X_norm)
                        # Use safe extraction method
                        derivatives[0, i] = self._safe_extract_value(result)
                except Exception as e:
                    print(f"Error evaluating equation {i}: {e}")
                    derivatives[0, i] = 0.0  # Fallback to zero
            else:
                derivatives[0, i] = 0.0  # Default to zero if no equation

        # NEW: Apply numerical stability constraints
        derivatives = np.clip(derivatives, -1e2, 1e2)

        # Ensure result has the correct shape for odeint
        if derivatives.ndim == 2 and derivatives.shape[0] == 1:
            derivatives = derivatives[0]  # Flatten to 1D if needed

        # Return proper shape for odeint - ensure it matches input z shape
        # Ensure result is always 1D and same length as input z
        if derivatives.ndim > 1:
            derivatives = derivatives.flatten()

        if len(derivatives) != len(z):
            print(f"General derivative shape mismatch: expected {len(z)}, got {len(derivatives)}. Returning zeros.")
            return np.zeros_like(z)

        return derivatives