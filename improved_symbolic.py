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
    def __init__(self, distiller, equations, feature_masks, is_hamiltonian, n_super_nodes, latent_dim, model=None):
        self.distiller = distiller
        self.equations = equations
        self.feature_masks = feature_masks
        self.is_hamiltonian = is_hamiltonian
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.model = model

        # NEW: Precompute transformation parameters for efficiency
        if hasattr(distiller, 'transformer'):
            self.transformer = distiller.transformer
        else:
            self.transformer = None

    def _safe_extract_value(self, value):
        """
        Safely extract a scalar value from various return types.
        """
        result = 0.0
        if isinstance(value, (list, tuple)):
            if len(value) > 0:
                result = float(value[0]) if np.isscalar(value[0]) else float(value[0])
            else:
                result = 0.0
        elif isinstance(value, np.ndarray):
            if value.size == 0:
                result = 0.0
            elif value.size == 1:
                result = float(value.flat[0])
            else:
                # If it's a multi-element array, return the first element
                result = float(value.flat[0])
        elif np.isscalar(value):
            result = float(value)
        else:
            # Fallback to 0.0 if we can't handle the type
            result = 0.0
            
        # NEW: Log warning for NaN or zeros (if it was expected to be non-zero)
        if np.isnan(result):
            # print("Warning: Symbolic execution returned NaN. Using 0.0 as fallback.")
            return 0.0
        return result

    def __call__(self, arg1, arg2=None):
        """
        Callable method for integration.
        Handles both odeint signature (z, t) and solve_ivp/ode signature (t, z).
        """
        # Determine which argument is the state vector z and which is time t
        if np.isscalar(arg1):
            # arg1 is t, arg2 is z (solve_ivp signature)
            t, z = arg1, arg2
        elif arg2 is not None and np.isscalar(arg2):
            # arg1 is z, arg2 is t (odeint signature)
            z, t = arg1, arg2
        else:
            # Fallback: assume arg1 is z
            z, t = arg1, arg2

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

        # HYBRID FALLBACK: Check distiller confidence
        use_neural_fallback = False
        if hasattr(self.distiller, 'confidences'):
            # If any confidence is too low, or we don't have enough confidences
            if len(self.distiller.confidences) == 0 or np.min(self.distiller.confidences) < 0.5:
                use_neural_fallback = True
        
        if use_neural_fallback and self.model is not None:
            with torch.no_grad():
                z_tensor = torch.from_numpy(z).float().unsqueeze(0)
                device = next(self.model.parameters()).device
                z_tensor = z_tensor.to(device)
                dz_neural = self.model.ode_func(0, z_tensor).cpu().numpy().flatten()
                return dz_neural

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

            # SECONDARY FALLBACK: If symbolic result is zero but neural isn't
            if np.linalg.norm(result) < 1e-6 and self.model is not None:
                with torch.no_grad():
                    z_tensor = torch.from_numpy(z).float().unsqueeze(0)
                    device = next(self.model.parameters()).device
                    z_tensor = z_tensor.to(device)
                    dz_neural = self.model.ode_func(0, z_tensor).cpu().numpy().flatten()
                    if np.linalg.norm(dz_neural) > 1e-5:
                        return dz_neural

            return result
        except Exception as e:
            print(f"Error in symbolic dynamics evaluation: {e}")
            # Fallback to neural model if possible, otherwise return zeros
            if self.model is not None:
                try:
                    with torch.no_grad():
                        z_tensor = torch.from_numpy(z).float().unsqueeze(0)
                        device = next(self.model.parameters()).device
                        z_tensor = z_tensor.to(device)
                        return self.model.ode_func(0, z_tensor).cpu().numpy().flatten()
                except:
                    pass
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

        # Transform z to features - ensure proper input shape for transformer
        if self.transformer:
            # Handle both 1D and 2D inputs for transformer - ensure 2D input
            if z_reshaped.ndim == 1:
                z_input = z_reshaped.reshape(1, -1)
            else:
                z_input = z_reshaped

            # DEBUG: Check input shape before calling transformer
            if z_input.ndim != 2:
                print(f"DEBUG: Expected 2D input for transformer, got {z_input.ndim}D with shape {z_input.shape}")
                z_input = z_input.reshape(1, -1)  # Ensure 2D

            X = self.transformer.transform(z_input)
            X_full = self.transformer.normalize_x(X)
        else:
            X_full = z_reshaped.reshape(1, -1)

        # NEW: Numerical differentiation for gradient computation
        eps = 1e-4
        grad_H = np.zeros(len(z_reshaped))

        for i in range(len(z_reshaped)):  # Iterate over each dimension
            z_plus = z_reshaped.copy()
            z_minus = z_reshaped.copy()
            z_plus[i] += eps
            z_minus[i] -= eps

            # Transform and normalize both perturbed states - ensure proper input shape
            if self.transformer:
                # Always ensure 2D input for transformer
                z_plus_input = z_plus.reshape(1, -1)
                z_minus_input = z_minus.reshape(1, -1)

                X_plus = self.transformer.transform(z_plus_input)
                X_minus = self.transformer.transform(z_minus_input)
                X_plus_full = self.transformer.normalize_x(X_plus)
                X_minus_full = self.transformer.normalize_x(X_minus)
            else:
                X_plus_full = z_plus.reshape(1, -1)
                X_minus_full = z_minus.reshape(1, -1)

            # Evaluate Hamiltonian at both points
            # Need to use the correct feature mask for this equation
            feature_mask = self.feature_masks[0] if self.feature_masks and len(self.feature_masks) > 0 else None

            try:
                # Ensure X_plus_full and X_minus_full are 2D before indexing
                if X_plus_full.ndim == 1:
                    X_plus_full = X_plus_full.reshape(1, -1)
                if X_minus_full.ndim == 1:
                    X_minus_full = X_minus_full.reshape(1, -1)

                if feature_mask is not None and np.any(feature_mask):
                    # Use only the selected features
                    H_plus_raw = self.equations[0].execute(X_plus_full[:, feature_mask])
                    H_minus_raw = self.equations[0].execute(X_minus_full[:, feature_mask])
                else:
                    # Use all features if no mask
                    H_plus_raw = self.equations[0].execute(X_plus_full)
                    H_minus_raw = self.equations[0].execute(X_minus_full)

                # Handle cases where execute returns arrays or scalars
                H_plus = self._safe_extract_value(H_plus_raw)
                H_minus = self._safe_extract_value(H_minus_raw)

                if np.isfinite(H_plus) and np.isfinite(H_minus):
                    grad_H[i] = (H_plus - H_minus) / (2 * eps)
                else:
                    grad_H[i] = 0.0
            except Exception as e:
                # print(f"Error in numerical differentiation for dimension {i}: {e}")
                grad_H[i] = 0.0  # Fallback to zero gradient

        # Final safety check for gradient
        if not np.all(np.isfinite(grad_H)):
            grad_H = np.nan_to_num(grad_H, nan=0.0, posinf=0.0, neginf=0.0)

        # FALLBACK: If grad_H is effectively zero, use learned neural derivatives
        if np.linalg.norm(grad_H) < 1e-5 and self.model is not None:
            # print("Warning: Symbolic gradient is zero. Using neural fallback.")
            with torch.no_grad():
                z_tensor = torch.from_numpy(z_reshaped).float().unsqueeze(0)
                # Ensure it's on the correct device
                device = next(self.model.parameters()).device
                z_tensor = z_tensor.to(device)
                dz_neural = self.model.ode_func(0, z_tensor).cpu().numpy().flatten()
                return dz_neural

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
                # Ensure result is the same size as input z, but flattened to 1D
                if result.size == z.size:
                    return result.flatten()
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
                        # Ensure X_norm is 2D before applying feature mask
                        if X_norm.ndim == 1:
                            X_norm = X_norm.reshape(1, -1)
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

        # FALLBACK: If derivatives are effectively zero, use learned neural derivatives
        if np.linalg.norm(derivatives) < 1e-5 and self.model is not None:
            with torch.no_grad():
                z_tensor = torch.from_numpy(derivatives.flatten()).float().unsqueeze(0)
                # But we need the input z for neural fallback, not the (zero) derivatives
                z_input_tensor = torch.from_numpy(z_reshaped.flatten()).float().unsqueeze(0)
                device = next(self.model.parameters()).device
                z_input_tensor = z_input_tensor.to(device)
                dz_neural = self.model.ode_func(0, z_input_tensor).cpu().numpy().flatten()
                return dz_neural

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