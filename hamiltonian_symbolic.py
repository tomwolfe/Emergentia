"""
Enhanced Symbolic Distiller that enforces Hamiltonian structure during symbolic regression.
"""

import numpy as np
import sympy as sp
from symbolic import SymbolicDistiller, FeatureTransformer

class HamiltonianSymbolicDistiller(SymbolicDistiller):
    """
    Enhanced SymbolicDistiller that enforces Hamiltonian structure.
    dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
    """
    
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, max_features=12, 
                 enforce_hamiltonian_structure=True):
        super().__init__(populations, generations, stopping_criteria, max_features)
        self.enforce_hamiltonian_structure = enforce_hamiltonian_structure
        
    def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None):
        if not self.enforce_hamiltonian_structure or latent_dim % 2 != 0:
            return super().distill(latent_states, targets, n_super_nodes, latent_dim, box_size)

        self.transformer = FeatureTransformer(n_super_nodes, latent_dim, box_size=box_size)
        self.transformer.fit(latent_states, targets)
        X_norm = self.transformer.normalize_x(self.transformer.transform(latent_states))
        
        # Targets for Hamiltonian is usually the scalar energy H
        # If targets is the derivative, we need to find H such that its derivatives match.
        # But usually we have a scalar H from the neural model if include_hamiltonian=True.
        Y_norm = self.transformer.normalize_y(targets)

        # Distill the Hamiltonian function H
        # We assume the first target is the Hamiltonian scalar
        h_prog, h_mask, h_conf = self._distill_single_target(0, X_norm, Y_norm, 1, latent_dim)
        
        if h_prog is None:
            return super().distill(latent_states, targets, n_super_nodes, latent_dim, box_size)
            
        # Wrap it in a class that can compute derivatives
        ham_eq = HamiltonianEquation(h_prog, h_mask, n_super_nodes, latent_dim)
        self.feature_masks = [h_mask]
        self.confidences = [h_conf]
        
        return [ham_eq]

class HamiltonianEquation:
    """
    Special equation class that computes derivatives from a scalar Hamiltonian.
    """
    def __init__(self, h_prog, feature_mask, n_super_nodes, latent_dim):
        self.h_prog = h_prog
        self.feature_mask = feature_mask
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.length_ = getattr(h_prog, 'length_', 1)

    def execute(self, X):
        return self.h_prog.execute(X)

    def compute_derivatives(self, z, transformer):
        """
        Numerically compute dq/dt = ∂H/∂p and dp/dt = -∂H/∂q
        """
        eps = 1e-4
        n_total = len(z)
        n_qp = n_total // 2
        dzdt = np.zeros(n_total)
        
        def get_H(state):
            X_p = transformer.transform(state.reshape(1, -1))
            X_n = transformer.normalize_x(X_p)
            return self.h_prog.execute(X_n[:, self.feature_mask])[0]

        # dq/dt = ∂H/∂p
        for i in range(n_qp):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[n_qp + i] += eps
            z_minus[n_qp + i] -= eps
            dzdt[i] = (get_H(z_plus) - get_H(z_minus)) / (2 * eps)
            
        # dp/dt = -∂H/∂q
        for i in range(n_qp):
            z_plus = z.copy()
            z_minus = z.copy()
            z_plus[i] += eps
            z_minus[i] -= eps
            dzdt[n_qp + i] = -(get_H(z_plus) - get_H(z_minus)) / (2 * eps)
            
        return dzdt

    def __str__(self):
        return f"HamiltonianH({self.h_prog})"
