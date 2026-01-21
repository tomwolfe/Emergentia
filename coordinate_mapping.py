"""
Improved coordinate mapping between neural and symbolic representations.
Addresses the issue where the GNN encoder might learn a rotated or non-linear
coordinate transformation that makes symbolic distillation less interpretable.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import sympy as sp


class CoordinateMapper:
    """
    Maps between neural latent space and interpretable physical coordinates.
    
    This addresses the issue where the neural encoder might learn a coordinate
    system that isn't aligned with the physical coordinates needed for symbolic
    regression. The mapper learns a transformation from the neural space to
    an interpretable coordinate system.
    """
    
    def __init__(self, n_latent_dims, n_physical_dims=2, use_rotation_alignment=True, 
                 use_nonlinear_mapping=True, max_degree=2):
        """
        Initialize the coordinate mapper.
        
        Args:
            n_latent_dims: Total number of latent dimensions per super-node
            n_physical_dims: Number of physical dimensions (typically 2 or 3)
            use_rotation_alignment: Whether to learn rotation alignment
            use_nonlinear_mapping: Whether to include nonlinear mappings
            max_degree: Maximum degree for polynomial mappings
        """
        self.n_latent_dims = n_latent_dims
        self.n_physical_dims = n_physical_dims
        self.use_rotation_alignment = use_rotation_alignment
        self.use_nonlinear_mapping = use_nonlinear_mapping
        self.max_degree = max_degree
        
        # Transformation models
        self.linear_mapper = LinearRegression()
        self.nonlinear_mapper = None
        self.neural_scaler = StandardScaler()
        self.physical_scaler = StandardScaler()
        
        # Transformation parameters
        self.linear_weights = None
        self.linear_bias = None
        
    def fit(self, neural_coords, physical_coords=None):
        """
        Fit the coordinate transformation.
        
        Args:
            neural_coords: [N, n_super_nodes, n_latent_dims] neural coordinates
            physical_coords: [N, n_super_nodes, n_physical_dims] physical coordinates (optional)
        """
        # Reshape to [N*n_super_nodes, n_latent_dims]
        N, K, D = neural_coords.shape
        neural_flat = neural_coords.reshape(-1, D)
        
        if physical_coords is not None:
            # If we have ground truth physical coordinates, learn the mapping
            physical_flat = physical_coords.reshape(-1, self.n_physical_dims)
            
            # Standardize neural coordinates
            neural_scaled = self.neural_scaler.fit_transform(neural_flat)
            
            # Standardize physical coordinates
            physical_scaled = self.physical_scaler.fit_transform(physical_flat)
            
            # 1. Fit linear mapping as base
            self.linear_mapper.fit(neural_scaled, physical_scaled)
            self.linear_weights = self.linear_mapper.coef_
            self.linear_bias = self.linear_mapper.intercept_
            
            # 2. Fit nonlinear mapping if requested (using a small MLP or symbolic-friendly model)
            if self.use_nonlinear_mapping:
                from sklearn.neural_network import MLPRegressor
                # Small MLP to capture non-linear residuals or complex mappings
                self.nonlinear_mapper = MLPRegressor(
                    hidden_layer_sizes=(16, 16), 
                    max_iter=500, 
                    alpha=0.1, # Strong regularization to stay "near-linear"
                    random_state=42
                )
                self.nonlinear_mapper.fit(neural_scaled, physical_scaled)
            
        else:
            # If no ground truth, try to infer physical coordinates from structure
            self._infer_from_structure(neural_flat)
    
    def _infer_from_structure(self, neural_flat):
        """
        Infer physical coordinates from the structure of the neural representation.
        """
        N, D = neural_flat.shape
        
        # Standardize neural coordinates
        neural_scaled = self.neural_scaler.fit_transform(neural_flat)
        
        # If using rotation alignment, try to find principal directions
        if self.use_rotation_alignment:
            pca = PCA(n_components=min(self.n_physical_dims, D))
            pca.fit(neural_scaled)
            self.linear_weights = pca.components_  # [n_phys_dims, D]
            self.linear_bias = np.zeros(self.n_physical_dims)
        else:
            # Simply take the first n_physical_dims
            self.linear_weights = np.eye(self.n_physical_dims, D)
            self.linear_bias = np.zeros(self.n_physical_dims)
    
    def neural_to_physical(self, neural_coords):
        """
        Transform neural coordinates to physical coordinates.
        """
        original_shape = neural_coords.shape
        is_single = len(original_shape) == 1
        
        if is_single:
            neural_coords = neural_coords.reshape(1, 1, -1)
        
        N, K, D = neural_coords.shape
        neural_flat = neural_coords.reshape(-1, D)
        
        # Standardize
        neural_scaled = self.neural_scaler.transform(neural_flat)
        
        # Apply transformation
        if self.nonlinear_mapper is not None:
            physical_scaled = self.nonlinear_mapper.predict(neural_scaled)
        else:
            physical_scaled = neural_scaled @ self.linear_weights.T + self.linear_bias
        
        # Reshape back
        if is_single:
            return physical_scaled[0]
        else:
            return physical_scaled.reshape(N, K, self.n_physical_dims)
    
    def physical_to_neural(self, physical_coords):
        """
        Transform physical coordinates back to neural coordinates.
        """
        # Linear inverse is well-defined, non-linear might be harder
        # For symbolic purposes, the neural->physical mapping is the most important
        if self.linear_weights is None:
            return physical_coords
            
        original_shape = physical_coords.shape
        is_single = len(original_shape) == 1
        
        if is_single:
            physical_coords = physical_coords.reshape(1, 1, -1)
        
        N, K, D_phys = physical_coords.shape
        physical_flat = physical_coords.reshape(-1, D_phys)
        
        # Use pseudo-inverse of linear weights
        weights_pinv = np.linalg.pinv(self.linear_weights) # [D, n_phys_dims]
        
        neural_scaled = (physical_flat - self.linear_bias) @ weights_pinv.T
        neural_flat = self.neural_scaler.inverse_transform(neural_scaled)
        
        if is_single:
            return neural_flat[0]
        else:
            return neural_flat.reshape(N, K, self.n_latent_dims)


class AlignedHamiltonianSymbolicDistiller:
    """
    Hamiltonian symbolic distiller with coordinate alignment.
    
    This ensures that the symbolic Hamiltonian is expressed in terms of
    properly aligned position and momentum coordinates, making the
    resulting equations more interpretable.
    """
    
    def __init__(self, populations=2000, generations=40, stopping_criteria=0.001, 
                 max_features=12, enforce_hamiltonian_structure=True):
        self.populations = populations
        self.generations = generations
        self.stopping_criteria = stopping_criteria
        self.max_features = max_features
        self.enforce_hamiltonian_structure = enforce_hamiltonian_structure
        
        # Coordinate mapper
        self.coord_mapper = None
        
    def fit_coordinate_mapper(self, neural_latents, physical_coords=None):
        """
        Fit the coordinate mapper to align neural latents with physical coordinates.
        
        Args:
            neural_latents: [N, n_super_nodes, latent_dim] neural representations
            physical_coords: [N, n_super_nodes, n_physical_dims] physical coordinates (optional)
        """
        n_super_nodes, latent_dim = neural_latents.shape[1], neural_latents.shape[2]
        n_physical_dims = physical_coords.shape[2] if physical_coords is not None else 2
        
        # Separate position and momentum parts if Hamiltonian
        if latent_dim % 2 == 0:
            pos_dims = latent_dim // 2
            # Take only position coordinates for alignment
            pos_latents = neural_latents[:, :, :pos_dims]
        else:
            pos_latents = neural_latents
        
        self.coord_mapper = CoordinateMapper(
            n_latent_dims=pos_latents.shape[-1],
            n_physical_dims=n_physical_dims
        )
        self.coord_mapper.fit(pos_latents, physical_coords)
    
    def transform_to_aligned_coords(self, neural_latents):
        """
        Transform neural latents to aligned coordinate system.
        
        Args:
            neural_latents: [N, n_super_nodes, latent_dim] neural representations
            
        Returns:
            aligned_latents: [N, n_super_nodes, latent_dim] aligned representations
        """
        if self.coord_mapper is None:
            raise ValueError("Coordinate mapper must be fitted before transformation")
        
        n_super_nodes, latent_dim = neural_latents.shape[1], neural_latents.shape[2]
        
        if latent_dim % 2 == 0:
            # Hamiltonian system: separate position and momentum
            pos_dims = latent_dim // 2
            pos_part = neural_latents[:, :, :pos_dims]
            mom_part = neural_latents[:, :, pos_dims:]
            
            # Transform position part to aligned coordinates
            aligned_pos = self.coord_mapper.neural_to_physical(pos_part)
            
            # For momentum, apply the same transformation (or learn a separate one)
            aligned_mom = self.coord_mapper.neural_to_physical(mom_part)
            
            # Concatenate back
            aligned_latents = np.concatenate([aligned_pos, aligned_mom], axis=-1)
        else:
            # Non-Hamiltonian: transform the whole thing
            aligned_latents = self.coord_mapper.neural_to_physical(neural_latents)
        
        return aligned_latents
    
    def distill_with_alignment(self, neural_latents, targets, n_super_nodes, latent_dim, 
                              physical_coords=None, box_size=None):
        """
        Distill symbolic equations with coordinate alignment.
        
        Args:
            neural_latents: [N, n_super_nodes, latent_dim] neural representations
            targets: [N, n_super_nodes * latent_dim] target derivatives
            n_super_nodes: Number of super-nodes
            latent_dim: Dimension of each latent node
            physical_coords: [N, n_super_nodes, n_physical_dims] physical coordinates (optional)
            box_size: Box size for periodic boundary conditions
        """
        # Fit coordinate mapper if not already fitted
        if self.coord_mapper is None:
            self.fit_coordinate_mapper(neural_latents, physical_coords)
        
        # Transform to aligned coordinates
        aligned_latents = self.transform_to_aligned_coords(neural_latents)
        
        # Flatten for symbolic regression
        aligned_flat = aligned_latents.reshape(aligned_latents.shape[0], -1)
        
        # Now proceed with standard symbolic distillation using aligned coordinates
        from hamiltonian_symbolic import HamiltonianSymbolicDistiller
        
        if self.enforce_hamiltonian_structure and latent_dim % 2 == 0:
            distiller = HamiltonianSymbolicDistiller(
                populations=self.populations,
                generations=self.generations,
                stopping_criteria=self.stopping_criteria,
                max_features=self.max_features,
                enforce_hamiltonian_structure=True
            )
            return distiller.distill(aligned_flat, targets, n_super_nodes, latent_dim, box_size)
        else:
            from symbolic import SymbolicDistiller
            distiller = SymbolicDistiller(
                populations=self.populations,
                generations=self.generations,
                stopping_criteria=self.stopping_criteria,
                max_features=self.max_features
            )
            return distiller.distill(aligned_flat, targets, n_super_nodes, latent_dim, box_size)


def create_aligned_coord_mapper(n_latent_dims, n_physical_dims=2):
    """
    Factory function to create an aligned coordinate mapper.
    """
    return CoordinateMapper(
        n_latent_dims=n_latent_dims,
        n_physical_dims=n_physical_dims,
        use_rotation_alignment=True,
        use_nonlinear_mapping=False
    )


class EnhancedCoordinateMapper(CoordinateMapper):
    """
    Enhanced coordinate mapper with additional features for complex transformations.
    """
    
    def __init__(self, n_latent_dims, n_physical_dims=2, use_rotation_alignment=True, 
                 use_nonlinear_mapping=True, max_degree=2, use_pca_alignment=True):
        super().__init__(n_latent_dims, n_physical_dims, use_rotation_alignment, 
                         use_nonlinear_mapping, max_degree)
        self.use_pca_alignment = use_pca_alignment
        self.complex_transformation = None
        
    def fit(self, neural_coords, physical_coords=None):
        """
        Enhanced fitting with PCA alignment and nonlinear mapping.
        """
        super().fit(neural_coords, physical_coords)
        
        # If using PCA alignment, enhance the transformation
        if self.use_pca_alignment and physical_coords is not None:
            self._enhance_with_pca(neural_coords, physical_coords)
    
    def _enhance_with_pca(self, neural_coords, physical_coords):
        """
        Enhance the mapping using PCA to better align with physical structure.
        """
        N, K, D = neural_coords.shape
        N_phy, K_phy, D_phy = physical_coords.shape
        
        # Reshape to flat arrays
        neural_flat = neural_coords.reshape(-1, D)
        physical_flat = physical_coords.reshape(-1, D_phy)
        
        # Apply the base transformation
        neural_scaled = self.neural_scaler.transform(neural_flat)
        physical_est = neural_scaled @ self.linear_weights.T + self.linear_bias
        
        # Compute residuals
        residuals = physical_flat - physical_est
        
        # Use PCA on residuals to find additional alignment directions
        if np.linalg.matrix_rank(neural_scaled) > 0:
            # Perform SVD to find optimal linear correction
            U, s, Vt = np.linalg.svd(neural_scaled.T @ residuals, full_matrices=False)
            # Update transformation with correction
            correction = Vt.T @ U.T
            self.linear_weights += 0.1 * correction  # Small learning rate for stability


def create_enhanced_coord_mapper(n_latent_dims, n_physical_dims=2):
    """
    Factory function to create an enhanced coordinate mapper.
    """
    return EnhancedCoordinateMapper(
        n_latent_dims=n_latent_dims,
        n_physical_dims=n_physical_dims,
        use_rotation_alignment=True,
        use_nonlinear_mapping=True,
        use_pca_alignment=True
    )