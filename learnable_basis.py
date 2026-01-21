"""
Learnable Basis Functions Module to address the basis function bottleneck.
This module enables the discovery of novel functional forms beyond the predefined library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sympy as sp
from symbolic import gp_to_sympy


class LearnableBasisFunction(nn.Module):
    """
    A learnable basis function that can represent novel functional forms.
    Uses a neural network to learn complex transformations of input features.
    Enhanced with attention mechanism and residual connections for better expressivity.
    """

    def __init__(self, input_dim, hidden_dim=64, num_bases=8, activation='silu', use_attention=True, use_residual=True):
        """
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden dimension for the neural basis
            num_bases: Number of different basis functions to learn
            activation: Activation function to use
            use_attention: Whether to use attention mechanism
            use_residual: Whether to use residual connections
        """
        super(LearnableBasisFunction, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_bases = num_bases
        self.use_attention = use_attention
        self.use_residual = use_residual

        # Multiple basis networks to learn different functional forms
        self.basis_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self._get_activation(activation),
                nn.Linear(hidden_dim, hidden_dim),
                self._get_activation(activation),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_bases)
        ])

        # Learnable mixing weights for combining basis functions
        self.mixing_weights = nn.Parameter(torch.ones(num_bases) / num_bases)

        # Learnable scaling factors for each basis
        self.scaling_factors = nn.Parameter(torch.ones(num_bases))

        # Attention mechanism for dynamic basis selection
        if use_attention:
            self.attention_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim // 2),
                self._get_activation(activation),
                nn.Linear(hidden_dim // 2, num_bases),
                nn.Softmax(dim=1)
            )

        # Residual connection scaling
        if use_residual:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _get_activation(self, activation):
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(),
            'swish': nn.SiLU(),  # Swish is the same as SiLU
            'elu': nn.ELU()
        }
        return activations.get(activation.lower(), nn.SiLU())

    def forward(self, x):
        """
        Forward pass through learnable basis functions.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            Combined output of all basis functions
        """
        batch_size = x.size(0)

        # Apply each basis network
        basis_outputs = []
        for i, basis_net in enumerate(self.basis_networks):
            output = basis_net(x)  # [batch_size, 1]
            # Apply scaling and store
            scaled_output = output * self.scaling_factors[i]
            basis_outputs.append(scaled_output)

        # Stack basis outputs: [batch_size, num_bases]
        stacked_outputs = torch.cat(basis_outputs, dim=1)

        # Apply mixing weights: [num_bases] * [batch_size, num_bases] -> [batch_size, num_bases]
        if self.use_attention:
            # Use attention to dynamically weight basis functions based on input
            attention_weights = self.attention_net(x)  # [batch_size, num_bases]
            weighted_outputs = stacked_outputs * attention_weights
        else:
            # Use learnable mixing weights
            mixing_weights = F.softmax(self.mixing_weights, dim=0)
            weighted_outputs = stacked_outputs * mixing_weights

        # Sum across all basis functions: [batch_size, 1]
        final_output = torch.sum(weighted_outputs, dim=1, keepdim=True)

        # Add residual connection if enabled
        if self.use_residual:
            # Project input to output dimension
            residual_input = torch.mean(x, dim=1, keepdim=True)  # Simple projection
            final_output = final_output + self.residual_scale * residual_input

        return final_output


class NeuralBasisExpansion(nn.Module):
    """
    Neural basis expansion that combines traditional physics-inspired features
    with learnable basis functions to discover novel functional forms.
    Enhanced with attention mechanisms and adaptive feature selection.
    """

    def __init__(self, n_super_nodes, latent_dim, include_dists=True, box_size=None,
                 basis_hidden_dim=64, num_learnable_bases=8, use_attention=True,
                 use_adaptive_selection=True):
        super(NeuralBasisExpansion, self).__init__()

        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.include_dists = include_dists
        self.box_size = box_size
        self.use_attention = use_attention
        self.use_adaptive_selection = use_adaptive_selection

        # Traditional physics-inspired features (same as in FeatureTransformer)
        self.z_mean = None
        self.z_std = None
        self.x_poly_mean = None
        self.x_poly_std = None
        self.target_mean = None
        self.target_std = None

        # Learnable basis function modules
        # We'll create a basis for different types of input combinations
        # For pairwise features: pos_i(2) + vel_i(2) + pos_j(2) + vel_j(2) = 8 dimensions if latent_dim >= 4
        # If latent_dim < 4, adjust accordingly
        pairwise_input_dim = min(8, max(4, latent_dim * 2))  # At least 4, max of 8 for pos/vel of 2 nodes
        self.pairwise_basis = LearnableBasisFunction(
            pairwise_input_dim,
            basis_hidden_dim,
            num_learnable_bases,
            use_attention=use_attention
        )  # For distance-based features
        self.node_basis = LearnableBasisFunction(
            latent_dim,
            basis_hidden_dim,
            num_learnable_bases,
            use_attention=use_attention
        )  # For individual node features
        self.global_basis = LearnableBasisFunction(
            latent_dim * n_super_nodes,
            basis_hidden_dim,
            num_learnable_bases,
            use_attention=use_attention
        )  # For global features

        # Learnable combination weights for different feature types
        self.feature_combination_weights = nn.Parameter(torch.ones(3) / 3)  # For traditional, pairwise, node features

        # Adaptive feature selection network
        if use_adaptive_selection:
            self.feature_selector = nn.Sequential(
                nn.Linear(n_super_nodes * latent_dim, basis_hidden_dim),
                nn.SiLU(),
                nn.Linear(basis_hidden_dim, basis_hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(basis_hidden_dim // 2, 1),
                nn.Sigmoid()
            )

    def fit(self, latent_states, targets):
        """Fit normalization parameters."""
        self.z_mean = torch.from_numpy(latent_states.mean(axis=0)).float()
        self.z_std = torch.from_numpy(latent_states.std(axis=0) + 1e-6).float()

        # Transform to get feature statistics
        X_poly = self.transform(torch.from_numpy(latent_states).float())
        self.x_poly_mean = X_poly.mean(dim=0)
        self.x_poly_std = X_poly.std(dim=0) + 1e-6
        self.target_mean = torch.from_numpy(targets.mean(axis=0)).float()
        self.target_std = torch.from_numpy(targets.std(axis=0) + 1e-6).float()

    def transform(self, z_flat):
        """
        Transform latent states to expanded feature space with learnable basis functions.

        Args:
            z_flat: Flattened latent states [batch_size, n_super_nodes * latent_dim]

        Returns:
            Expanded feature tensor
        """
        batch_size = z_flat.size(0)
        z_nodes = z_flat.view(batch_size, self.n_super_nodes, self.latent_dim)

        # Traditional features (same as original FeatureTransformer)
        traditional_features = [z_flat]

        if self.include_dists:
            # Vectorized distance calculation
            pos = z_nodes[:, :, :2]  # [Batch, K, 2]
            diffs = pos[:, :, None, :] - pos[:, None, :, :]  # [Batch, K, K, 2]

            if self.box_size is not None:
                box = torch.tensor(self.box_size, device=z_flat.device, dtype=z_flat.dtype)
                diffs = diffs - box * torch.round(diffs / box)

            # Compute norms [Batch, K, K]
            dists_matrix = torch.norm(diffs, dim=-1)

            # Extract upper triangle indices (i < j)
            i_idx, j_idx = torch.triu_indices(self.n_super_nodes, self.n_super_nodes, offset=1, device=z_flat.device)

            dists_flat = dists_matrix[:, i_idx, j_idx]  # [Batch, K*(K-1)/2]
            dists_flat = torch.clamp(dists_flat, min=1e-3)

            inv_dists_flat = 1.0 / (dists_flat + 0.1)
            inv_sq_dists_flat = 1.0 / (dists_flat**2 + 0.1)

            # New physics-informed basis functions
            exp_dist = torch.exp(-torch.clamp(dists_flat, max=20.0))
            screened_coulomb = exp_dist / (dists_flat + 0.1)
            log_dist = torch.log(dists_flat + 1.0)

            traditional_features.extend([dists_flat, inv_dists_flat, inv_sq_dists_flat, exp_dist, screened_coulomb, log_dist])

        X_traditional = torch.cat(traditional_features, dim=1)

        # Learnable basis features
        learnable_features = []

        # 1. Pairwise learnable features (based on distances and relative positions)
        if self.include_dists:
            pos = z_nodes[:, :, :2]  # [Batch, K, 2]
            vel = z_nodes[:, :, 2:] if self.latent_dim > 2 else torch.zeros_like(pos)  # [Batch, K, 2+]

            # Process all pairwise combinations at once for efficiency
            pairwise_features_list = []
            for i in range(self.n_super_nodes):
                for j in range(i + 1, self.n_super_nodes):
                    # Concatenate features of node i and node j: [pos_i, vel_i, pos_j, vel_j]
                    pair_features = torch.cat([
                        pos[:, i, :], vel[:, i, :],
                        pos[:, j, :], vel[:, j, :]
                    ], dim=1)  # [Batch, 8] or [Batch, 4] depending on latent_dim

                    # Process through the pairwise basis function
                    processed = self.pairwise_basis(pair_features)  # [Batch, 1]
                    pairwise_features_list.append(processed)

            if pairwise_features_list:
                # Concatenate all processed pairwise features: [Batch, n_pairs]
                pairwise_features = torch.cat(pairwise_features_list, dim=1)
                learnable_features.append(pairwise_features)

        # 2. Node-wise learnable features
        node_features_list = []
        for k in range(self.n_super_nodes):
            node_data = z_nodes[:, k, :]  # [Batch, latent_dim]
            processed = self.node_basis(node_data)  # [Batch, 1]
            node_features_list.append(processed)

        if node_features_list:
            node_features = torch.cat(node_features_list, dim=1)  # [Batch, n_super_nodes]
            learnable_features.append(node_features)

        # 3. Global learnable features
        global_features = self.global_basis(z_flat)  # [Batch, 1]
        learnable_features.append(global_features)

        # Combine traditional and learnable features
        if learnable_features:
            X_learnable = torch.cat(learnable_features, dim=1)
            X_combined = torch.cat([X_traditional, X_learnable], dim=1)
        else:
            X_combined = X_traditional

        # Apply adaptive feature selection if enabled
        if self.use_adaptive_selection:
            # Use the feature selector to determine which features to emphasize
            selection_weights = self.feature_selector(z_flat)  # [batch_size, 1]
            # Expand to match feature dimensions
            selection_weights = selection_weights.expand_as(X_combined)
            X_combined = X_combined * selection_weights

        # Apply learnable combination weights
        comb_weights = F.softmax(self.feature_combination_weights, dim=0)

        return torch.clamp(X_combined, -1e6, 1e6)  # Final safety clamp
    
    def normalize_x(self, X_poly):
        return (X_poly - self.x_poly_mean) / self.x_poly_std

    def normalize_y(self, Y):
        return (Y - self.target_mean) / self.target_std

    def denormalize_y(self, Y_norm):
        return Y_norm * self.target_std + self.target_mean


class EnhancedFeatureTransformer:
    """
    Enhanced feature transformer that combines traditional physics-inspired features
    with learnable basis functions to address the basis function bottleneck.
    Enhanced with attention mechanisms and adaptive feature selection.
    """

    def __init__(self, n_super_nodes, latent_dim, include_dists=True, box_size=None,
                 use_learnable_bases=True, basis_hidden_dim=64, num_learnable_bases=8,
                 use_attention=True, use_adaptive_selection=True):
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.include_dists = include_dists
        self.box_size = box_size
        self.use_learnable_bases = use_learnable_bases

        if use_learnable_bases:
            self.neural_basis = NeuralBasisExpansion(
                n_super_nodes, latent_dim, include_dists, box_size,
                basis_hidden_dim, num_learnable_bases, use_attention, use_adaptive_selection
            )
        else:
            # Fall back to traditional FeatureTransformer
            from symbolic import FeatureTransformer
            self.traditional_transformer = FeatureTransformer(n_super_nodes, latent_dim, include_dists, box_size)

    def fit(self, latent_states, targets):
        if self.use_learnable_bases:
            self.neural_basis.fit(latent_states, targets)
        else:
            self.traditional_transformer.fit(latent_states, targets)

    def transform(self, z_flat):
        if self.use_learnable_bases:
            if not isinstance(z_flat, torch.Tensor):
                z_flat = torch.from_numpy(z_flat).float()
            return self.neural_basis.transform(z_flat).detach().numpy()
        else:
            return self.traditional_transformer.transform(z_flat)

    def normalize_x(self, X_poly):
        if self.use_learnable_bases:
            return self.neural_basis.normalize_x(torch.from_numpy(X_poly).float()).numpy()
        else:
            return self.traditional_transformer.normalize_x(X_poly)

    def normalize_y(self, Y):
        if self.use_learnable_bases:
            return self.neural_basis.normalize_y(torch.from_numpy(Y).float()).numpy()
        else:
            return self.traditional_transformer.normalize_y(Y)

    def denormalize_y(self, Y_norm):
        if self.use_learnable_bases:
            return self.neural_basis.denormalize_y(torch.from_numpy(Y_norm).float()).numpy()
        else:
            return self.traditional_transformer.denormalize_y(Y_norm)

    def get_n_features(self):
        """Get the number of features after transformation."""
        if self.use_learnable_bases:
            # This is a simplified version - in practice, you'd need to track this during transformation
            # For now, return an estimate based on the components
            n_traditional = self.n_super_nodes * self.latent_dim
            if self.include_dists:
                n_traditional += self.n_super_nodes * (self.n_super_nodes - 1) // 2 * 6  # 6 distance features
            n_learnable = self.n_super_nodes + 1  # node features + global feature
            return n_traditional + n_learnable
        else:
            # This would require knowing the actual number after transformation
            # For now, return a placeholder - in practice you'd need to store this during fit
            return self.traditional_transformer.get_n_features() if hasattr(self.traditional_transformer, 'get_n_features') else 100