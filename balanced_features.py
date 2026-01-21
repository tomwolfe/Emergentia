"""
Enhanced FeatureTransformer that balances domain knowledge with pure discovery capabilities.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class BalancedFeatureTransformer:
    """
    Enhanced FeatureTransformer that balances domain knowledge with pure discovery capabilities.
    
    Key improvements:
    1. Configurable basis functions to allow for both physics-inspired and generic features
    2. Automatic feature selection to reduce dimensionality and noise
    3. Robust normalization techniques
    4. Support for different types of physical laws beyond inverse-square
    """
    
    def __init__(self, n_super_nodes, latent_dim, include_dists=True, box_size=None, 
                 basis_functions='physics_informed', max_degree=2, feature_selection_method='mutual_info'):
        """
        Initialize the balanced feature transformer.

        Args:
            n_super_nodes (int): Number of super-nodes
            latent_dim (int): Dimension of each latent node
            include_dists (bool): Whether to include distance-based features
            box_size (tuple): Box size for periodic boundary conditions
            basis_functions (str): Type of basis functions ('physics_informed', 'polynomial', 'adaptive')
            max_degree (int): Maximum degree for polynomial features
            feature_selection_method (str): Method for feature selection ('mutual_info', 'f_test', 'lasso')
        """
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.include_dists = include_dists
        self.box_size = box_size
        self.basis_functions = basis_functions
        self.max_degree = max_degree
        self.feature_selection_method = feature_selection_method
        
        # Normalization parameters
        self.z_mean = None
        self.z_std = None
        self.x_poly_mean = None
        self.x_poly_std = None
        self.target_mean = None
        self.target_std = None
        
        # Feature selection parameters
        self.selected_feature_indices = None
        self.n_selected_features = None
        
        # For polynomial features
        self.poly_transformer = None

    def fit(self, latent_states, targets):
        """
        Fit the transformer to the data.

        Args:
            latent_states: [N, n_super_nodes * latent_dim] - Latent state vectors
            targets: [N, n_super_nodes * latent_dim] - Target derivatives or values
        """
        # 1. Fit raw latent normalization
        self.z_mean = latent_states.mean(axis=0)
        self.z_std = latent_states.std(axis=0) + 1e-6

        # 2. Transform to poly features
        X_poly = self.transform(latent_states, fit_transformer=True)

        # 3. Fit poly feature normalization
        self.x_poly_mean = X_poly.mean(axis=0)
        self.x_poly_std = X_poly.std(axis=0) + 1e-6

        # 4. Fit target normalization
        if targets.ndim == 1:
            self.target_mean = targets.mean()
            self.target_std = targets.std() + 1e-6
        else:
            self.target_mean = targets.mean(axis=0)
            self.target_std = targets.std(axis=0) + 1e-6

        # 5. Perform feature selection
        self._perform_feature_selection(X_poly, targets)

    def transform(self, z_flat, fit_transformer=False):
        """
        Transform latent states to polynomial features.
        
        Args:
            z_flat: [Batch, n_super_nodes * latent_dim] - Flattened latent states
            fit_transformer: Whether to fit polynomial transformers (only during fitting)
        """
        # z_flat: [Batch, n_super_nodes * latent_dim]
        z_nodes = z_flat.reshape(-1, self.n_super_nodes, self.latent_dim)

        features = [z_flat]
        
        if self.include_dists and self.basis_functions != 'polynomial_only':
            dist_features = self._compute_distance_features(z_nodes)
            if dist_features:
                features.extend(dist_features)

        X = np.hstack(features)

        # Apply basis function expansion based on configuration
        if self.basis_functions == 'polynomial':
            X_expanded = self._polynomial_expansion(X, fit_transformer)
        elif self.basis_functions == 'physics_informed':
            X_expanded = self._physics_informed_expansion(X)
        elif self.basis_functions == 'adaptive':
            X_expanded = self._adaptive_expansion(X, fit_transformer)
        else:
            # Default to physics-informed
            X_expanded = self._physics_informed_expansion(X)

        return X_expanded

    def _compute_distance_features(self, z_nodes):
        """
        Compute distance-based features between super-nodes.
        """
        dists = []
        inv_dists = []
        inv_sq_dists = []
        exp_dists = []  # Additional: exponential decay features

        for i in range(self.n_super_nodes):
            for j in range(i + 1, self.n_super_nodes):
                # Relative distance between super-nodes (using first 2 dims as positions)
                diff = z_nodes[:, i, :2] - z_nodes[:, j, :2]

                # Apply Minimum Image Convention for PBC if box_size is provided
                if self.box_size is not None:
                    for dim_idx in range(2):  # Assuming 2D for position coordinates
                        diff[:, dim_idx] -= self.box_size[dim_idx] * np.round(diff[:, dim_idx] / self.box_size[dim_idx])

                d = np.linalg.norm(diff, axis=1, keepdims=True)
                
                # Standard distance features
                dists.append(d)
                
                # Physics-informed features: inverse laws
                inv_dists.append(1.0 / (d + 0.1))
                inv_sq_dists.append(1.0 / (d**2 + 0.1))
                
                # Additional: exponential decay features (for different interaction types)
                exp_dists.append(np.exp(-d))

        if dists:
            return [np.hstack(dists), np.hstack(inv_dists), np.hstack(inv_sq_dists), np.hstack(exp_dists)]
        else:
            return []

    def _polynomial_expansion(self, X, fit_transformer):
        """
        Perform standard polynomial expansion.
        """
        if fit_transformer or self.poly_transformer is None:
            self.poly_transformer = PolynomialFeatures(
                degree=self.max_degree, 
                include_bias=False, 
                interaction_only=False
            )
            X_poly = self.poly_transformer.fit_transform(X)
        else:
            X_poly = self.poly_transformer.transform(X)
        
        return X_poly

    def _physics_informed_expansion(self, X):
        """
        Perform physics-informed polynomial expansion with cross-terms.
        """
        poly_features = [X]
        n_latents = self.n_super_nodes * self.latent_dim

        # Squares of latent variables
        for i in range(n_latents):
            poly_features.append((X[:, i:i+1]**2))

            node_idx = i // self.latent_dim
            dim_idx = i % self.latent_dim

            # Cross-terms within same node (e.g. q1*p1)
            for j in range(i + 1, (node_idx + 1) * self.latent_dim):
                poly_features.append(X[:, i:i+1] * X[:, j:j+1])

            # Cross-terms with same dimension in other nodes (e.g. q1*q2)
            for other_node in range(node_idx + 1, self.n_super_nodes):
                other_idx = other_node * self.latent_dim + dim_idx
                poly_features.append(X[:, i:i+1] * X[:, other_idx:other_idx+1])

        return np.hstack(poly_features)

    def _adaptive_expansion(self, X, fit_transformer):
        """
        Adaptive expansion that learns the most relevant feature combinations.
        """
        # Start with physics-informed features
        X_physics = self._physics_informed_expansion(X)
        
        # If we're fitting, we can try to learn which features are most predictive
        if fit_transformer and hasattr(self, '_targets_for_adaptation'):
            # Use the targets to guide feature selection
            from sklearn.feature_selection import SelectKBest, f_regression
            
            # Select top features based on their relationship with targets
            selector = SelectKBest(score_func=f_regression, k=min(500, X_physics.shape[1]//2))
            X_selected = selector.fit_transform(X_physics, self._targets_for_adaptation)
            self.adaptive_selector = selector
            return X_selected
        elif hasattr(self, 'adaptive_selector'):
            # Use previously fitted selector
            return self.adaptive_selector.transform(X_physics)
        else:
            # Fall back to physics-informed if we can't adapt
            return self._physics_informed_expansion(X)

    def _perform_feature_selection(self, X, y):
        """
        Perform feature selection to reduce dimensionality and noise.
        """
        n_features = X.shape[1]
        
        # Determine how many features to select (at most 20% of original or 1000, whichever is smaller)
        max_features = min(int(0.2 * n_features), 1000, n_features)
        
        if max_features >= n_features:
            # No selection needed
            self.selected_feature_indices = np.arange(n_features)
            self.n_selected_features = n_features
            return

        if self.feature_selection_method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=max_features)
        elif self.feature_selection_method == 'f_test':
            selector = SelectKBest(score_func=f_regression, k=max_features)
        elif self.feature_selection_method == 'lasso':
            # Use LassoCV for feature selection
            lasso = LassoCV(cv=3, max_iter=3000)
            lasso.fit(X, y.ravel() if y.ndim > 1 else y)
            # Select features with non-zero coefficients
            coef_mask = np.abs(lasso.coef_) > 1e-5
            selected_indices = np.where(coef_mask)[0]
            # If too many features selected, use top-k
            if len(selected_indices) > max_features:
                scores = np.abs(lasso.coef_)
                top_k_indices = np.argsort(scores)[-max_features:]
                self.selected_feature_indices = top_k_indices
            else:
                self.selected_feature_indices = selected_indices
            self.n_selected_features = len(self.selected_feature_indices)
            return
        else:
            # Default to mutual info
            selector = SelectKBest(score_func=mutual_info_regression, k=max_features)

        selector.fit(X, y.ravel() if y.ndim > 1 else y)
        self.selected_feature_indices = selector.get_support(indices=True)
        self.n_selected_features = len(self.selected_feature_indices)

    def normalize_x(self, X_poly):
        """
        Normalize features using fitted statistics.
        """
        # Only normalize the selected features
        X_selected = X_poly[:, self.selected_feature_indices]
        return (X_selected - self.x_poly_mean[self.selected_feature_indices]) / self.x_poly_std[self.selected_feature_indices]

    def normalize_y(self, Y):
        """
        Normalize targets using fitted statistics.
        """
        return (Y - self.target_mean) / self.target_std

    def denormalize_y(self, Y_norm):
        """
        Denormalize targets using fitted statistics.
        """
        return Y_norm * self.target_std + self.target_mean

    def get_n_features(self):
        """
        Get the number of features after selection.
        """
        return self.n_selected_features if self.n_selected_features is not None else self.x_poly_mean.shape[0]


class AdaptiveFeatureTransformer(BalancedFeatureTransformer):
    """
    Even more adaptive version that can adjust its feature generation based on the data.
    """
    
    def __init__(self, n_super_nodes, latent_dim, include_dists=True, box_size=None, 
                 basis_functions='adaptive', max_degree=2, feature_selection_method='mutual_info'):
        super().__init__(n_super_nodes, latent_dim, include_dists, box_size, 
                         basis_functions, max_degree, feature_selection_method)
        self.interaction_types = ['inverse', 'exponential', 'gaussian', 'power_law']
        self.best_interaction_type = 'inverse'  # Will be determined during fitting

    def _compute_distance_features(self, z_nodes):
        """
        Compute distance-based features with multiple interaction types.
        """
        all_dist_features = []
        
        for i in range(self.n_super_nodes):
            for j in range(i + 1, self.n_super_nodes):
                # Relative distance between super-nodes
                diff = z_nodes[:, i, :2] - z_nodes[:, j, :2]

                # Apply Minimum Image Convention for PBC if box_size is provided
                if self.box_size is not None:
                    for dim_idx in range(2):
                        diff[:, dim_idx] -= self.box_size[dim_idx] * np.round(diff[:, dim_idx] / self.box_size[dim_idx])

                d = np.linalg.norm(diff, axis=1, keepdims=True)
                
                # Generate features based on the best interaction type
                if self.best_interaction_type == 'inverse':
                    all_dist_features.extend([
                        d,
                        1.0 / (d + 0.1),
                        1.0 / (d**2 + 0.1)
                    ])
                elif self.best_interaction_type == 'exponential':
                    all_dist_features.extend([
                        d,
                        np.exp(-d),
                        np.exp(-d**2)
                    ])
                elif self.best_interaction_type == 'gaussian':
                    all_dist_features.extend([
                        d,
                        np.exp(-0.5 * d**2),
                        np.exp(-0.5 * d)
                    ])
                elif self.best_interaction_type == 'power_law':
                    all_dist_features.extend([
                        d,
                        d**(-1.5),
                        d**(-2.5)
                    ])
                else:  # Default to inverse
                    all_dist_features.extend([
                        d,
                        1.0 / (d + 0.1),
                        1.0 / (d**2 + 0.1)
                    ])

        if all_dist_features:
            return [np.hstack(all_dist_features)]
        else:
            return []