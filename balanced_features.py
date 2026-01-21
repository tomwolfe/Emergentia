"""
Enhanced FeatureTransformer that balances domain knowledge with pure discovery capabilities.
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class RecursiveFeatureSelector:
    """
    Selects features recursively by iteratively adding the most informative ones.
    This prevents the combinatorial explosion of polynomial terms.
    """
    def __init__(self, max_features=100, tolerance=1e-4):
        self.max_features = max_features
        self.tolerance = tolerance
        self.selected_indices = []

    def fit(self, X, y):
        from sklearn.linear_model import OrthogonalMatchingPursuit
        
        n_samples, n_features = X.shape
        n_to_select = min(self.max_features, n_features, n_samples)
        
        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_to_select, tol=self.tolerance)
        try:
            omp.fit(X, y.ravel() if y.ndim > 1 else y)
            self.selected_indices = np.where(omp.coef_ != 0)[0]
        except:
            # Fallback to a simpler selection if OMP fails
            from sklearn.feature_selection import f_regression
            scores, _ = f_regression(X, y.ravel() if y.ndim > 1 else y)
            self.selected_indices = np.argsort(scores)[-n_to_select:]
            
        return self

    def transform(self, X):
        return X[:, self.selected_indices]


class BalancedFeatureTransformer:
    """
    Enhanced FeatureTransformer that balances domain knowledge with pure discovery capabilities.
    
    Key improvements:
    1. Configurable basis functions to allow for both physics-inspired and generic features
    2. Automatic feature selection to reduce dimensionality and noise
    3. Robust normalization techniques
    4. Support for different types of physical laws beyond inverse-square
    5. Recursive feature selection to prevent combinatorial explosion
    """
    
    def __init__(self, n_super_nodes, latent_dim, include_dists=True, box_size=None, 
                 basis_functions='physics_informed', max_degree=2, feature_selection_method='recursive'):
        """
        Initialize the balanced feature transformer.

        Args:
            n_super_nodes (int): Number of super-nodes
            latent_dim (int): Dimension of each latent node
            include_dists (bool): Whether to include distance-based features
            box_size (tuple): Box size for periodic boundary conditions
            basis_functions (str): Type of basis functions ('physics_informed', 'polynomial', 'adaptive')
            max_degree (int): Maximum degree for polynomial features
            feature_selection_method (str): Method for feature selection ('recursive', 'mutual_info', 'f_test', 'lasso')
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
        self.recursive_selector = None
        
        # For polynomial features
        self.poly_transformer = None

    def fit(self, latent_states, targets):
        """
        Fit the transformer to the data.
        """
        # 1. Fit raw latent normalization
        self.z_mean = latent_states.mean(axis=0)
        self.z_std = latent_states.std(axis=0) + 1e-6

        # 2. Transform to poly features
        X_poly = self.transform(latent_states, fit_transformer=True)

        # 3. Perform feature selection BEFORE fitting poly normalization to save memory/time
        self._perform_feature_selection(X_poly, targets)
        
        # 4. Fit poly feature normalization on SELECTED features
        X_selected = X_poly[:, self.selected_feature_indices]
        self.x_poly_mean = X_selected.mean(axis=0)
        self.x_poly_std = X_selected.std(axis=0) + 1e-6

        # 5. Fit target normalization
        if targets.ndim == 1:
            self.target_mean = targets.mean()
            self.target_std = targets.std() + 1e-6
        else:
            self.target_mean = targets.mean(axis=0)
            self.target_std = targets.std(axis=0) + 1e-6

    def transform(self, z_flat, fit_transformer=False):
        """
        Transform latent states to polynomial features.
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
            X_expanded = self._physics_informed_expansion(X)

        return X_expanded

    def _compute_distance_features(self, z_nodes):
        """
        Compute distance-based features between super-nodes.
        """
        dists = []
        inv_dists = []
        inv_sq_dists = []
        exp_dists = []

        for i in range(self.n_super_nodes):
            for j in range(i + 1, self.n_super_nodes):
                diff = z_nodes[:, i, :2] - z_nodes[:, j, :2]

                if self.box_size is not None:
                    for dim_idx in range(2):
                        diff[:, dim_idx] -= self.box_size[dim_idx] * np.round(diff[:, dim_idx] / self.box_size[dim_idx])

                d = np.linalg.norm(diff, axis=1, keepdims=True)
                dists.append(d)
                inv_dists.append(1.0 / (d + 0.1))
                inv_sq_dists.append(1.0 / (d**2 + 0.1))
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
        Optimized to avoid unnecessary terms.
        """
        poly_features = [X]
        n_latents = self.n_super_nodes * self.latent_dim
        
        # Limit cross-terms to avoid explosion
        # Only squares and interaction terms that make physical sense
        for i in range(n_latents):
            poly_features.append((X[:, i:i+1]**2))

            node_idx = i // self.latent_dim
            dim_idx = i % self.latent_dim

            # Cross-terms within same node (e.g. q1*p1)
            for j in range(i + 1, (node_idx + 1) * self.latent_dim):
                poly_features.append(X[:, i:i+1] * X[:, j:j+1])

            # Interaction terms between nodes for the same dimension
            for other_node in range(node_idx + 1, self.n_super_nodes):
                other_idx = other_node * self.latent_dim + dim_idx
                poly_features.append(X[:, i:i+1] * X[:, other_idx:other_idx+1])

        return np.hstack(poly_features)

    def _adaptive_expansion(self, X, fit_transformer):
        """
        Adaptive expansion that learns the most relevant feature combinations.
        """
        X_physics = self._physics_informed_expansion(X)
        
        if fit_transformer and hasattr(self, '_targets_for_adaptation'):
            selector = SelectKBest(score_func=f_regression, k=min(200, X_physics.shape[1]))
            X_selected = selector.fit_transform(X_physics, self._targets_for_adaptation)
            self.adaptive_selector = selector
            return X_selected
        elif hasattr(self, 'adaptive_selector'):
            return self.adaptive_selector.transform(X_physics)
        else:
            return X_physics

    def _perform_feature_selection(self, X, y):
        """
        Perform feature selection to reduce dimensionality and noise.
        """
        n_features = X.shape[1]
        
        if self.feature_selection_method == 'recursive':
            self.recursive_selector = RecursiveFeatureSelector(max_features=min(200, n_features))
            self.recursive_selector.fit(X, y)
            self.selected_feature_indices = self.recursive_selector.selected_indices
        elif self.feature_selection_method == 'lasso':
            lasso = LassoCV(cv=3, max_iter=2000)
            lasso.fit(X, y.ravel() if y.ndim > 1 else y)
            self.selected_feature_indices = np.where(np.abs(lasso.coef_) > 1e-5)[0]
        else:
            # Fallback to mutual info
            max_f = min(200, n_features)
            selector = SelectKBest(score_func=f_regression, k=max_f)
            selector.fit(X, y.ravel() if y.ndim > 1 else y)
            self.selected_feature_indices = selector.get_support(indices=True)
            
        self.n_selected_features = len(self.selected_feature_indices)

    def normalize_x(self, X_poly):
        """
        Normalize features using fitted statistics.
        """
        X_selected = X_poly[:, self.selected_feature_indices]
        return (X_selected - self.x_poly_mean) / self.x_poly_std

    def normalize_y(self, Y):
        return (Y - self.target_mean) / self.target_std

    def denormalize_y(self, Y_norm):
        return Y_norm * self.target_std + self.target_mean

    def get_n_features(self):
        return self.n_selected_features if self.n_selected_features is not None else 0


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