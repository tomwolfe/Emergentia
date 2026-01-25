"""
Enhanced FeatureTransformer that balances domain knowledge with pure discovery capabilities.
"""

import numpy as np
import re
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV, ElasticNetCV, LassoLarsIC
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')


class BICFeatureSelector:
    """
    Selects features using Lasso with Bayesian Information Criterion (BIC).
    BIC penalizes complexity more harshly than AIC or CV, leading to more parsimonious models.
    """
    def __init__(self, max_features=40, min_variance=1e-6):
        self.max_features = max_features
        self.min_variance = min_variance
        self.selected_indices = []

    def fit(self, X, y, feature_names=None):
        # 0. Clean input data
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)

        # 1. Variance filtering
        variances = np.var(X, axis=0)
        high_variance_indices = np.where(variances > self.min_variance)[0]
        
        if len(high_variance_indices) == 0:
            self.selected_indices = np.array([], dtype=int)
            return self
            
        X_filtered = X[:, high_variance_indices]
        filtered_names = [feature_names[i] for i in high_variance_indices] if feature_names else None
        
        # 2. Hierarchical Basis Pruning: Only allow higher-order power laws if needed
        if filtered_names:
            # Group indices by power
            power_groups = {} # power -> list of indices in X_filtered
            other_indices = []
            
            for i, name in enumerate(filtered_names):
                match = re.search(r'sum_inv_d(\d+)', name)
                if match:
                    p = int(match.group(1))
                    if p not in power_groups: power_groups[p] = []
                    power_groups[p].append(i)
                elif 'sum_inv_d' in name: # case for 'sum_inv_d' which is d1
                    if 1 not in power_groups: power_groups[1] = []
                    power_groups[1].append(i)
                else:
                    other_indices.append(i)
            
            if power_groups:
                allowed_indices = list(other_indices)
                sorted_powers = sorted(power_groups.keys())
                
                # Baseline fit check for each target
                n_outputs = y.shape[1] if y.ndim > 1 else 1
                for i in range(n_outputs):
                    yi = y[:, i] if y.ndim > 1 else y
                    current_allowed = list(allowed_indices)
                    
                    for p in sorted_powers:
                        current_allowed.extend(power_groups[p])
                        # Check if current set achieves a baseline fit (R2 > 0.8)
                        from sklearn.linear_model import Ridge
                        model = Ridge(alpha=1e-3)
                        model.fit(X_filtered[:, current_allowed], yi)
                        if model.score(X_filtered[:, current_allowed], yi) > 0.8:
                            # If fit is good enough, we stop adding higher powers for this target
                            break
                
                # The union of all allowed indices across targets
                # (Simplified: we'll just use the ones allowed for the first target or all)
                # To be safer and truly hierarchical, we limit the search space
                final_allowed = set(other_indices)
                for p in sorted_powers:
                    final_allowed.update(power_groups[p])
                    # If we have enough features, stop
                    if len(final_allowed) > self.max_features * 2:
                        break
                
                X_filtered = X_filtered[:, sorted(list(final_allowed))]
                # Map back to original filtered indices
                mapping = sorted(list(final_allowed))
            else:
                mapping = np.arange(X_filtered.shape[1])
        else:
            mapping = np.arange(X_filtered.shape[1])

        # 3. Use LassoLarsIC with BIC
        try:
            # We select features for each output dimension and union them
            union_indices = set()
            n_outputs = y.shape[1] if y.ndim > 1 else 1
            n_samples, n_features = X_filtered.shape
            
            for i in range(n_outputs):
                yi = y[:, i] if y.ndim > 1 else y
                
                # ALWAYS provide a shrinkage-based noise variance estimate using Ridge baseline.
                # This ensures stability even when N is not much larger than P or when features are correlated.
                from sklearn.linear_model import Ridge
                ridge = Ridge(alpha=1.0).fit(X_filtered, yi)
                mse = np.mean((yi - ridge.predict(X_filtered))**2)
                noise_variance = max(mse, 1e-6)
                
                # LassoLarsIC is very fast and provides a clear BIC path
                model = LassoLarsIC(criterion='bic', max_iter=500, noise_variance=noise_variance)
                model.fit(X_filtered, yi)
                
                # Get indices of non-zero coefficients
                nonzero = np.where(np.abs(model.coef_) > 1e-10)[0]
                union_indices.update([mapping[idx] for idx in nonzero])
            
            selected_filtered_indices = np.array(list(union_indices), dtype=int)
            
            if len(selected_filtered_indices) > self.max_features:
                # If too many, take those with largest combined importance
                importance = np.zeros(X_filtered.shape[1])
                for i in range(n_outputs):
                    yi = y[:, i] if y.ndim > 1 else y
                    # Use a simple Ridge for importance instead of LassoLarsIC if it's too slow
                    from sklearn.linear_model import Ridge
                    model = Ridge(alpha=1e-3)
                    model.fit(X_filtered, yi)
                    importance += np.abs(model.coef_)
                
                # Take top indices from the mapping (indices in X_filtered)
                top_mapped_indices = np.argsort(importance)[-self.max_features:]
                selected_filtered_indices = np.array([mapping[idx] for idx in top_mapped_indices], dtype=int)
        except Exception as e:
            print(f"Warning: BIC selection failed ({e}), falling back to f_regression.")
            from sklearn.feature_selection import f_regression
            y_sel = y[:, 0] if y.ndim > 1 else y
            # X_filtered might have been pruned by hierarchical strategy
            scores, _ = f_regression(X_filtered, y_sel)
            scores = np.nan_to_num(scores)
            top_f = np.argsort(scores)[-min(self.max_features, len(scores)):]
            selected_filtered_indices = [mapping[idx] for idx in top_f]
            
        self.selected_indices = high_variance_indices[selected_filtered_indices]
        return self

    def transform(self, X):
        return X[:, self.selected_indices]


class RecursiveFeatureSelector:
    """
    Selects features recursively by iteratively adding the most informative ones.
    This prevents the combinatorial explosion of polynomial terms.
    """
    def __init__(self, max_features=40, tolerance=1e-4, min_variance=1e-6):
        self.max_features = max_features
        self.tolerance = tolerance
        self.min_variance = min_variance
        self.selected_indices = []

    def fit(self, X, y):
        # 0. Clean input data
        X = np.nan_to_num(X, nan=0.0, posinf=1e9, neginf=-1e9)
        y = np.nan_to_num(y, nan=0.0, posinf=1e9, neginf=-1e9)

        # Memory optimization: sample data if it's too large for stable selection
        if X.shape[0] > 5000:
            indices = np.random.choice(X.shape[0], 5000, replace=False)
            X_fit = X[indices]
            y_fit = y[indices]
        else:
            X_fit = X
            y_fit = y

        # 1. Variance filtering to remove constant or near-constant features
        variances = np.var(X_fit, axis=0)
        high_variance_indices = np.where(variances > self.min_variance)[0]
        
        if len(high_variance_indices) == 0:
            self.selected_indices = np.array([], dtype=int)
            return self
            
        X_filtered = X_fit[:, high_variance_indices]
        
        # 2. Use Orthogonal Matching Pursuit or LassoLars for selection
        from sklearn.linear_model import OrthogonalMatchingPursuit, LassoLarsCV
        
        n_samples, n_features = X_filtered.shape
        n_to_select = min(self.max_features, n_features, n_samples - 1)
        
        if n_to_select <= 0:
            self.selected_indices = high_variance_indices[:min(self.max_features, len(high_variance_indices))]
            return self

        try:
            # LassoLars is often more stable for feature selection than OMP
            lasso = LassoLarsCV(cv=3, max_iter=500)
            # Use the first output for feature selection if multi-output is provided
            y_selection = y_fit[:, 0] if y_fit.ndim > 1 else y_fit
            lasso.fit(X_filtered, y_selection)
            
            # Get indices of non-zero coefficients
            nonzero = np.where(np.abs(lasso.coef_) > 1e-10)[0]
            
            if len(nonzero) > 0:
                # If too many features selected, take the ones with largest coefficients
                if len(nonzero) > self.max_features:
                    top_indices = np.argsort(np.abs(lasso.coef_[nonzero]))[-self.max_features:]
                    selected_filtered_indices = nonzero[top_indices]
                else:
                    selected_filtered_indices = nonzero
            else:
                # Fallback to OMP if Lasso selected nothing
                omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_to_select, tol=self.tolerance)
                omp.fit(X_filtered, y_selection)
                selected_filtered_indices = np.where(omp.coef_ != 0)[0]
        except:
            # Fallback to simple univariate selection
            from sklearn.feature_selection import f_regression
            y_selection = y[:, 0] if y.ndim > 1 else y
            scores, _ = f_regression(X_filtered, y_selection)
            selected_filtered_indices = np.argsort(np.nan_to_num(scores))[-n_to_select:]
            
        self.selected_indices = high_variance_indices[selected_filtered_indices]
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
                 basis_functions='physics_informed', max_degree=2, feature_selection_method='recursive',
                 include_raw_latents=False, sim_type=None, hamiltonian=False):
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
            include_raw_latents (bool): Whether to include raw linear latents (often leads to trivial H=z0)
            sim_type (str): Type of simulation ('spring', 'lj', etc.)
            hamiltonian (bool): Whether this is for a Hamiltonian system
        """
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.include_dists = include_dists
        self.box_size = box_size
        self.basis_functions = basis_functions
        self.max_degree = max_degree
        self.feature_selection_method = feature_selection_method
        self.include_raw_latents = include_raw_latents
        self.sim_type = sim_type
        self.hamiltonian = hamiltonian
        
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
        self.feature_names = []
        
        # For polynomial features
        self.poly_transformer = None
        self.feature_dims = [] # NEW: store dimension of each feature (L, M, T)

    def _get_dim(self, name):
        """Helper to get dimension of a feature by name."""
        if 'sum_inv_d' in name:
            # Treat all inverse distance powers as having a generic "potential" dimension
            # to allow them to be combined in the symbolic search.
            return (-1, 0, 0) 
        if 'sum_d' in name: return (1, 0, 0)
        if 'p' in name and '2_sum' in name: return (0, 2, -2) # P^2
        if name.startswith('z'):
            try:
                # Use regex to find the dimension index within the latent dim
                match = re.search(r'z(\d+)', name)
                if match:
                    idx = int(match.group(1))
                    dim_idx = idx % self.latent_dim
                    if dim_idx < self.latent_dim // 2: return (1, 0, 0) # Q ~ L
                    return (0, 1, -1) # P ~ M L T^-1
            except: pass
        return (0, 0, 0) # Dimensionless or unknown

    def fit(self, latent_states, targets):
        """
        Fit the transformer to the data.
        """
        # 1. Fit raw latent normalization
        self.z_mean = latent_states.mean(axis=0)
        self.z_std = latent_states.std(axis=0) + 1e-6

        # 2. Transform to poly features
        X_poly = self.transform(latent_states, fit_transformer=True)

        # 3. Fit poly feature normalization on ALL features (to support full buffers in Torch)
        self.x_poly_mean = X_poly.mean(axis=0)
        self.x_poly_std = X_poly.std(axis=0) + 1e-6

        # 4. Perform feature selection
        self._perform_feature_selection(X_poly, targets)
        
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
        # Robust clipping of input latents to prevent extreme values
        z_flat_clipped = np.clip(z_flat, -1e3, 1e3)
        
        # NEW: Center and normalize raw latents to prevent trivial H = X0 discovery
        if self.z_mean is not None:
            z_flat_norm = (z_flat_clipped - self.z_mean) / self.z_std
        else:
            z_flat_norm = z_flat_clipped
            
        z_nodes = z_flat_clipped.reshape(-1, self.n_super_nodes, self.latent_dim)

        # Use normalized latents instead of raw ones in the feature set if requested
        features = []
        if self.include_raw_latents:
            features.append(z_flat_norm)
        
        # Generate base names
        if fit_transformer:
            base_names = []
            if self.include_raw_latents:
                base_names.extend([f"z{i}" for i in range(self.n_super_nodes * self.latent_dim)])
        
        if self.include_dists and self.basis_functions != 'polynomial_only':
            dist_features = self._compute_distance_features(z_nodes)
            if dist_features:
                features.extend(dist_features)
                if fit_transformer:
                    base_names.append("sum_d")
                    base_names.append("sum_inv_d")
                    # Expanded spectrum of power laws for unbiased discovery
                    for n in [2, 3, 4, 5, 6, 8, 10, 12, 14]:
                        base_names.append(f"sum_inv_d{n}")
                    base_names.append("sum_exp_d")
                    base_names.append("sum_yukawa_d")
                    base_names.append("sum_log_d")

        X = np.hstack(features)
        # Final safety clip before expansion
        X = np.clip(X, -1e6, 1e6)

        # Apply basis function expansion based on configuration
        raw_latent_names = [f"z{i}" for i in range(self.n_super_nodes * self.latent_dim)] if fit_transformer else None
        
        if self.basis_functions == 'polynomial':
            X_expanded = self._polynomial_expansion(X, fit_transformer)
            if fit_transformer:
                # PolynomialFeatures.get_feature_names_out is standard
                self.feature_names = self.poly_transformer.get_feature_names_out(base_names)
        elif self.basis_functions == 'physics_informed':
            X_expanded, expanded_names = self._physics_informed_expansion(X, z_flat_norm, base_names if fit_transformer else None, raw_latent_names)
            if fit_transformer:
                self.feature_names = expanded_names
        elif self.basis_functions == 'adaptive':
            X_expanded = self._adaptive_expansion(X, fit_transformer)
        else:
            X_expanded, expanded_names = self._physics_informed_expansion(X, z_flat_norm, base_names if fit_transformer else None, raw_latent_names)
            if fit_transformer:
                self.feature_names = expanded_names

        # Final safety clip and NaN handling
        X_expanded = np.nan_to_num(X_expanded, nan=0.0, posinf=1e9, neginf=-1e9)
        return np.clip(X_expanded, -1e12, 1e12)

    def _compute_distance_features(self, z_nodes):
        """
        Compute symmetric sums of distance-based features over all pairs of super-nodes.
        This provides O(1) features regardless of the number of particles, which is
        essential for symbolic discovery in N-body systems.
        """
        n_batch = z_nodes.shape[0]
        if self.n_super_nodes < 2:
            return []

        # Get all pairs of indices
        i_idx, j_idx = np.triu_indices(self.n_super_nodes, k=1)

        # [Batch, n_pairs, 2]
        diff = z_nodes[:, i_idx, :2] - z_nodes[:, j_idx, :2]

        if self.box_size is not None:
            # Minimum Image Convention
            box = np.array(self.box_size)
            diff -= box * np.round(diff / box)

        # [Batch, n_pairs]
        d = np.linalg.norm(diff, axis=2) + 1e-6
        
        aggregated_features = []
        
        # We compute the features for EACH pair and then SUM them across the pair dimension (axis 1)
        # This creates "Global Symmetric Features"
        
        # 1. Basic distance and inverse distance
        aggregated_features.append(d.sum(axis=1, keepdims=True))
        aggregated_features.append((1.0 / (d + 0.001)).sum(axis=1, keepdims=True))
        
        # 2. Spectrum of power laws: 1/r^n (unbiased)
        for n in [2, 3, 4, 5, 6, 8, 10, 12, 14]:
            aggregated_features.append((1.0 / (d**n + 0.001)).sum(axis=1, keepdims=True))
        
        # 3. Short-range interaction terms (Exponential/Yukawa-like)
        aggregated_features.append(np.exp(-d).sum(axis=1, keepdims=True))
        aggregated_features.append((np.exp(-d) / (d + 0.001)).sum(axis=1, keepdims=True))
        
        # 4. Logarithmic interactions (2D gravity/electrostatics)
        aggregated_features.append(np.log(d + 0.001).sum(axis=1, keepdims=True))

        return aggregated_features

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

    def _physics_informed_expansion(self, X, z_flat_norm, base_names=None, raw_latent_names=None):
        """
        Perform physics-informed polynomial expansion with cross-terms.
        Includes a memory safety check and vectorization for performance.
        """
        batch_size, n_total_features = X.shape
        n_raw_latents = self.n_super_nodes * self.latent_dim
        
        # 1. Base features: (Optional raw latents) + distance features
        features = [X]
        names = list(base_names) if base_names is not None else None
        
        # 2. Squares and Transcendental of raw latents
        X_raw = z_flat_norm
        if self.include_raw_latents:
            # Squares
            features.append(X_raw**2)
            if names is not None and raw_latent_names is not None:
                for i in range(n_raw_latents):
                    names.append(f"{raw_latent_names[i]}^2")
            
            # Transcendental: Sin, Cos, Log, Exp
            # We clip inputs to these functions to maintain numerical stability
            features.append(np.sin(X_raw))
            features.append(np.cos(X_raw))
            # Log(abs(x) + eps) to capture logarithmic dependencies
            features.append(np.log(np.abs(X_raw) + 1e-3))
            # Exp(x) clipped to prevent overflow
            features.append(np.exp(np.clip(X_raw, -10, 2)))
            
            if names is not None and raw_latent_names is not None:
                for i in range(n_raw_latents): names.append(f"sin({raw_latent_names[i]})")
                for i in range(n_raw_latents): names.append(f"cos({raw_latent_names[i]})")
                for i in range(n_raw_latents): names.append(f"log(|{raw_latent_names[i]}|)")
                for i in range(n_raw_latents): names.append(f"exp({raw_latent_names[i]})")

        # NEW: Explicit sum of squares of momentum dims (p^2) per super-node
        # Assuming last half of latent_dim are momentum dims
        d_sub = self.latent_dim // 2
        X_nodes = X_raw.reshape(batch_size, self.n_super_nodes, self.latent_dim)
        p_sq_sum = (X_nodes[:, :, d_sub:]**2).sum(axis=2) # [Batch, K]
        features.append(p_sq_sum)
        if names is not None:
            for k in range(self.n_super_nodes):
                names.append(f"p{k}^2_sum")

        # For small systems, target < 60 features by skipping cross-terms
        if self.n_super_nodes <= 4:
            X_res = np.concatenate(features, axis=1)
            return X_res, names

        # 3. Intra-node cross-terms: O(K * D^2)
        # Using vectorized outer product per node
        X_nodes = X_raw.reshape(batch_size, self.n_super_nodes, self.latent_dim)
        intra_terms = []
        for i in range(self.latent_dim):
            for j in range(i + 1, self.latent_dim):
                intra_terms.append(X_nodes[:, :, i] * X_nodes[:, :, j])
                if names is not None:
                    # This name generation is slightly complex due to vectorization across K
                    # We'll just use a generic name for these
                    pass
        
        if intra_terms:
            features.append(np.stack(intra_terms, axis=-1).reshape(batch_size, -1))
            if names is not None:
                for i in range(self.latent_dim):
                    for j in range(i + 1, self.latent_dim):
                        for k in range(self.n_super_nodes):
                            names.append(f"z{k*self.latent_dim+i}*z{k*self.latent_dim+j}")
                    
        # 4. Inter-node cross-terms (same dimension): O(D * K^2)
        # We only consider pairs of nodes here.
        inter_terms = []
        for d in range(self.latent_dim):
            # Vectorized pair-wise multiplication for this dimension
            # [Batch, K]
            val_d = X_nodes[:, :, d]
            i_idx, j_idx = np.triu_indices(self.n_super_nodes, k=1)
            inter_terms.append(val_d[:, i_idx] * val_d[:, j_idx])
            if names is not None:
                for i, j in zip(i_idx, j_idx):
                    names.append(f"z{i*self.latent_dim+d}*z{j*self.latent_dim+d}")
            
        if inter_terms:
            features.append(np.stack(inter_terms, axis=-1).reshape(batch_size, -1))
            
        # Combine all features
        X_expanded = np.concatenate(features, axis=1)
        
        # Memory Safety: If the number of features is still too large, 
        # we can perform a quick variance-based pruning here
        if X_expanded.shape[1] > 1000:
            variances = np.var(X_expanded, axis=0)
            top_indices = np.argsort(variances)[-1000:]
            X_expanded = X_expanded[:, top_indices]
            if names is not None:
                names = [names[idx] for idx in top_indices]
        
        # NEW: Update feature_dims if names are available
        if names is not None:
            self.feature_dims = [self._get_dim(n) for n in names]
            
        return X_expanded, names

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

        if self.feature_selection_method == 'bic':
            self.bic_selector = BICFeatureSelector(max_features=min(30, n_features))
            self.bic_selector.fit(X, y)
            self.selected_feature_indices = self.bic_selector.selected_indices
        elif self.feature_selection_method == 'recursive':
            self.recursive_selector = RecursiveFeatureSelector(max_features=min(40, n_features))
            self.recursive_selector.fit(X, y)
            self.selected_feature_indices = self.recursive_selector.selected_indices
        elif self.feature_selection_method == 'lasso':
            lasso = LassoCV(cv=3, max_iter=2000)
            y_selection = y[:, 0] if y.ndim > 1 else y
            lasso.fit(X, y_selection)
            self.selected_feature_indices = np.where(np.abs(lasso.coef_) > 1e-5)[0]
        # Fallback to mutual info
        else:
            max_f = min(200, n_features)
            selector = SelectKBest(score_func=f_regression, k=max_f)
            y_selection = y[:, 0] if y.ndim > 1 else y
            selector.fit(X, y_selection)
            self.selected_feature_indices = selector.get_support(indices=True)

        # HAMILTONIAN PRIORITIZATION: Ensure p^2 and key potential terms are included
        if self.hamiltonian:
            # Find indices of p^2 terms (p0^2_sum, p1^2_sum...)
            p_sq_indices = [i for i, name in enumerate(self.feature_names) if 'p' in name and '2_sum' in name]
            
            # Find indices of important distance terms (sum_inv_d6, sum_inv_d12 for LJ)
            dist_indices = []
            if self.sim_type == 'lj':
                dist_indices = [i for i, name in enumerate(self.feature_names) if 'sum_inv_d6' in name or 'sum_inv_d12' in name]
            elif self.sim_type == 'spring':
                dist_indices = [i for i, name in enumerate(self.feature_names) if 'sum_inv_d2' in name or 'sum_d' in name]
            
            important_indices = set(p_sq_indices + dist_indices)
            if important_indices:
                # Add them to selected indices if not already there
                current_selected = set(self.selected_feature_indices)
                updated_selected = sorted(list(current_selected.union(important_indices)))
                # If too many, remove some from the original selection (not the important ones)
                if len(updated_selected) > 40:
                    # Keep all important ones, and take the rest from original
                    non_important_original = [i for i in self.selected_feature_indices if i not in important_indices]
                    to_keep = 40 - len(important_indices)
                    updated_selected = sorted(list(important_indices) + non_important_original[:to_keep])
                
                self.selected_feature_indices = np.array(updated_selected)

        self.n_selected_features = len(self.selected_feature_indices)

    def normalize_x(self, X_poly):
        """
        Normalize features using fitted statistics.
        """
        # Normalize ALL first, then slice. This ensures consistency with TorchFeatureTransformer
        X_norm_full = (X_poly - self.x_poly_mean) / self.x_poly_std
        return X_norm_full[:, self.selected_feature_indices]

    def normalize_y(self, Y):
        return (Y - self.target_mean) / self.target_std

    def denormalize_y(self, Y_norm):
        return Y_norm * self.target_std + self.target_mean

    def transform_jacobian(self, z_flat):
        """
        Compute Jacobian dX/dz for the balanced transformation.
        """
        n_latents = self.n_super_nodes * self.latent_dim
        z_nodes = z_flat.reshape(self.n_super_nodes, self.latent_dim)
        z_flat_norm = (z_flat - self.z_mean) / self.z_std if self.z_mean is not None else z_flat
        
        # 1. Base features Jacobian (Latents + Distances)
        jac_list = []
        if self.include_raw_latents:
            if self.z_std is not None:
                jac_list.append(np.diag(1.0 / self.z_std))
            else:
                jac_list.append(np.eye(n_latents))
        
        if self.include_dists:
            i_idx, j_idx = np.triu_indices(self.n_super_nodes, k=1)
            
            # Initialize sum jacobians
            jd_sum = np.zeros(n_latents)
            j_inv_d_sum = np.zeros(n_latents)
            j_pow_sums = {n: np.zeros(n_latents) for n in [2, 3, 4, 5, 6, 8, 10, 12, 14]}
            j_exp_sum = np.zeros(n_latents)
            j_yukawa_sum = np.zeros(n_latents)
            j_log_sum = np.zeros(n_latents)

            softening = 0.001

            for i, j in zip(i_idx, j_idx):
                diff = z_nodes[i, :2] - z_nodes[j, :2]
                if self.box_size is not None:
                    box = np.array(self.box_size)
                    diff -= box * np.round(diff / box)
                
                d = np.linalg.norm(diff) + 1e-9
                
                # Gradient of d(i,j) with respect to all latents
                jd_pair = np.zeros(n_latents)
                jd_pair[i*self.latent_dim : i*self.latent_dim+2] = diff / d
                jd_pair[j*self.latent_dim : j*self.latent_dim+2] = -diff / d
                
                # Sum the jacobians of individual terms
                jd_sum += jd_pair
                j_inv_d_sum += -1.0 / (d + softening)**2 * jd_pair
                
                for n in [2, 3, 4, 5, 6, 8, 10, 12, 14]:
                    j_pow_sums[n] += -n * d**(n-1) / (d**n + softening)**2 * jd_pair
                    
                j_exp_sum += -np.exp(-d) * jd_pair
                j_yukawa_sum += ((-np.exp(-d)*(d + softening) - np.exp(-d)) / (d + softening)**2 * jd_pair)
                j_log_sum += 1.0 / (d + softening) * jd_pair

            # Add aggregated jacobians to list
            jac_list.append(jd_sum.reshape(1, -1))
            jac_list.append(j_inv_d_sum.reshape(1, -1))
            for n in [2, 3, 4, 5, 6, 8, 10, 12, 14]:
                jac_list.append(j_pow_sums[n].reshape(1, -1))
            jac_list.append(j_exp_sum.reshape(1, -1))
            jac_list.append(j_yukawa_sum.reshape(1, -1))
            jac_list.append(j_log_sum.reshape(1, -1))

        X_base_jac = np.vstack(jac_list)
        
        # 2. Physics-informed expansion Jacobian
        # Original features
        poly_jacs = [X_base_jac]
        
        # Squares of normalized latents
        # d(z_norm^2)/dz = 2 * z_norm * d(z_norm)/dz = 2 * z_norm * (1/z_std)
        z_norm_jac = np.diag(1.0 / self.z_std) if self.z_std is not None else np.eye(n_latents)
        for i in range(n_latents):
            poly_jacs.append((2 * z_flat_norm[i] * z_norm_jac[i]).reshape(1, -1))
            
        # Sum of squares of momentum dims (p^2) per super-node
        d_sub = self.latent_dim // 2
        for k in range(self.n_super_nodes):
            p_sq_jac = np.zeros(n_latents)
            for d in range(d_sub, self.latent_dim):
                idx = k * self.latent_dim + d
                p_sq_jac += 2 * z_flat_norm[idx] * z_norm_jac[idx]
            poly_jacs.append(p_sq_jac.reshape(1, -1))

        if self.n_super_nodes > 4:
            # Intra-node cross-terms
            for node_idx in range(self.n_super_nodes):
                start_idx = node_idx * self.latent_dim
                for i in range(self.latent_dim):
                    for j in range(i + 1, self.latent_dim):
                        idx_i = start_idx + i
                        idx_j = start_idx + j
                        # d(zi_n*zj_n)/dz = zi_n*d(zj_n)/dz + zj_n*d(zi_n)/dz
                        jd_cross = z_flat_norm[idx_i] * z_norm_jac[idx_j] + z_flat_norm[idx_j] * z_norm_jac[idx_i]
                        poly_jacs.append(jd_cross.reshape(1, -1))
                        
            # Inter-node cross-terms (same dimension)
            for dim_idx in range(self.latent_dim):
                for i in range(self.n_super_nodes):
                    for j in range(i + 1, self.n_super_nodes):
                        idx_i = i * self.latent_dim + dim_idx
                        idx_j = j * self.latent_dim + dim_idx
                        jd_cross = z_flat_norm[idx_i] * z_norm_jac[idx_j] + z_flat_norm[idx_j] * z_norm_jac[idx_i]
                        poly_jacs.append(jd_cross.reshape(1, -1))
                    
        X_expanded_jac = np.vstack(poly_jacs)
        
        # Apply feature selection if indices are present
        if self.selected_feature_indices is not None:
            # We need to make sure we don't exceed the number of rows in X_expanded_jac
            valid_indices = self.selected_feature_indices[self.selected_feature_indices < X_expanded_jac.shape[0]]
            return X_expanded_jac[valid_indices]
        return X_expanded_jac

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
                        1.0 / (d + 1e-4),
                        1.0 / (d**2 + 1e-4)
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
                        1.0 / (d + 1e-4),
                        1.0 / (d**2 + 1e-4)
                    ])

        if all_dist_features:
            return [np.hstack(all_dist_features)]
        else:
            return []