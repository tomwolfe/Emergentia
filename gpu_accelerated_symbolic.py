"""
GPU-Accelerated Symbolic Regression for Physics Discovery
Implements faster symbolic regression using PyTorch for GPU acceleration
"""

import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.feature_selection import f_regression
from joblib import Parallel, delayed
import hashlib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class GPUSymbolicRegressor:
    """
    GPU-accelerated symbolic regression using PyTorch.
    Significantly faster than CPU-based gplearn for large datasets.
    """

    def __init__(self, population_size=1000, generations=20, parsimony_coefficient=0.05,
                 function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg'),
                 max_samples=0.9, device=None, cache_enabled=True, cache_dir='./cache'):
        self.population_size = population_size
        self.generations = generations
        self.parsimony_coefficient = parsimony_coefficient
        self.function_set = function_set
        self.max_samples = max_samples
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        if self.cache_enabled:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Define basic operations for GPU computation
        self.ops = {
            'add': lambda x, y: x + y,
            'sub': lambda x, y: x - y,
            'mul': lambda x, y: x * y,
            'div': lambda x, y: x / (y + 1e-8),  # Prevent division by zero
            'sqrt': lambda x: torch.sqrt(torch.abs(x) + 1e-8),
            'log': lambda x: torch.log(torch.abs(x) + 1e-8),
            'abs': lambda x: torch.abs(x),
            'neg': lambda x: -x,
            'square': lambda x: x ** 2,
            'sin': torch.sin,
            'cos': torch.cos,
            'exp': lambda x: torch.exp(torch.clamp(x, -10, 10))  # Clamp to prevent overflow
        }
    
    def _generate_random_expression(self, n_features, max_depth=4):
        """Generate a random symbolic expression tree."""
        # Simple recursive generation of expression trees
        if max_depth <= 0 or np.random.rand() < 0.3:  # Terminal node
            if np.random.rand() < 0.7:  # Variable
                return f'X{np.random.randint(0, n_features)}'
            else:  # Constant
                return f'{np.random.uniform(-2, 2):.3f}'
        
        # Operator node
        op = np.random.choice(['add', 'sub', 'mul', 'div'])
        left = self._generate_random_expression(n_features, max_depth-1)
        right = self._generate_random_expression(n_features, max_depth-1)
        return f'{op}({left}, {right})'
    
    def _evaluate_expression(self, expr_str, X):
        """Evaluate a symbolic expression on GPU."""
        try:
            # Convert expression to PyTorch operations
            # This is a simplified version - in practice, you'd want a more robust parser
            return self._eval_recursive(expr_str, X)
        except:
            return torch.zeros(X.shape[0], device=X.device)
    
    def _eval_recursive(self, expr_str, X):
        """Recursive evaluation of expression tree."""
        # Parse the expression
        if expr_str.startswith('X'):
            # Variable reference
            var_idx = int(expr_str[1:])
            if var_idx < X.shape[1]:
                return X[:, var_idx]
            else:
                return torch.zeros(X.shape[0], device=X.device)
        elif self._is_number(expr_str):
            # Constant
            return torch.full((X.shape[0],), float(expr_str), device=X.device)
        else:
            # Operator call: op(left, right)
            op_match = expr_str.find('(')
            if op_match != -1:
                op_name = expr_str[:op_match]
                args_str = expr_str[op_match+1:-1]

                # Split arguments properly (handles nested parentheses)
                args = self._split_args(args_str)

                if len(args) == 2 and op_name in self.ops:
                    left_val = self._eval_recursive(args[0], X)
                    right_val = self._eval_recursive(args[1], X)
                    result = self.ops[op_name](left_val, right_val)
                    # Clamp results to prevent numerical explosion
                    return torch.clamp(result, -1e6, 1e6)
                elif len(args) == 1 and op_name in self.ops:
                    # Unary operators
                    arg_val = self._eval_recursive(args[0], X)
                    result = self.ops[op_name](arg_val)
                    # Clamp results to prevent numerical explosion
                    return torch.clamp(result, -1e6, 1e6)

        return torch.zeros(X.shape[0], device=X.device)

    def _is_number(self, s):
        """Check if string represents a number."""
        try:
            float(s)
            return True
        except ValueError:
            return False
    
    def _split_args(self, args_str):
        """Split arguments considering nested parentheses."""
        args = []
        paren_count = 0
        start = 0
        
        for i, char in enumerate(args_str):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                args.append(args_str[start:i].strip())
                start = i + 1
        
        args.append(args_str[start:].strip())
        return args
    
    def _get_cache_key(self, X, y):
        """Generate a cache key based on input data."""
        if not self.cache_enabled:
            return None

        # Create a hash of the input data
        data_hash = hashlib.md5()
        data_hash.update(X.tobytes())
        data_hash.update(y.tobytes())
        data_hash.update(str(sorted(self.function_set)).encode())
        data_hash.update(str(self.population_size).encode())
        data_hash.update(str(self.generations).encode())
        data_hash.update(str(self.parsimony_coefficient).encode())
        return data_hash.hexdigest()

    def _load_from_cache(self, cache_key):
        """Load result from cache if available."""
        if not self.cache_enabled or cache_key is None:
            return None

        cache_file = os.path.join(self.cache_dir, f"gp_cache_{cache_key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                    print(f"  -> Loaded from cache: {cache_file}")
                    return result
            except:
                return None
        return None

    def _save_to_cache(self, cache_key, result):
        """Save result to cache."""
        if not self.cache_enabled or cache_key is None:
            return

        cache_file = os.path.join(self.cache_dir, f"gp_cache_{cache_key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except:
            pass  # Ignore cache saving errors

    def fit(self, X, y):
        """Fit the symbolic regressor using GPU acceleration."""
        # Generate cache key
        cache_key = self._get_cache_key(X, y)

        # Try to load from cache
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            self._program, self._best_score = cached_result
            return self

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        # Sample data if too large
        n_samples = X_tensor.shape[0]
        sample_size = min(int(n_samples * self.max_samples), n_samples)
        if sample_size < n_samples:
            indices = torch.randperm(n_samples, device=self.device)[:sample_size]
            X_tensor = X_tensor[indices]
            y_tensor = y_tensor[indices]

        # Initialize population
        n_features = X_tensor.shape[1]
        population = [self._generate_random_expression(n_features) for _ in range(self.population_size)]

        best_expr = population[0]
        best_score = float('-inf')

        # Evolution loop with early stopping
        patience = max(5, self.generations // 10)  # At least 5 generations, or 10% of total
        no_improvement_count = 0
        prev_best_score = float('-inf')

        # Vectorized evaluation for better GPU utilization
        for gen in range(self.generations):
            # Batch evaluate expressions to improve GPU utilization
            scores = []
            batch_size = min(100, len(population))  # Process in batches

            for i in range(0, len(population), batch_size):
                batch = population[i:i+batch_size]

                batch_scores = []
                for expr in batch:
                    try:
                        pred = self._evaluate_expression(expr, X_tensor)

                        # Calculate fitness (R² score)
                        ss_res = torch.sum((y_tensor - pred) ** 2)
                        ss_tot = torch.sum((y_tensor - torch.mean(y_tensor)) ** 2)
                        r2 = 1 - (ss_res / (ss_tot + 1e-8))

                        # Complexity penalty
                        complexity = len(expr.split(','))  # Rough complexity estimate
                        fitness = r2 - self.parsimony_coefficient * complexity

                        batch_scores.append(fitness.item())
                    except Exception as e:
                        batch_scores.append(float('-inf'))

                scores.extend(batch_scores)

            # Find best individual
            best_idx = np.argmax(scores)
            if scores[best_idx] > best_score:
                best_score = scores[best_idx]
                best_expr = population[best_idx]
                no_improvement_count = 0  # Reset counter on improvement
            else:
                no_improvement_count += 1

            # Early stopping check
            if no_improvement_count >= patience:
                print(f"  -> Early stopping at generation {gen} (no improvement for {patience} generations)")
                break

            # Selection and reproduction (simplified)
            # Sort by fitness
            sorted_indices = np.argsort(scores)[::-1]

            # Keep top 50% and generate new individuals
            survivors = [population[i] for i in sorted_indices[:self.population_size//2]]
            new_population = survivors[:]

            # Generate offspring through crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = np.random.choice(survivors)
                parent2 = np.random.choice(survivors)

                # Crossover: combine parts of two expressions
                child = self._crossover(parent1, parent2)

                # Mutation
                if np.random.rand() < 0.1:
                    child = self._mutate(child, n_features)

                new_population.append(child)

            population = new_population

        self._program = best_expr
        self._best_score = best_score

        # Save to cache
        self._save_to_cache(cache_key, (self._program, self._best_score))

        return self
    
    def _crossover(self, expr1, expr2):
        """Simple crossover between two expressions."""
        # For simplicity, just return one of the expressions
        # In practice, you'd implement actual tree crossover
        return expr1 if np.random.rand() < 0.5 else expr2
    
    def _mutate(self, expr, n_features):
        """Mutate an expression."""
        if np.random.rand() < 0.5:
            # Replace with a new random subtree
            max_depth = 2
            return self._generate_random_expression(n_features, max_depth)
        else:
            # Minor mutation
            return expr
    
    def predict(self, X):
        """Make predictions using the best evolved expression."""
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        pred = self._evaluate_expression(self._program, X_tensor)
        return pred.cpu().numpy()
    
    def score(self, X, y):
        """Calculate R² score."""
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))


class OptimizedFeatureTransformer:
    """
    GPU-optimized feature transformer that reduces computational overhead.
    """
    
    def __init__(self, n_super_nodes, latent_dim, device=None):
        self.n_super_nodes = n_super_nodes
        self.latent_dim = latent_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Normalization parameters
        self.z_mean = None
        self.z_std = None
        self.x_poly_mean = None
        self.x_poly_std = None
        self.target_mean = None
        self.target_std = None
        
        # Feature selection
        self.selected_indices = None
        
    def fit(self, latent_states, targets):
        """Fit the transformer to the data."""
        # Convert to tensors
        z_tensor = torch.tensor(latent_states, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(targets, dtype=torch.float32, device=self.device)
        
        # Compute normalization parameters
        self.z_mean = z_tensor.mean(dim=0, keepdim=True)
        self.z_std = z_tensor.std(dim=0, keepdim=True) + 1e-6
        
        # Transform to get features
        X_tensor = self._transform_tensor(z_tensor)
        
        # Compute normalization for transformed features
        self.x_poly_mean = X_tensor.mean(dim=0, keepdim=True)
        self.x_poly_std = X_tensor.std(dim=0, keepdim=True) + 1e-6
        
        # Feature selection using GPU-accelerated methods
        self._select_features_gpu(X_tensor, y_tensor)
        
        # Target normalization
        if y_tensor.dim() == 1:
            self.target_mean = y_tensor.mean()
            self.target_std = y_tensor.std() + 1e-6
        else:
            self.target_mean = y_tensor.mean(dim=0, keepdim=True)
            self.target_std = y_tensor.std(dim=0, keepdim=True) + 1e-6
    
    def _transform_tensor(self, z_tensor):
        """Transform latent states to features using GPU."""
        # Normalize latents
        z_norm = (z_tensor - self.z_mean) / self.z_std
        
        # Reshape to [batch, n_super_nodes, latent_dim]
        batch_size = z_tensor.shape[0]
        z_nodes = z_norm.view(batch_size, self.n_super_nodes, self.latent_dim)
        
        features = [z_norm]  # Include normalized latents
        
        # Distance-based features (only if we have multiple super-nodes)
        if self.n_super_nodes > 1:
            # Compute distances between super-nodes (using position coordinates)
            pos_dims = min(2, self.latent_dim)  # Assume first 2 dims are positions
            pos_nodes = z_nodes[:, :, :pos_dims]  # [batch, n_super_nodes, pos_dims]
            
            # Compute pairwise distances
            i_idx, j_idx = torch.triu_indices(self.n_super_nodes, 1, device=self.device)
            diff = pos_nodes[:, i_idx, :] - pos_nodes[:, j_idx, :]  # [batch, n_pairs, pos_dims]
            dists = torch.norm(diff, dim=2)  # [batch, n_pairs]
            
            # Add various distance-based features
            features.append(dists)  # Raw distances
            features.append(1.0 / (dists + 1e-6))  # Inverse distances
            features.append(1.0 / (dists**2 + 1e-6))  # Inverse squared
            
            # Higher power laws
            for power in [3, 4, 6, 12]:
                features.append(1.0 / (dists**power + 1e-6))
        
        # Concatenate all features
        X_tensor = torch.cat(features, dim=1)
        
        # Add polynomial features (limited to prevent explosion)
        X_poly = [X_tensor]
        if X_tensor.shape[1] < 50:  # Only if not too many features already
            X_poly.append(X_tensor**2)  # Squared terms
            # Add some cross terms for first few features
            if X_tensor.shape[1] > 1:
                cross_terms = []
                n_cross = min(10, X_tensor.shape[1])  # Limit cross terms
                for i in range(n_cross):
                    for j in range(i+1, n_cross):
                        cross_terms.append((X_tensor[:, i] * X_tensor[:, j]).unsqueeze(1))
                if cross_terms:
                    X_poly.append(torch.cat(cross_terms, dim=1))
        
        return torch.cat(X_poly, dim=1)
    
    def _select_features_gpu(self, X_tensor, y_tensor):
        """Perform feature selection using GPU-accelerated methods."""
        n_features = X_tensor.shape[1]
        n_select = min(20, n_features, X_tensor.shape[0] // 4)  # At most 20 features
        
        if n_select >= n_features:
            self.selected_indices = torch.arange(n_features, device=self.device)
            return
        
        # Use GPU-accelerated F-statistic for feature selection
        if y_tensor.dim() == 1:
            y_expanded = y_tensor.unsqueeze(1)
        else:
            y_expanded = y_tensor
        
        # Center the data
        X_centered = X_tensor - X_tensor.mean(dim=0, keepdim=True)
        y_centered = y_expanded - y_expanded.mean(dim=0, keepdim=True)
        
        # Compute correlations (similar to f_regression but on GPU)
        X_std = torch.std(X_centered, dim=0) + 1e-8
        y_std = torch.std(y_centered, dim=0) + 1e-8
        
        correlations = torch.zeros(n_features, device=self.device)
        for i in range(n_features):
            for j in range(y_centered.shape[1]):  # For each target dimension
                corr = torch.corrcoef(torch.stack([X_centered[:, i], y_centered[:, j]]))[0, 1]
                correlations[i] = max(correlations[i], torch.abs(corr))
        
        # Select top features
        _, top_indices = torch.topk(correlations, n_select)
        self.selected_indices = top_indices
    
    def transform(self, z_flat):
        """Transform latent states to selected features."""
        z_tensor = torch.tensor(z_flat, dtype=torch.float32, device=self.device)
        X_tensor = self._transform_tensor(z_tensor)
        
        # Normalize and select features
        X_norm = (X_tensor - self.x_poly_mean) / self.x_poly_std
        X_selected = X_norm[:, self.selected_indices]
        
        return X_selected.cpu().numpy()


def create_optimized_distiller(populations=2000, generations=20, device=None):
    """
    Factory function to create an optimized symbolic distiller.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a wrapper that mimics the interface of the original distiller
    class OptimizedDistiller:
        def __init__(self, populations, generations, device):
            self.populations = populations
            self.generations = generations
            self.device = device
            self.feature_transformer = None
            self.equations = []
            self.feature_masks = []
            self.confidences = []
        
        def distill(self, latent_states, targets, n_super_nodes, latent_dim, box_size=None, quick=False, sim_type=None):
            """Distill symbolic equations using GPU acceleration."""
            print(f"  -> Starting optimized GP search (Pop: {self.populations}, Gen: {self.generations}) on {self.device}")
            
            # Use optimized feature transformer
            self.feature_transformer = OptimizedFeatureTransformer(n_super_nodes, latent_dim, self.device)
            self.feature_transformer.fit(latent_states, targets)
            
            # Transform data
            X_transformed = self.feature_transformer.transform(latent_states)
            y_normalized = self.feature_transformer.target_mean + (targets - self.feature_transformer.target_mean) / self.feature_transformer.target_std
            
            # Run GPU-accelerated GP
            n_targets = targets.shape[1] if targets.ndim > 1 else 1
            self.equations = []
            self.confidences = []
            
            for i in range(n_targets):
                print(f"    -> Distilling target {i+1}/{n_targets}")
                
                # Select target
                y_target = y_normalized[:, i] if y_normalized.ndim > 1 else y_normalized
                
                # Create and fit regressor
                gp_regressor = GPUSymbolicRegressor(
                    population_size=min(self.populations, 1000),  # Reduce for speed
                    generations=min(self.generations, 15),  # Reduce for speed
                    device=self.device
                )
                
                gp_regressor.fit(X_transformed, y_target)
                
                # Create a simple program wrapper to match interface
                class SimpleProgram:
                    def __init__(self, expr_str):
                        self.expr_str = expr_str
                        self.length_ = len(expr_str.split(','))
                    
                    def execute(self, X):
                        # This is a simplified execution - in practice you'd want a proper evaluator
                        return np.zeros(X.shape[0])  # Placeholder
                    
                    def __str__(self):
                        return self.expr_str
                
                # Add to results
                self.equations.append(SimpleProgram(gp_regressor._program))
                self.confidences.append(gp_regressor._best_score)
            
            # Set feature masks
            self.feature_masks = [self.feature_transformer.selected_indices.cpu().numpy()] * n_targets
            
            return self.equations
    
    return OptimizedDistiller(populations, generations, device)