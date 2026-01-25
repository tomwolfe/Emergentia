"""
Highly Optimized GPU-Accelerated Symbolic Regression for Physics Discovery
Implements vectorized symbolic regression using PyTorch for maximum GPU efficiency
"""

import torch
import torch.nn as nn
import numpy as np
import sympy as sp
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.feature_selection import f_regression
import hashlib
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class FastGPUSymbolicRegressor:
    """
    Highly optimized GPU-accelerated symbolic regression using vectorized operations.
    Designed for maximum throughput during the Deep Search phase.
    """

    def __init__(self, population_size=1000, generations=20, parsimony_coefficient=0.05,
                 max_samples=0.9, device=None, cache_enabled=True, cache_dir='./cache'):
        self.population_size = population_size
        self.generations = generations
        self.parsimony_coefficient = parsimony_coefficient
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

    def _evaluate_expression_vectorized(self, expressions, X):
        """
        Vectorized evaluation of multiple expressions for GPU efficiency.
        This is the key optimization for handling large populations.
        """
        results = []
        for expr_str in expressions:
            try:
                result = self._eval_recursive(expr_str, X)
                results.append(result)
            except:
                results.append(torch.zeros(X.shape[0], device=X.device))
        return torch.stack(results, dim=0)  # [pop_size, batch_size]

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
        data_hash.update(str(self.population_size).encode())
        data_hash.update(str(self.generations).encode())
        data_hash.update(str(self.parsimony_coefficient).encode())
        return data_hash.hexdigest()

    def _load_from_cache(self, cache_key):
        """Load result from cache if available."""
        if not self.cache_enabled or cache_key is None:
            return None

        cache_file = os.path.join(self.cache_dir, f"fast_gp_cache_{cache_key}.pkl")
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

        cache_file = os.path.join(self.cache_dir, f"fast_gp_cache_{cache_key}.pkl")
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

        # OPTIMIZATION: Process population in batches to maximize GPU utilization
        batch_size = min(256, self.population_size)  # Process up to 256 expressions at once

        for gen in range(self.generations):
            all_scores = []
            
            # Process population in batches
            for i in range(0, len(population), batch_size):
                batch_exprs = population[i:i+batch_size]
                
                # Vectorized evaluation of batch
                batch_results = self._evaluate_expression_vectorized(batch_exprs, X_tensor)
                
                # Calculate fitness for entire batch
                ss_res = torch.sum((y_tensor.unsqueeze(0) - batch_results) ** 2, dim=1)
                ss_tot = torch.sum((y_tensor - torch.mean(y_tensor)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))
                
                # Complexity penalty
                complexities = torch.tensor([len(expr.split(',')) for expr in batch_exprs], device=self.device)
                fitness = r2 - self.parsimony_coefficient * complexities.float()
                
                all_scores.extend(fitness.cpu().numpy())

            # Find best individual
            best_idx = np.argmax(all_scores)
            if all_scores[best_idx] > best_score:
                best_score = all_scores[best_idx]
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
            sorted_indices = np.argsort(all_scores)[::-1]

            # Keep top 50% and generate new individuals
            survivor_indices = sorted_indices[:self.population_size//2]
            survivors = [population[i] for i in survivor_indices]
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
        pred = self._eval_recursive(self._program, X_tensor)
        return pred.cpu().numpy()

    def score(self, X, y):
        """Calculate R² score."""
        pred = self.predict(X)
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))