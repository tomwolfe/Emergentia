"""
Robust Symbolic Proxy to improve stability of symbolic-in-the-loop training.
This addresses the symbolic proxy fragility issue identified in the analysis.
"""

import numpy as np
import torch
import torch.nn as nn
import sympy as sp
from symbolic import gp_to_sympy
from enhanced_symbolic import SymPyToTorch
import warnings
warnings.filterwarnings('ignore')


class EquationValidator:
    """
    Validates symbolic equations to ensure they are well-behaved before integration.
    """
    
    def __init__(self, max_complexity=50, max_depth=10, stability_threshold=1e6):
        self.max_complexity = max_complexity
        self.max_depth = max_depth
        self.stability_threshold = stability_threshold
    
    def validate_expression(self, expr_str, n_features=None):
        """
        Validate a symbolic expression string for potential issues.
        
        Args:
            expr_str: String representation of the symbolic expression
            n_features: Number of input features
            
        Returns:
            dict: Validation results with 'is_valid', 'score', and 'issues'
        """
        try:
            # Convert to SymPy expression
            sympy_expr = gp_to_sympy(expr_str, n_features=n_features)
            
            # Check complexity
            complexity = self._measure_complexity(sympy_expr)
            if complexity > self.max_complexity:
                return {
                    'is_valid': False,
                    'score': 0.0,
                    'issues': [f'Expression too complex: {complexity} > {self.max_complexity}']
                }
            
            # Check depth
            depth = self._measure_depth(sympy_expr)
            if depth > self.max_depth:
                return {
                    'is_valid': False,
                    'score': 0.0,
                    'issues': [f'Expression too deep: {depth} > {self.max_depth}']
                }
            
            # Check for potential numerical instabilities
            issues = []
            instability_score = 0.0
            
            # Look for divisions by variables (potential poles)
            div_issues = self._check_divisions(sympy_expr)
            if div_issues:
                issues.extend(div_issues)
                instability_score += 0.3 * len(div_issues)
            
            # Look for exponential functions (potential overflow)
            exp_issues = self._check_exponentials(sympy_expr)
            if exp_issues:
                issues.extend(exp_issues)
                instability_score += 0.2 * len(exp_issues)
            
            # Look for nested functions that might cause issues
            nested_issues = self._check_nested_functions(sympy_expr)
            if nested_issues:
                issues.extend(nested_issues)
                instability_score += 0.1 * len(nested_issues)
            
            # Calculate validity score (higher is better)
            base_score = 1.0
            score = max(0.0, base_score - instability_score)
            
            is_valid = len(issues) == 0
            if is_valid:
                score = min(1.0, score)  # Cap at 1.0
            else:
                score = max(0.0, score)  # Ensure non-negative
            
            return {
                'is_valid': is_valid,
                'score': score,
                'issues': issues,
                'complexity': complexity,
                'depth': depth
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'score': 0.0,
                'issues': [f'Parsing error: {str(e)}'],
                'error': str(e)
            }
    
    def _measure_complexity(self, expr):
        """Measure the complexity of a SymPy expression."""
        return expr.count_ops()
    
    def _measure_depth(self, expr):
        """Measure the depth of a SymPy expression tree."""
        if expr.is_Atom:
            return 1
        if not expr.args:
            return 1
        return 1 + max(self._measure_depth(arg) for arg in expr.args)
    
    def _check_divisions(self, expr):
        """Check for potential division-by-zero issues."""
        issues = []
        
        # Look for division operations
        for atom in expr.atoms(sp.Pow):
            if atom.args[1] == -1:  # Division by something
                divisor = atom.args[0]
                # Check if divisor contains variables (potential pole)
                if divisor.free_symbols:
                    issues.append(f'Division by variable expression: {divisor}')
        
        # Look for explicit division
        for atom in expr.atoms(sp.Mul):
            for arg in atom.args:
                if isinstance(arg, sp.Pow) and arg.args[1] == -1:
                    divisor = arg.args[0]
                    if divisor.free_symbols:
                        issues.append(f'Division by variable expression: {divisor}')
        
        return issues
    
    def _check_exponentials(self, expr):
        """Check for potential exponential overflow issues."""
        issues = []
        
        for atom in expr.atoms(sp.exp):
            # Check if argument contains variables that could cause overflow
            if atom.args[0].free_symbols:
                issues.append(f'Exponential function with variable argument: exp({atom.args[0]})')
        
        return issues
    
    def _check_nested_functions(self, expr):
        """Check for potentially problematic nested functions."""
        issues = []
        
        # Look for deeply nested transcendental functions
        for atom in expr.atoms(sp.Function):
            if isinstance(atom, (sp.sin, sp.cos, sp.tan, sp.log, sp.exp)):
                # Check if the argument itself contains transcendental functions
                for arg_atom in atom.args[0].atoms(sp.Function):
                    if isinstance(arg_atom, (sp.sin, sp.cos, sp.tan, sp.log, sp.exp)):
                        issues.append(f'Nested transcendental functions: {atom}')
        
        return issues


class RobustSymbolicProxy(nn.Module):
    """
    A robust symbolic proxy that validates equations before using them in training.
    """
    
    def __init__(self, equations, feature_masks, transformer, validator=None, 
                 confidence_threshold=0.5, validation_sample_size=100):
        super(RobustSymbolicProxy, self).__init__()
        
        self.equations = equations
        self.feature_masks = feature_masks
        self.transformer = transformer
        self.validator = validator or EquationValidator()
        self.confidence_threshold = confidence_threshold
        self.validation_sample_size = validation_sample_size
        
        # Track which equations have been validated and are safe to use
        self.valid_equations = []
        self.equation_scores = []
        self.validation_results = []
        
        # Convert validated equations to PyTorch modules for differentiable training
        self.torch_modules = nn.ModuleList()
        
        # Validate all equations
        self._validate_all_equations()
    
    def _validate_all_equations(self):
        """Validate all symbolic equations and prepare PyTorch modules."""
        for i, eq in enumerate(self.equations):
            if eq is None:
                self.valid_equations.append(False)
                self.equation_scores.append(0.0)
                self.validation_results.append({'is_valid': False, 'issues': ['No equation']})
                self.torch_modules.append(None)
                continue
            
            # Get the expression string
            expr_str = str(eq)
            
            # Validate the expression
            validation_result = self.validator.validate_expression(
                expr_str, 
                n_features=len(self.feature_masks[i]) if i < len(self.feature_masks) else None
            )
            
            self.validation_results.append(validation_result)
            self.valid_equations.append(validation_result['is_valid'])
            self.equation_scores.append(validation_result['score'])
            
            # Create PyTorch module only if equation is valid
            if validation_result['is_valid']:
                try:
                    # Convert to SymPy and then to PyTorch
                    sympy_expr = gp_to_sympy(expr_str)
                    n_inputs = len(self.feature_masks[i]) if i < len(self.feature_masks) else 1
                    torch_module = SymPyToTorch(sympy_expr, n_inputs)
                    self.torch_modules.append(torch_module)
                except Exception as e:
                    print(f"Failed to convert equation {i} to PyTorch: {e}")
                    self.torch_modules.append(None)
                    self.valid_equations[-1] = False
                    self.equation_scores[-1] = 0.0
            else:
                self.torch_modules.append(None)
    
    def forward(self, z_flat):
        """
        Forward pass through the symbolic proxy.
        
        Args:
            z_flat: Flattened latent states [batch_size, n_super_nodes * latent_dim]
            
        Returns:
            Predicted derivatives [batch_size, n_outputs] for valid equations,
            zeros for invalid equations
        """
        batch_size = z_flat.size(0)
        
        # Transform to feature space
        X_poly = self.transformer.transform(z_flat.detach().cpu().numpy())
        X_norm = self.transformer.normalize_x(X_poly)
        X_tensor = torch.from_numpy(X_norm).float().to(z_flat.device)
        
        outputs = []
        
        for i, (eq_valid, torch_module) in enumerate(zip(self.valid_equations, self.torch_modules)):
            if eq_valid and torch_module is not None:
                # Use the validated equation
                try:
                    mask = self.feature_masks[i] if i < len(self.feature_masks) else slice(None)
                    X_selected = X_tensor[:, mask] if isinstance(mask, (list, np.ndarray)) else X_tensor
                    output = torch_module(X_selected).view(-1, 1)
                    outputs.append(output)
                except Exception:
                    # Fallback to zero if evaluation fails
                    outputs.append(torch.zeros(batch_size, 1, device=z_flat.device))
            else:
                # Return zeros for invalid equations
                outputs.append(torch.zeros(batch_size, 1, device=z_flat.device))
        
        if outputs:
            return torch.cat(outputs, dim=1)
        else:
            return torch.zeros(batch_size, 0, device=z_flat.device)
    
    def get_validation_summary(self):
        """Get a summary of validation results."""
        n_total = len(self.equations)
        n_valid = sum(self.valid_equations)
        avg_score = np.mean(self.equation_scores) if self.equation_scores else 0.0
        
        summary = {
            'total_equations': n_total,
            'valid_equations': n_valid,
            'invalid_equations': n_total - n_valid,
            'validation_success_rate': n_valid / n_total if n_total > 0 else 0.0,
            'average_score': avg_score,
            'details': self.validation_results
        }
        
        return summary


class AdaptiveSymbolicProxy(nn.Module):
    """
    An adaptive symbolic proxy that gradually introduces symbolic equations
    as they become more reliable during training.
    """
    
    def __init__(self, equations, feature_masks, transformer, validator=None,
                 confidence_threshold=0.5, reliability_threshold=0.7):
        super(AdaptiveSymbolicProxy, self).__init__()
        
        self.robust_proxy = RobustSymbolicProxy(
            equations, feature_masks, transformer, validator,
            confidence_threshold
        )
        
        self.confidence_threshold = confidence_threshold
        self.reliability_threshold = reliability_threshold
        
        # Track reliability of each equation over time
        self.register_buffer('equation_reliability', 
                           torch.ones(len(equations)) if equations else torch.tensor([]))
        self.register_buffer('equation_usage_counts', 
                           torch.zeros(len(equations)) if equations else torch.tensor([]))
        self.register_buffer('equation_success_counts', 
                           torch.zeros(len(equations)) if equations else torch.tensor([]))
    
    def update_reliability(self, equation_indices, successes):
        """
        Update reliability scores based on recent performance.
        
        Args:
            equation_indices: Indices of equations to update
            successes: Boolean tensor indicating success/failure for each equation
        """
        for idx, success in zip(equation_indices, successes):
            if idx < len(self.equation_usage_counts):
                self.equation_usage_counts[idx] += 1
                if success:
                    self.equation_success_counts[idx] += 1
                
                # Update reliability as success rate with smoothing
                reliability = (self.equation_success_counts[idx] + 1.0) / (
                    self.equation_usage_counts[idx] + 2.0
                )
                self.equation_reliability[idx] = reliability
    
    def forward(self, z_flat, training_phase=1.0):
        """
        Forward pass with adaptive use of symbolic equations based on reliability.
        
        Args:
            z_flat: Flattened latent states
            training_phase: Float between 0 and 1 indicating training progress
            
        Returns:
            Predicted derivatives with adaptive equation usage
        """
        # Get predictions from robust proxy
        raw_predictions = self.robust_proxy(z_flat)
        
        # Apply reliability-based masking
        reliability_mask = (self.equation_reliability >= self.reliability_threshold).float()
        
        # During early training, be more conservative
        conservative_factor = max(0.5, 1.0 - training_phase * 0.5)
        adaptive_mask = reliability_mask * conservative_factor + (1 - conservative_factor)
        
        # Apply the adaptive mask
        masked_predictions = raw_predictions * adaptive_mask.unsqueeze(0)
        
        return masked_predictions
    
    def get_reliability_report(self):
        """Get a report on equation reliability."""
        return {
            'reliability_scores': self.equation_reliability.tolist(),
            'usage_counts': self.equation_usage_counts.tolist(),
            'success_counts': self.equation_success_counts.tolist(),
            'validation_summary': self.robust_proxy.get_validation_summary()
        }


def create_robust_symbolic_proxy(equations, feature_masks, transformer, 
                               confidence_threshold=0.5, use_adaptive=True):
    """
    Factory function to create a robust symbolic proxy.
    
    Args:
        equations: List of symbolic equations
        feature_masks: Corresponding feature masks
        transformer: Feature transformer
        confidence_threshold: Minimum confidence for equation acceptance
        use_adaptive: Whether to use adaptive reliability tracking
        
    Returns:
        Robust symbolic proxy module
    """
    if use_adaptive:
        return AdaptiveSymbolicProxy(
            equations, feature_masks, transformer,
            confidence_threshold=confidence_threshold
        )
    else:
        return RobustSymbolicProxy(
            equations, feature_masks, transformer,
            confidence_threshold=confidence_threshold
        )