"""
Configuration management system for the Neural-Symbolic Discovery Pipeline.
Provides auto-tuning and hyperparameter optimization capabilities.
"""

import json
import yaml
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.model_selection import ParameterGrid
import copy


@dataclass
class ModelConfig:
    """Configuration for the Discovery Engine Model."""
    n_particles: int = 16
    n_super_nodes: int = 4
    node_features: int = 4
    latent_dim: int = 4
    hidden_dim: int = 64
    hamiltonian: bool = True
    dissipative: bool = True
    min_active_super_nodes: int = 2
    use_learnable_bases: bool = True
    basis_hidden_dim: int = 64
    num_learnable_bases: int = 8
    use_attention: bool = True
    use_adaptive_selection: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    epochs: int = 1000
    steps: int = 200
    lr: float = 2e-4
    seq_len: int = 20
    dynamic_radius: float = 1.5
    box_size: tuple = (10.0, 10.0)
    sparsity_scheduler_enabled: bool = True
    sparsity_initial_weight: float = 0.001
    sparsity_target_weight: float = 0.05
    sparsity_warmup_steps: int = 100
    sparsity_max_steps: int = 800
    skip_consistency_freq: int = 3
    enable_gradient_accumulation: bool = True
    grad_acc_steps: int = 2
    warmup_epochs: int = 400
    align_anneal_epochs: int = 1000
    hard_assignment_start: float = 0.7
    use_enhanced_balancer: bool = True
    enhanced_balancer_strategy: str = 'gradient_based'


@dataclass
class SymbolicConfig:
    """Configuration for symbolic distillation."""
    populations: int = 2000
    generations: int = 40
    stopping_criteria: float = 0.001
    max_features: int = 12
    secondary_optimization: bool = True
    opt_method: str = 'L-BFGS-B'
    opt_iterations: int = 100
    use_sindy_pruning: bool = True
    sindy_threshold: float = 0.05
    enforce_hamiltonian_structure: bool = True
    estimate_dissipation: bool = True


@dataclass
class SystemConfig:
    """System-wide configuration."""
    device: str = 'auto'  # 'auto', 'cpu', 'cuda', 'mps'
    seed: int = 42
    verbose: bool = True
    save_checkpoints: bool = True
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    results_dir: str = './results'


@dataclass
class AutoTuneConfig:
    """Configuration for auto-tuning capabilities."""
    enabled: bool = True
    tuning_method: str = 'grid_search'  # 'grid_search', 'random_search', 'bayesian'
    n_trials: int = 20
    validation_split: float = 0.2
    metric: str = 'loss'  # 'loss', 'reconstruction', 'consistency', 'symbolic_confidence'
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4


class ConfigManager:
    """
    Centralized configuration manager for the Neural-Symbolic Discovery Pipeline.
    Handles loading, saving, validation, and auto-tuning of configurations.
    """
    
    def __init__(self):
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.symbolic_config = SymbolicConfig()
        self.system_config = SystemConfig()
        self.auto_tune_config = AutoTuneConfig()
        
        # Default parameter ranges for auto-tuning
        self.default_tuning_ranges = {
            'model': {
                'hidden_dim': [32, 64, 128],
                'latent_dim': [4, 6, 8],
                'n_super_nodes': [2, 4, 6]
            },
            'training': {
                'lr': [1e-4, 2e-4, 5e-4, 1e-3],
                'seq_len': [10, 15, 20],
                'grad_acc_steps': [1, 2, 4]
            },
            'symbolic': {
                'populations': [1000, 2000, 3000],
                'generations': [20, 40, 60],
                'max_features': [8, 12, 16]
            }
        }
    
    def load_from_file(self, config_path: str) -> 'ConfigManager':
        """Load configuration from a YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        # Load each configuration section
        if 'model' in config_dict:
            self.model_config = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            self.training_config = TrainingConfig(**config_dict['training'])
        if 'symbolic' in config_dict:
            self.symbolic_config = SymbolicConfig(**config_dict['symbolic'])
        if 'system' in config_dict:
            self.system_config = SystemConfig(**config_dict['system'])
        if 'auto_tune' in config_dict:
            self.auto_tune_config = AutoTuneConfig(**config_dict['auto_tune'])
        
        return self
    
    def save_to_file(self, config_path: str):
        """Save configuration to a YAML or JSON file."""
        config_dict = {
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'symbolic': asdict(self.symbolic_config),
            'system': asdict(self.system_config),
            'auto_tune': asdict(self.auto_tune_config)
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                json.dump(config_dict, f, indent=2)
    
    def get_parameter_grid(self, tuning_ranges: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Generate parameter grid for hyperparameter tuning."""
        if tuning_ranges is None:
            tuning_ranges = self.default_tuning_ranges
        
        # Combine all parameter ranges
        all_params = {}
        for section, params in tuning_ranges.items():
            for param, values in params.items():
                all_params[f"{section}.{param}"] = values
        
        # Generate grid
        grid = list(ParameterGrid(all_params))
        
        # Convert to nested dictionaries
        nested_grids = []
        for params in grid:
            nested_config = {}
            for key, value in params.items():
                section, param = key.split('.', 1)
                if section not in nested_config:
                    nested_config[section] = {}
                nested_config[section][param] = value
            nested_grids.append(nested_config)
        
        return nested_grids
    
    def apply_config(self, config_updates: Dict[str, Any]):
        """Apply configuration updates."""
        for section, updates in config_updates.items():
            if section == 'model':
                for param, value in updates.items():
                    setattr(self.model_config, param, value)
            elif section == 'training':
                for param, value in updates.items():
                    setattr(self.training_config, param, value)
            elif section == 'symbolic':
                for param, value in updates.items():
                    setattr(self.symbolic_config, param, value)
            elif section == 'system':
                for param, value in updates.items():
                    setattr(self.system_config, param, value)
            elif section == 'auto_tune':
                for param, value in updates.items():
                    setattr(self.auto_tune_config, param, value)
    
    def get_current_config(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return {
            'model': asdict(self.model_config),
            'training': asdict(self.training_config),
            'symbolic': asdict(self.symbolic_config),
            'system': asdict(self.system_config),
            'auto_tune': asdict(self.auto_tune_config)
        }
    
    def validate_config(self) -> List[str]:
        """Validate current configuration and return list of issues."""
        issues = []
        
        # Validate model config
        if self.model_config.latent_dim % 2 != 0 and self.model_config.hamiltonian:
            issues.append("latent_dim must be even for Hamiltonian systems")
        
        if self.model_config.n_super_nodes < self.model_config.min_active_super_nodes:
            issues.append("n_super_nodes must be >= min_active_super_nodes")
        
        # Validate training config
        if self.training_config.grad_acc_steps < 1:
            issues.append("grad_acc_steps must be >= 1")
        
        # Validate symbolic config
        if self.symbolic_config.populations < 100:
            issues.append("populations should be >= 100 for meaningful GP")
        
        if self.symbolic_config.generations < 10:
            issues.append("generations should be >= 10 for meaningful GP")
        
        return issues
    
    def suggest_optimal_config(self, problem_type: str = 'general') -> Dict[str, Any]:
        """
        Suggest optimal configuration based on problem type.
        
        Args:
            problem_type: Type of problem ('general', 'small', 'large', 'physics', 'chaos')
        
        Returns:
            Dictionary with suggested configuration updates
        """
        suggestions = {}
        
        if problem_type == 'small':
            # For small problems, use smaller networks and fewer resources
            suggestions['model'] = {
                'hidden_dim': 32,
                'n_super_nodes': 2,
                'latent_dim': 4
            }
            suggestions['training'] = {
                'populations': 1000,
                'generations': 20,
                'epochs': 500
            }
            suggestions['symbolic'] = {
                'populations': 1000,
                'generations': 20
            }
        elif problem_type == 'large':
            # For large problems, use larger networks and more resources
            suggestions['model'] = {
                'hidden_dim': 128,
                'n_super_nodes': 8,
                'latent_dim': 8
            }
            suggestions['training'] = {
                'epochs': 2000,
                'enable_gradient_accumulation': True,
                'grad_acc_steps': 4
            }
            suggestions['symbolic'] = {
                'populations': 3000,
                'generations': 60
            }
        elif problem_type == 'physics':
            # For physics problems, emphasize Hamiltonian structure
            suggestions['model'] = {
                'hamiltonian': True,
                'dissipative': True
            }
            suggestions['symbolic'] = {
                'enforce_hamiltonian_structure': True,
                'estimate_dissipation': True
            }
        elif problem_type == 'chaos':
            # For chaotic systems, emphasize stability and longer training
            suggestions['training'] = {
                'epochs': 1500,
                'warmup_epochs': 600,
                'align_anneal_epochs': 1500
            }
            suggestions['symbolic'] = {
                'secondary_optimization': True,
                'use_sindy_pruning': True
            }
        
        return suggestions
    
    def auto_tune(self, train_fn, validation_data, problem_type: str = 'general') -> Dict[str, Any]:
        """
        Perform automatic hyperparameter tuning.
        
        Args:
            train_fn: Training function that takes config and returns metrics
            validation_data: Validation data for evaluation
            problem_type: Type of problem for initial suggestions
        
        Returns:
            Best configuration found
        """
        if not self.auto_tune_config.enabled:
            return self.get_current_config()
        
        # Get initial suggestions based on problem type
        initial_suggestions = self.suggest_optimal_config(problem_type)
        self.apply_config(initial_suggestions)
        
        # Generate parameter grid for tuning
        param_grid = self.get_parameter_grid()
        
        best_config = None
        best_metric = float('inf') if self.auto_tune_config.metric != 'symbolic_confidence' else float('-inf')
        best_metrics = None
        
        print(f"Starting auto-tuning with {len(param_grid)} configurations...")
        
        for i, config_updates in enumerate(param_grid[:self.auto_tune_config.n_trials]):
            print(f"Trial {i+1}/{min(self.auto_tune_config.n_trials, len(param_grid))}")
            
            # Save current config
            original_config = copy.deepcopy(self.get_current_config())
            
            # Apply updates
            self.apply_config(config_updates)
            
            # Validate config
            issues = self.validate_config()
            if issues:
                print(f"  Skipping config due to issues: {issues}")
                continue
            
            try:
                # Train with current config
                metrics = train_fn(self, validation_data)
                
                # Extract target metric
                if self.auto_tune_config.metric in metrics:
                    current_metric = metrics[self.auto_tune_config.metric]
                    
                    # Update best if this is better
                    is_better = (
                        current_metric < best_metric if self.auto_tune_config.metric != 'symbolic_confidence' 
                        else current_metric > best_metric
                    )
                    
                    if is_better:
                        best_metric = current_metric
                        best_config = copy.deepcopy(self.get_current_config())
                        best_metrics = metrics
                        print(f"  New best {self.auto_tune_config.metric}: {current_metric}")
                else:
                    print(f"  Warning: {self.auto_tune_config.metric} not found in metrics")
                    
            except Exception as e:
                print(f"  Error during training: {e}")
                continue
            finally:
                # Restore original config
                self.apply_config(original_config)
        
        if best_config:
            # Apply best configuration
            self.apply_config(best_config)
            print(f"Auto-tuning completed. Best {self.auto_tune_config.metric}: {best_metric}")
            return best_config
        else:
            print("Auto-tuning failed to find a valid configuration")
            return self.get_current_config()


def create_default_config() -> ConfigManager:
    """Create a default configuration manager."""
    return ConfigManager()


def load_config(config_path: Optional[str] = None) -> ConfigManager:
    """Load configuration from file or create default."""
    config_manager = create_default_config()
    if config_path and os.path.exists(config_path):
        config_manager.load_from_file(config_path)
    return config_manager