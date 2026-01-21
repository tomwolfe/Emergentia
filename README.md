# Neural-Symbolic Discovery Pipeline

This project implements a sophisticated **Neural-Symbolic Discovery Pipeline** for automated coarse-graining of particle dynamics. It bridges the gap between micro-scale particle dynamics and meso-scale symbolic equations using a stack of Graph Neural Networks (GNNs), Symplectic ODEs, and Genetic Programming.

## Overview

The pipeline consists of three main stages:

1. **Neural Encoder**: GNN-based encoder with hierarchical soft-assignment pooling to compress particle dynamics into super-node representations
2. **Latent Dynamics**: Hamiltonian-constrained ODE dynamics that preserve physical inductive biases
3. **Symbolic Distillation**: Genetic programming to extract interpretable symbolic equations from learned neural dynamics

## Key Features

### 1. Hamiltonian Inductive Bias
- Enforces canonical equations: $\dot{q} = \partial H/\partial p$ and $\dot{p} = -\partial H/\partial q$
- Maintains Liouville's Theorem (phase-space volume preservation)
- Includes learnable dissipation terms for realistic systems

### 2. Stable Hierarchical Pooling
- Prevents "latent flickering" through temporal consistency
- Ensures spatial contiguity of super-nodes
- Dynamic resolution selection to find optimal meso-scale

### 3. Enhanced Symbolic Regression
- Physics-informed feature engineering
- Secondary optimization for constant refinement
- Coordinate alignment between neural and physical spaces
- Hamiltonian structure preservation

## Architecture

```
Particle Dynamics → GNN Encoder → Super-Node Latents → Hamiltonian ODE → Symbolic Equations
                    ↓              ↓                   ↓                ↓
                Pooling         Assignment       Symplectic        Genetic
                Loss            Consistency      Constraints       Programming
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Recommended: Enhanced Execution

For the most advanced execution with all improvements, use the enhanced version:

```bash
python main_enhanced_complete.py --config config.yaml --problem_type physics
```

The enhanced version includes:
- Learnable basis functions to address the basis function bottleneck
- Optimized ODE functions to reduce adjoint sensitivity complexity
- Improved hyperparameter management with auto-tuning
- Enhanced symbolic distillation with secondary optimization
- Robust symbolic proxy with validation
- Multi-scale loss balancing
- Configurable problem type optimization

### Alternative: Fast Execution

For faster execution with performance optimizations, use the optimized version:

```bash
python fast_train.py --epochs 1000 --steps 200 --particles 8 --super_nodes 2
```

The optimized version includes:
- Gradient accumulation for memory efficiency
- Selective consistency loss computation
- Efficient ODE solving with looser tolerances during training
- Early stopping to prevent overfitting
- Edge caching for faster data preparation

### Original Version

The original version is still available for reference:

```bash
python main.py --epochs 1000 --steps 200 --particles 8 --super_nodes 2
```

Note: The original version is significantly slower and may timeout with larger parameters.

### Basic Example

```python
import torch
from model import DiscoveryEngineModel
from symbolic import SymbolicDistiller
from enhanced_symbolic import EnhancedSymbolicDistiller
from coordinate_mapping import AlignedHamiltonianSymbolicDistiller
from stable_pooling import StableHierarchicalPooling

# Initialize model
model = DiscoveryEngineModel(
    n_particles=100,
    n_super_nodes=10,
    node_features=4,
    latent_dim=4,
    hamiltonian=True
)

# Train the model on particle dynamics data
# ... training code ...

# Extract latent dynamics data
latent_states, latent_derivs = extract_latent_data(model, dataset, dt=0.01)

# Distill symbolic equations with enhanced features
distiller = EnhancedSymbolicDistiller(
    populations=2000,
    generations=40,
    secondary_optimization=True
)

equations = distiller.distill_with_secondary_optimization(
    latent_states=latent_states,
    targets=latent_derivs,
    n_super_nodes=10,
    latent_dim=4
)

# For Hamiltonian systems with coordinate alignment
hamiltonian_distiller = AlignedHamiltonianSymbolicDistiller(
    populations=2000,
    generations=40,
    enforce_hamiltonian_structure=True
)

# Fit coordinate mapper if physical coordinates are available
# hamiltonian_distiller.fit_coordinate_mapper(neural_latents, physical_coords)

# Distill with alignment
equations = hamiltonian_distiller.distill_with_alignment(
    neural_latents=latent_states,
    targets=latent_derivs,
    n_super_nodes=10,
    latent_dim=4
)
```

### Advanced Configuration

```python
# Configure model with enhanced pooling
from stable_pooling import StableHierarchicalPooling, DynamicLossBalancer

model = DiscoveryEngineModel(
    n_particles=100,
    n_super_nodes=15,  # Adjust based on your system
    node_features=4,
    latent_dim=6,      # Must be even for Hamiltonian systems
    hamiltonian=True,
    dissipative=True   # Include energy dissipation
)

# Configure enhanced symbolic distillation
distiller = EnhancedSymbolicDistiller(
    populations=3000,           # Larger population for complex systems
    generations=50,             # More generations for better exploration
    secondary_optimization=True, # Refine constants with scipy optimization
    opt_method='L-BFGS-B',      # Optimization algorithm
    opt_iterations=200          # Max iterations for optimization
)

# Use coordinate alignment for better interpretability
coord_distiller = AlignedHamiltonianSymbolicDistiller(
    populations=2500,
    generations=45,
    enforce_hamiltonian_structure=True
)
```

## Key Improvements Over Baseline

### 1. Enhanced Symbolic Regression
- **Secondary Optimization**: Uses scipy.optimize to refine constants in discovered expressions
- **Constant Refinement**: Improves accuracy of physics-specific parameters
- **Pareto Optimization**: Balances accuracy and complexity

### 2. Coordinate Alignment
- **Neural-Physical Mapping**: Aligns neural latent space with interpretable physical coordinates
- **Rotation Invariance**: Handles rotated coordinate systems learned by the encoder
- **Hamiltonian Structure Preservation**: Maintains proper q/p coordinate relationships

### 3. Collapse Prevention
- **Dynamic Loss Balancing**: Adjusts loss weights during training to prevent mode collapse
- **Minimum Active Nodes**: Ensures sufficient super-nodes remain active
- **Balance Loss**: Encourages uniform usage of super-nodes

### 4. Learnable Basis Functions
- **Adaptive Feature Generation**: Learns novel functional forms beyond predefined primitives
- **Attention Mechanisms**: Dynamically selects relevant basis functions based on input
- **Residual Connections**: Maintains information flow through the basis expansion

### 5. Optimized ODE Computation
- **Reduced Adjoint Complexity**: Efficient gradient computation for Hamiltonian systems
- **Memory-Efficient Solvers**: Trade accuracy for reduced memory usage when needed
- **Adaptive Integration**: Chooses appropriate solver based on system properties

### 6. Hyperparameter Auto-Tuning
- **Configurable Problem Types**: Optimizes for small, large, physics, or chaotic systems
- **Automatic Parameter Search**: Finds optimal hyperparameters for specific problems
- **Validation-Based Selection**: Uses held-out data to select best configurations

## Mathematical Foundation

### Hamiltonian Mechanics
For a system with generalized coordinates $q$ and momenta $p$, the Hamiltonian $H(q,p)$ defines the dynamics:
$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$

Our model learns $H$ as a neural network and computes gradients analytically.

### Pooling Objective
The hierarchical pooling minimizes:
$$\mathcal{L}_{pool} = \mathcal{L}_{entropy} + \mathcal{L}_{diversity} + \mathcal{L}_{spatial} + \mathcal{L}_{consistency} + \mathcal{L}_{collapse}$$

Where:
- $\mathcal{L}_{entropy}$: Encourages hard assignments
- $\mathcal{L}_{diversity}$: Prevents all nodes from assigning to one super-node
- $\mathcal{L}_{spatial}$: Maintains spatial coherence
- $\mathcal{L}_{consistency}$: Ensures temporal stability
- $\mathcal{L}_{collapse}$: Prevents resolution collapse

## Testing

Run the test suite:

```bash
python -m pytest test_*.py -v
```

Or run individual tests:

```bash
python test_symbolic.py
python test_ode.py
python test_pareto.py
python test_implemented_fixes.py
```

## Files Overview

- `model.py`: Core neural network architecture
- `symbolic.py`: Basic symbolic regression implementation
- `enhanced_symbolic.py`: Enhanced symbolic regression with secondary optimization
- `coordinate_mapping.py`: Neural-physical coordinate alignment
- `stable_pooling.py`: Enhanced pooling with collapse prevention
- `hamiltonian_symbolic.py`: Hamiltonian structure preservation
- `balanced_features.py`: Physics-informed feature engineering
- `optimized_symbolic.py`: Optimized symbolic dynamics with caching
- `simulator.py`: Particle dynamics simulators
- `engine.py`: Main discovery engine
- `learnable_basis.py`: Learnable basis functions to address basis function bottleneck
- `optimized_ode.py`: Optimized ODE functions to reduce adjoint sensitivity complexity
- `config_manager.py`: Configuration management and hyperparameter auto-tuning
- `main_enhanced_complete.py`: Complete enhanced pipeline with all improvements
- `config.yaml`: Configuration file for the enhanced pipeline

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@article{neural-symbolic-discovery,
  title={Neural-Symbolic Discovery of Physical Laws from Particle Dynamics},
  author={Emergentia Team},
  year={2026}
}
```

## Acknowledgments

- Inspired by recent advances in neural-symbolic integration
- Built on top of PyTorch Geometric and gplearn
- Thanks to the physics and ML communities for continued inspiration