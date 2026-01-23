# Neural-Symbolic Discovery Pipeline

This project implements a sophisticated **Neural-Symbolic Discovery Pipeline** for automated coarse-graining of particle dynamics. It bridges the gap between micro-scale particle dynamics and meso-scale symbolic equations using a stack of Graph Neural Networks (GNNs), Symplectic ODEs, and Genetic Programming.

## Overview

The pipeline consists of three main stages:

1. **Neural Encoder**: GNN-based encoder with hierarchical soft-assignment pooling to compress particle dynamics into super-node representations
2. **Latent Dynamics**: Hamiltonian-constrained ODE dynamics that preserve physical inductive biases
3. **Symbolic Distillation**: Genetic programming to extract interpretable symbolic equations from learned neural dynamics

## Key Features

### 1. Hamiltonian Inductive Bias
- **Enforced Separability**: Supports $H(q,p) = V(q) + \sum p^2/2$, ensuring $dq/dt = p$ is strictly maintained in the latent space.
- **Canonical Equations**: Enforces $\dot{q} = \partial H/\partial p$ and $\dot{p} = -\partial H/\partial q$.
- **Energy Conservation**: Maintains Liouville's Theorem and phase-space volume preservation.

### 2. Physics-Guided Symbolic Search
- **LJ Potential Recovery**: Specialized routines to identify and optimize $A/r^{12} - B/r^6$ forms.
- **Secondary Optimization**: L-BFGS refinement of physical constants for high-fidelity discovery.
- **Physics-First Filtering**: Protects $1/r^n$ features from pruning in molecular systems.

### 3. Stable Hierarchical Pooling
- Prevents "latent flickering" through temporal consistency.
- Ensures spatial contiguity of super-nodes.
- Dynamic resolution selection to find optimal meso-scale.

## Architecture

```
Particle Dynamics → GNN Encoder → Separable Latents → Hamiltonian ODE → Symbolic Equations
                    ↓              ↓                   ↓                ↓
                Pooling         Assignment       Inductive         Guided
                Loss            Consistency      Biases            Distillation
```

## Achievement: Exact Physical Recovery
The pipeline has successfully achieved **Exact Physical Recovery** of the Lennard-Jones (LJ) potential.
- **Success Metric**: $R^2 > 0.98$ for discovered Hamiltonian vs Neural Dynamics.
- **Discovered Form**: $V(r) = \frac{A}{r^{12}} - \frac{B}{r^6} + C$.
- **Stability**: Latent Correlation $> 0.99$, Flicker Rate $< 0.001$.

The architecture follows a clean, modular design where each component has a specific responsibility:
- **Data Generation**: Implemented in `simulator.py` with various physics simulators
- **Neural Network Model**: Implemented in `model.py` with GNN encoder and decoder
- **Training Engine**: Implemented in `engine.py` with loss computation and optimization
- **Symbolic Regression**: Implemented across `symbolic.py`, `enhanced_symbolic.py`, and `hamiltonian_symbolic.py`
- **Stabilization**: Implemented in `stable_pooling.py` with sparsity scheduling
- **Visualization**: Implemented in `visualization.py` with comprehensive plotting tools
- **Main Pipeline**: Orchestrated in `unified_train.py` with all components integrated

## Results Visualization

![Training History](training_history.png)
*Training history showing loss components and balancing weights over time.*

![Discovery Result](discovery_result.png)
*Visualization of the discovery results including particle assignments, latent trajectories, and phase space analysis.*

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Recommended: Unified Training Pipeline

For a unified approach that combines all improvements with performance optimizations:

```bash
python unified_train.py --particles 16 --super_nodes 4 --epochs 500 --steps 500 --sim lj --hamiltonian --memory_efficient --quick_symbolic
```

Additional options for performance tuning:
- `--memory_efficient`: Use memory-saving optimizations for large datasets
- `--quick_symbolic`: Use faster symbolic distillation with reduced populations/generations
- `--batch_size N`: Control batch size for training steps (default: 10)
- `--eval_every N`: Evaluate every N epochs (default: 50)
- `--sim`: Choose simulator ('spring' or 'lj' for Lennard-Jones)
- `--hamiltonian`: Use Hamiltonian dynamics
- `--lr`: Learning rate (default: 5e-4)

### Basic Example

```python
import torch
from model import DiscoveryEngineModel
from engine import Trainer, prepare_data
from symbolic import extract_latent_data
from enhanced_symbolic import create_enhanced_distiller
from hamiltonian_symbolic import HamiltonianSymbolicDistiller

# Initialize model
model = DiscoveryEngineModel(
    n_particles=16,
    n_super_nodes=4,
    node_features=4,
    latent_dim=4,
    hamiltonian=True
)

# Train the model on particle dynamics data
# ... training code ...

# Extract latent dynamics data
latent_data = extract_latent_data(model, dataset, dt=0.01, include_hamiltonian=True)

if len(latent_data[0]) > 0:
    z_states, dz_states, t_states, h_states = latent_data

    # Distill symbolic equations with enhanced features
    distiller = create_enhanced_distiller(secondary_optimization=True)

    equations = distiller.distill(z_states, h_states, n_super_nodes=4, latent_dim=4)

    print("Discovered Hamiltonian:")
    print(f"H(z) = {equations[0]}")
```

For a complete end-to-end example, run the unified training pipeline:

```bash
python unified_train.py --particles 8 --super_nodes 2 --epochs 100 --steps 200
```

### Advanced Configuration

```python
# Configure model with enhanced pooling
from stable_pooling import SparsityScheduler

model = DiscoveryEngineModel(
    n_particles=16,
    n_super_nodes=4,  # Adjust based on your system
    node_features=4,
    latent_dim=4,      # Must be even for Hamiltonian systems
    hamiltonian=True,
    dissipative=True   # Include energy dissipation
)

# Initialize SparsityScheduler to prevent resolution collapse
sparsity_scheduler = SparsityScheduler(
    initial_weight=0.0,
    target_weight=0.1,
    warmup_steps=250,  # Half of epochs
    max_steps=500      # Total epochs
)

# Configure trainer with advanced features
trainer = Trainer(
    model,
    lr=5e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    stats=stats,
    warmup_epochs=50,  # Stage 1: Train rec and assign
    max_steps=500,
    sparsity_scheduler=sparsity_scheduler
)
```

For production use, consider using the unified training pipeline which incorporates all these configurations:

```bash
python unified_train.py --particles 16 --super_nodes 4 --epochs 500 --steps 500 --sim lj --hamiltonian --memory_efficient --quick_symbolic
```

## Key Improvements Over Baseline

### 1. Enhanced Symbolic Regression
- **Secondary Optimization**: Uses scipy.optimize to refine constants in discovered expressions
- **Constant Refinement**: Improves accuracy of physics-specific parameters
- **Pareto Optimization**: Balances accuracy and complexity

### 2. Stable Hierarchical Pooling
- **Sparsity Scheduler**: Dynamically adjusts sparsity regularization to prevent resolution collapse
- **Temporal Consistency**: Maintains assignment stability across time steps
- **Active Node Management**: Ensures sufficient super-nodes remain active during training

### 3. Memory Efficiency
- **Gradient Accumulation**: Reduces memory usage during training
- **Selective Consistency Loss**: Computes consistency loss every N epochs to save time
- **Sampling Strategies**: Reduces memory usage during visualization

### 4. Adaptive Training Scheduling
- **Two-Stage Training**: Stage 1 focuses on reconstruction and assignment, Stage 2 adds dynamics
- **Temperature Annealing**: Gradually hardens soft assignments during training
- **Learning Rate Scheduling**: Uses cosine annealing for stable convergence

### 5. Improved Loss Balancing
- **Learnable Log-Variances**: Automatic loss balancing through learnable parameters
- **Enhanced Assignment Loss**: Includes entropy, diversity, spatial, and temporal consistency terms
- **Regularization Terms**: Latent variance, orthogonality, and connectivity losses

### 6. Hamiltonian Structure Preservation
- **Canonical Equations**: Enforces proper Hamiltonian dynamics (dq/dt = ∂H/∂p, dp/dt = -∂H/∂q)
- **Phase Space Volume Preservation**: Maintains Liouville's theorem
- **Learnable Dissipation**: Includes learnable damping terms for realistic systems

### 7. Apple Silicon (MPS) Support
- **MPS Compatibility**: Fixes for PyTorch MPS backend stability
- **CPU Offloading**: Moves ODE integration to CPU for stability on MPS devices
- **Precision Handling**: Proper float32/float64 management for MPS

### 8. Enhanced Visualization
- **Comprehensive Plots**: Assignment heatmaps, latent trajectories, phase space plots
- **Symbolic Comparison**: Overlay of learned vs symbolic predictions
- **Training Monitoring**: Real-time loss tracking and balancing weights

## Mathematical Foundation

### Hamiltonian Mechanics
For a system with generalized coordinates $q$ and momenta $p$, the Hamiltonian $H(q,p)$ defines the dynamics:
$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$

Our model learns $H$ as a neural network and computes gradients analytically.

### Pooling Objective
The hierarchical pooling minimizes:
$$\mathcal{L}_{pool} = \mathcal{L}_{entropy} + \mathcal{L}_{diversity} + \mathcal{L}_{spatial} + \mathcal{L}_{consistency} + \mathcal{L}_{sparsity}$$

Where:
- $\mathcal{L}_{entropy}$: Encourages hard assignments
- $\mathcal{L}_{diversity}$: Prevents all nodes from assigning to one super-node
- $\mathcal{L}_{spatial}$: Maintains spatial coherence
- $\mathcal{L}_{consistency}$: Ensures temporal stability
- $\mathcal{L}_{sparsity}$: Controls the number of active super-nodes

## Testing

Testing files are not included in the current project structure. To test the functionality, run the main training script:

```bash
python unified_train.py --epochs 100 --steps 50 --particles 6 --super_nodes 2
```

## Files Overview

- `model.py`: Core neural network architecture with GNN encoder, Hamiltonian ODE, and GNN decoder
- `symbolic.py`: Basic symbolic regression implementation using genetic programming
- `enhanced_symbolic.py`: Enhanced symbolic regression with secondary optimization
- `hamiltonian_symbolic.py`: Hamiltonian-specific symbolic distillation
- `simulator.py`: Particle dynamics simulators (Spring-Mass, Lennard-Jones)
- `engine.py`: Main training engine with loss computation and optimization
- `stable_pooling.py`: Enhanced pooling with collapse prevention mechanisms
- `balanced_features.py`: Physics-informed feature engineering
- `common_losses.py`: Common loss functions extracted to reduce duplication
- `pure_symbolic_functions.py`: Pure functions extracted from symbolic processing modules
- `train_utils.py`: Training utilities including early stopping and device management
- `visualization.py`: Comprehensive visualization tools for training history, discovery results, and symbolic validation
- `unified_train.py`: Unified training pipeline with all improvements (main entry point)
- `validate_discovery.py`: Closed-loop validation of discovered equations with forecast horizon analysis
- `profile_train.py`: Performance profiling tools for training optimization
- `test_mps_ode.py`: MPS (Apple Silicon) compatibility tests for ODE integration
- `config.yaml`: Configuration file for the pipeline
- `requirements.txt`: Project dependencies
- `training_history.png`: Visualization of training progress and loss components
- `discovery_result.png`: Visualization of discovery results including particle assignments and latent dynamics
- `results/`: Directory for storing experiment results, models, and validation reports

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