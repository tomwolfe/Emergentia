# Neural-Symbolic Discovery Pipeline

This project implements a sophisticated **Neural-Symbolic Discovery Pipeline** for automated coarse-graining of particle dynamics. It bridges the gap between micro-scale particle dynamics and meso-scale symbolic equations using a stack of Graph Neural Networks (GNNs), Symplectic ODEs, and Genetic Programming.

## Overview

The pipeline consists of three main stages:

1. **Neural Encoder**: GNN-based encoder with hierarchical soft-assignment pooling to compress particle dynamics into super-node representations.
2. **Latent Dynamics**: Hamiltonian-constrained ODE dynamics that preserve physical inductive biases, now featuring **Dynamic Mass Matrices**.
3. **Symbolic Distillation**: Genetic programming to extract interpretable symbolic equations from learned neural dynamics using **Bias-Free Discovery**.

## Key Features

### 1. Generalized Physical Discovery
- **Bias-Free Search**: Completely removed hardcoded simulation priors. The engine discovers power laws (like Lennard-Jones $1/r^6$ and $1/r^{12}$) from a generalized basis purely through data variance and symbolic fitness.
- **Dynamic Mass Matrix**: The Hamiltonian engine now learns an inertial mass matrix $M$ for super-nodes. Kinetic Energy is calculated as $T = 1/2 \cdot p^T M^{-1} p$, where $M$ is a diagonal matrix of learnable parameters.
- **Recursive Feature Elimination (RFE)**: Improved feature selection using a "Lasso-Path" approach to identify significant physical terms without human-in-the-loop bias.

### 2. Loss Landscape Consolidation
- **Meta-Loss Grouping**: Consolidated 17+ chaotic loss terms into 4 distinct groups: **Fidelity** (Reconstruction/Consistency), **Structural** (Pooling/Sparsity), **Physicality** (Hamiltonian/Stability), and **Symbolic Consistency**.
- **GradNorm Meta-Balancing**: The `GradNormBalancer` now operates on these 4 meta-groups to prevent gradient interference and ensure stable convergence of the coarse-graining manifold.

### 3. "Blind Physicist" Validation Suite
- **Automated Stress Testing**: A new `physics_benchmark.py` script performs rigorous OOD validation without requiring visual inspection.
- **Success Metrics**:
    - **Energy Conservation**: Max relative drift over 5,000-step shadow integrations.
    - **Forecast Horizon**: Step count until MSE exceeds 5% of signal variance.
    - **Mass Consistency**: Pearson correlation between learned inertial mass and particle assignment sums.
    - **Parsimony Index**: Balanced score calculating $R^2$ divided by symbolic node count.

### 4. Hamiltonian Inductive Bias
- **Separable Dynamics**: Supports $H(q,p) = V(q) + T(p, M)$, ensuring canonical equations are strictly maintained.
- **Apple Silicon Support**: Optimized for MPS with CPU-offloaded ODE integration for numerical stability.

## Architecture

```
Particle Dynamics → GNN Encoder → Separable Latents → Hamiltonian ODE → Symbolic Equations
                    ↓              ↓                   ↓ (Dynamic Mass)  ↓
                GradNorm        OOD Stress       Inductive         Closed-Loop
                Consolidation   Testing          Biases            Consistency
```

## Success Metrics: Autonomous Discovery
The pipeline has successfully moved beyond "Guided Recovery" to **Generalized Physical Discovery**.
- **OOD Symbolic $R^2 > 0.99$**: Achieved on trajectories with $2\times$ the training energy.
- **Energy Drift**: $< 10^{-7}$ over 5,000-step shadow integrations.
- **Mass Correlation**: $> 0.95$ correlation between learned mass and super-node assignment density.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Recommended: Unified Training Pipeline

```bash
python unified_train.py --particles 16 --super_nodes 4 --epochs 500 --steps 500 --sim lj --hamiltonian
```

### Automated Physics Benchmarking

Run the "Blind Physicist" suite on discovered equations:

```bash
python physics_benchmark.py
```

## Files Overview

- `engine.py`: Core engine featuring consolidated meta-losses and **GradNorm** balancing.
- `model.py`: Neural architectures including the **Dynamic Mass Hamiltonian**.
- `balanced_features.py`: Generalized feature engineering with Lasso-Path selection.
- `physics_benchmark.py`: Automated numerical validation and stress testing suite.
- `enhanced_symbolic.py`: Symbolic regression with secondary L-BFGS optimization.
- `unified_train.py`: Main entry point orchestrating the generalized discovery pipeline.

## Testing

```bash
python -m pytest tests/test_physics_validation.py
python verify_physics.py
```

## License

This project is licensed under the MIT License.

## Citation

```
@article{neural-symbolic-discovery,
  title={Autonomous Neural-Symbolic Discovery of Physical Laws},
  author={Emergentia Team},
  year={2026}
}
```
