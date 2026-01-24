# Neural-Symbolic Discovery Pipeline

This project implements a sophisticated **Neural-Symbolic Discovery Pipeline** for automated coarse-graining of particle dynamics. It bridges the gap between micro-scale particle dynamics and meso-scale symbolic equations using a stack of Graph Neural Networks (GNNs), Symplectic ODEs, and Genetic Programming.

## Overview

The pipeline consists of three main stages:

1. **Neural Encoder**: GNN-based encoder with hierarchical soft-assignment pooling to compress particle dynamics into super-node representations
2. **Latent Dynamics**: Hamiltonian-constrained ODE dynamics that preserve physical inductive biases
3. **Symbolic Distillation**: Genetic programming to extract interpretable symbolic equations from learned neural dynamics

## Key Features

### 1. Autonomous Physics Discovery
- **Bias-Free Search**: Successfully discovers power laws (like Lennard-Jones $1/r^6$ and $1/r^{12}$) from a minimal basis ($1/r, 1/r^2, 1/r^4$) without explicit prior inclusion.
- **SINDy-style Pruning**: Uses Sequential Thresholded Least Squares (STLSQ) with relative thresholding to prune feature space before Genetic Programming.
- **Dimensional Analysis Filter**: Enforces physical units consistency (L, M, T) during symbolic search using a dimensionality penalty.
- **Secondary Optimization**: L-BFGS refinement of physical constants for high-fidelity discovery.

### 2. Automated Multi-Objective Balancing
- **GradNorm Integration**: Dynamically scales task weights by normalizing gradient magnitudes, ensuring balanced learning across 17+ loss terms.
- **Uncertainty Weighting**: Uses learnable log-variances ($\sigma$) to automatically weight losses ($L = L \cdot e^{-s} + s$) from the start of the dynamics phase.
- **PCGrad Optimization**: Projected Conflicting Gradients to handle directional conflicts between Reconstruction, Physicality, and Sparsity objectives.

### 3. Stress Test & OOD Validation
- **Forecast Horizon**: New metric calculating the step count until symbolic trajectories deviate $>5\%$ from ground-truth variance.
- **Long-term Stability**: Verified via 2000-step shadow integration (4x training horizon) under OOD conditions.
- **Energy Conservation**: Verified near-zero energy drift ($< 10^{-10}$) and minimal symplectic drift.

### 4. Hamiltonian Inductive Bias
- **Enforced Separability**: Supports $H(q,p) = V(q) + \sum p^2/2$, ensuring $dq/dt = p$ is strictly maintained.
- **Canonical Equations**: Enforces $\dot{q} = \partial H/\partial p$ and $\dot{p} = -\partial H/\partial q$.
- **Apple Silicon Support**: Optimized for MPS with CPU-offloaded ODE integration for stability.

## Architecture

```
Particle Dynamics → GNN Encoder → Separable Latents → Hamiltonian ODE → Symbolic Equations
                    ↓              ↓                   ↓                ↓
                GradNorm        OOD Stress       Inductive         Closed-Loop
                Balancing       Testing          Biases            Consistency
```

## Success Metrics: Autonomous Discovery
The pipeline has successfully moved beyond "Physical Recovery" to **Autonomous Discovery** of the Lennard-Jones (LJ) potential.
- **Symbolic $R^2 > 0.95$**: Achieved using a reduced basis set (no hardcoded $1/r^{12}$).
- **Forecast Horizon**: 100% stability over 2000-step OOD stress tests.
- **Latent Correlation**: $> 0.99$ physical alignment between latent nodes and system center-of-mass.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Recommended: Unified Training Pipeline

```bash
python unified_train.py --particles 16 --super_nodes 4 --epochs 500 --steps 500 --sim lj --hamiltonian
```

### Validation & Stress Testing

Run the discovered equations through a 2000-step OOD stress test:

```bash
python validate_discovery.py --model_path ./results/model_latest.pt --results_path ./results/discovery_latest.json --steps 2000
```

## Files Overview

- `engine.py`: Training engine featuring **GradNormBalancer** and Uncertainty Weighting.
- `balanced_features.py`: Feature engineering with **Reduced Basis** for autonomous discovery.
- `enhanced_symbolic.py`: Symbolic regression with **STLSQ Pruning** and secondary optimization.
- `validate_discovery.py`: Stress test script calculating **Forecast Horizon**.
- `hamiltonian_symbolic.py`: Hamiltonian-specific symbolic distillation logic.
- `stable_pooling.py`: Sinkhorn-Knopp based hierarchical pooling for assignment stability.
- `unified_train.py`: Main entry point orchestrating the 3-stage pipeline.

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
