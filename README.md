# Discovery Engine: Meso-scale Physics Learning

This repository contains the **Discovery Engine**, a neural-symbolic pipeline designed to learn meso-scale (coarse-grained) physical laws from micro-scale particle simulations. It utilizes Graph Neural Networks (GNNs), Latent ODEs, and Symbolic Regression to bridge the gap between high-dimensional raw data and interpretable mathematical expressions.

## 🏗 System Architecture

The engine operates in three distinct phases:

1. **Simulation**: A `SpringMassSimulator` generates micro-scale trajectories of particles with support for Periodic Boundary Conditions (PBC) and Velocity Verlet integration.
2. **Discovery (Neural)**: A `DiscoveryEngineModel` uses Hierarchical Pooling to aggregate particles into "super-nodes". A Latent ODE then learns the continuous-time dynamics of these super-nodes.
3. **Distillation (Symbolic)**: A `SymbolicDistiller` extracts the learned latent derivatives and uses Genetic Programming to find the underlying symbolic equations.



---

## 🛠 Features

- **Hierarchical Soft-Assignment**: Uses a learned assignment matrix to preserve spatial locality during coarse-graining.
- **Continuous Dynamics**: Employs `torchdiffeq` to model latent states as an autonomous ODE system.
- **PBC Support**: Advanced neighbor discovery using KDTree tiling to handle periodic boundaries in simulation.
- **Pareto-Optimal Training**: Configurable loss weights to balance reconstruction, latent consistency, and state-assignment stability.
- **Two-Stage Distillation**: Implements a "Coarse-to-Fine" symbolic search to optimize computational resources.



---

## 🚀 Getting Started

### Prerequisites

Install the required dependencies listed in `requirements.txt`:

```bash
pip install torch torch-geometric torchdiffeq gplearn numpy matplotlib scipy pandas
```

### Running the Discovery Pipeline

To train the model and distill symbolic laws from a spring-mass system, run:

```bash
python main.py
```

---

## 📊 File Structure

| File | Description |
| --- | --- |
| `main.py` | The primary entry point; manages the full end-to-end pipeline. |
| `model.py` | Contains the GNN Encoder, Hierarchical Pooling, Latent ODE, and Decoder. |
| `engine.py` | Handles data preparation, normalization, and the `Trainer` class. |
| `simulator.py` | A physics engine for mass-spring-damper systems with PBC support. |
| `symbolic.py` | Uses Genetic Programming to distill latent ODEs into symbolic equations. |


---

## ⚖️ License

This project is licensed under the **MIT License**.

Copyright (c) 2026 Thomas Wolfe.

