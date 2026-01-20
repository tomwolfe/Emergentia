# Discovery Engine: Meso-scale Physics Learning

This repository contains the **Discovery Engine**, a neural-symbolic pipeline designed to learn meso-scale (coarse-grained) physical laws from micro-scale particle simulations. It utilizes Graph Neural Networks (GNNs), Latent ODEs, and Symbolic Regression to bridge the gap between high-dimensional raw data and interpretable mathematical expressions.

## 🏗 System Architecture

The engine operates in three distinct phases:

1. **Simulation**: A `SpringMassSimulator` generates micro-scale trajectories of particles with support for Periodic Boundary Conditions (PBC) and Velocity Verlet integration.
2. **Discovery (Neural)**: A `DiscoveryEngineModel` uses Hierarchical Pooling to aggregate particles into "super-nodes". A Latent ODE then learns the continuous-time dynamics of these super-nodes.
3. **Distillation (Symbolic)**: A `SymbolicDistiller` extracts the learned latent derivatives and uses Genetic Programming to find the underlying symbolic equations.

---

## 🔬 Meso-scale Theory & Hierarchical Pooling

### Theoretical Background
Physical systems often exhibit different behaviors at different scales. While micro-scale dynamics (e.g., individual atoms) are governed by fundamental forces, the macro-scale (e.g., fluid flow) is described by coarse-grained equations like Navier-Stokes. The **Meso-scale** is the bridge between these two, where emergent structures (clusters of particles) form the basis of the dynamics.

### Hierarchical Pooling (Soft-Assignment)
To learn these meso-scale structures, the Discovery Engine employs a differentiable pooling layer. Instead of fixed clustering (like K-Means), it learns an **Assignment Matrix** $S \in \mathbb{R}^{N \times K}$ using a Gumbel-Softmax distribution:
- **$N$**: Number of micro-particles.
- **$K$**: Number of super-nodes (coarse-grained objects).
- **$S_{i,j}$**: The probability that particle $i$ belongs to super-node $j$.

By minimizing an entropy loss and an orthogonality constraint on $S$, the model is forced to find distinct, non-overlapping clusters that maximize spatial locality, effectively "discovering" the meso-scale objects.

---

## 🛠 Features

- **Hierarchical Soft-Assignment**: Uses a learned assignment matrix with **Spatial Separation** and **Graph Connectivity** constraints to discover contiguous physical objects.
- **Hamiltonian Latent ODE**: Enforces symplectic constraints ($\dot{q} = \partial H/\partial p, \dot{p} = -\partial H/\partial q$) in the latent space for energy-conserving meso-scale dynamics.
- **Hybrid Symbolic Distillation**: Combines **RandomForest-Lasso** feature selection with **Genetic Programming** and **SINDy-inspired pruning** for scalable and robust law discovery.
- **Adaptive Phased Training**: Employs a learnable multi-objective loss balancing scheme and a cooling schedule for Gumbel-Softmax temperature ($\tau$).
- **PBC Support**: Advanced neighbor discovery using KDTree tiling to handle periodic boundaries in simulation.



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

