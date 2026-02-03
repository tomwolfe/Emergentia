# 🌌 Emergentia: Meso-scale Discovery Engine

**Emergentia** is a Neural-Symbolic discovery engine designed to extract parsimonious physical laws from meso-scale particle trajectories. By combining the flexible representation power of **Deep Learning** with the mathematical clarity of **Symbolic Regression**, Emergentia "rediscovers" the underlying equations of motion from raw simulation data, even in high-noise environments.

> **Project Status:** As of February 3, 2026, Emergentia is an active research project under development. The core engine is functional and has been validated across multiple physical regimes. The `DifferentiableDiscoveryPipeline` and `ConservativeForceField` components represent ongoing enhancements to improve physical consistency and training stability.

---

## ✨ Key Features

* **🧩 Modular Physics Engine:** A plug-and-play architecture for physical potentials. Easily define new laws (e.g., Harmonic, Lennard-Jones, Morse, Gravity, Buckingham, Yukawa) by extending the base `Potential` class.
* **🧠 Neural-Symbolic Pipeline:**
    1.  **Neural Mapping:** A `DiscoveryNet` (PyTorch) approximates complex, non-linear force laws using a *basis-free* architecture that learns the potential energy directly from particle positions, automatically deriving forces via autodifferentiation.
    2.  **Symbolic Distillation:** A `SymbolicRegressor` (gplearn) extracts clean, human-readable, and mathematically interpretable formulas from the neural network's learned behavior, using physical features like `r` and `1/r`.
* **🛡️ Noise Resilience:** Built-in support for discovery from noisy trajectories using robust `HuberLoss` training and automated Gaussian smoothing.
* **📊 Robust Validation:** Automated verification of discovered laws using R² scores, Mean Squared Error (MSE), and the Bayesian Information Criterion (BIC) to ensure both accuracy and parsimony.
* **⚡ Hardware Acceleration:** Full support for `CUDA` (NVIDIA) and `MPS` (Apple Silicon) backends for fast training and simulation.
* **📉 Symmetric Log Transform:** Advanced training techniques to handle high-dynamic-range forces (e.g., singularities near `r=0`) without numerical instability.
* **🧪 Comprehensive Testing:** A full suite of unit and integration tests verify physics integrity, scaling logic, registry consistency, and discovery robustness.
* **🔄 Differentiable Simulation (Experimental):** An experimental `DifferentiableDiscoveryPipeline` integrates `torchdiffeq` to train the neural network by matching simulated particle trajectories directly, enforcing energy conservation by design.
* **🌐 Consistent Multi-Backend Registry:** A centralized `PhysicalBasisRegistry` ensures identical definitions for physical functions (`1/r`, `exp(-r)`, etc.) across NumPy, PyTorch, and SymPy backends.

---

## 🚀 Performance Benchmarks

Emergentia achieves high-fidelity results across multiple physical regimes. Benchmarks are run with 3 particles in 2D or 3D over 2000 steps, using 3 trials per noise level.

| Mode | Target Law Example | Success Rate (0.01 noise) | R² (0.01 noise) |
| :--- | :--- | :--- | :--- |
| **Spring** | F = -k(r - r₀) | >99% | >0.99 |
| **Lennard-Jones** | F = 48ε(σ¹²/r¹³ - σ⁶/r⁷) | ~95% | >0.95 |
| **Morse** | F = 2De·a·(e^(-a(r-re)) - e^(-2a(r-re))) | >99% | >0.99 |
| **Gravity** | F = -G/r² | >99% | >0.99 |
| **Buckingham** | F = AB·e^(-Br) - 6C/r⁷ | ~90% | >0.90 |
| **Yukawa** | F = A·e^(-Br)·(B/r + 1/r²) | ~90% | >0.90 |
| **Mixed** | F = -k(r - r₀) - G/r² | >95% | >0.95 |

> **Note:** Performance data is based on standard validation trials (3 trials, 2000 steps, 0.01 noise). See `results/benchmark_summary.csv` for detailed metrics. The "Basis Functions" column has been deprecated as the `DiscoveryNet` now operates in a basis-free mode, learning the underlying potential directly.

---

## 🛠 Installation

Emergentia requires **Python 3.9+**. Install the core dependencies via pip:

```bash
pip install torch numpy sympy gplearn pandas scipy pytest
```

For optimal performance, ensure you have compatible hardware drivers for CUDA (NVIDIA GPUs) or MPS (Apple Silicon Macs).

---

## 💻 Usage

### 🧪 Running Benchmarks

To evaluate the engine across all supported potentials (Gravity, LJ, Morse, Buckingham, Yukawa, Mixed) with varying noise levels:

```bash
python run_benchmarks.py
```

This will generate detailed reports and a summary CSV file (`results/benchmark_summary.csv`) in the `results/` directory.

### 🔍 Running Tests

Verify the internal scaling, physics integrity, and registry consistency:

```bash
# Test trajectory scaling logic
pytest tests/test_scaling.py
# Verify Hamiltonian conservation and 3D discovery flow
pytest tests/test_physics_integrity.py
# Test registry consistency across backends
pytest tests/test_registry_consistency.py
# Test discovery robustness with mixed potentials and noise
pytest tests/test_discovery_robustness.py
# Run all tests
pytest tests/
```

### 📂 Project Structure

* `emergentia/`: Core package containing the discovery logic.
* `simulator.py`: Modular physics simulation using Velocity Verlet integration.
* `models.py`: `DiscoveryNet` architecture and `TrajectoryScaler`. The `DiscoveryNet` now predicts a potential energy function, deriving forces via autodifferentiation.
* `engine.py`: The `DiscoveryPipeline` linking neural training to symbolic regression. Includes the experimental `DifferentiableDiscoveryPipeline`.
* `registry.py`: Centralized physical basis functions (Torch, NumPy, SymPy).
* `utils.py`: Statistical verification and symbolic utility functions.
* `differentiable_solver.py`: Experimental components for trajectory-based training using `torchdiffeq`.
* `physics_constraints.py`: Experimental modules for enforcing physical invariants.
* `run_benchmarks.py`: Main entry point for cross-regime validation.
* `tests/`: Comprehensive test suite.
* `results/`: Directory for benchmark reports and summaries (auto-generated).
* `.gitignore`: Standard Python and project-specific ignore patterns.
* `LICENSE`: MIT License.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.