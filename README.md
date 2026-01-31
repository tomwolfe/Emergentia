# 🌌 Emergentia: Meso-scale Discovery Engine

**Emergentia** is a Neural-Symbolic discovery engine designed to extract parsimonious physical laws from meso-scale particle trajectories. By combining the flexible representation power of **Deep Learning** with the mathematical clarity of **Symbolic Regression**, Emergentia "rediscovers" the underlying equations of motion from raw simulation data, even in high-noise environments.

---

## ✨ Key Features

* **🧩 Modular Physics Engine:** A plug-and-play architecture for physical potentials. Easily define new laws (e.g., Harmonic, Lennard-Jones, Morse, Gravity, Buckingham, Yukawa) by extending the base `Potential` class.
* **🧠 Neural-Symbolic Pipeline:**
    1.  **Neural Mapping:** A `DiscoveryNet` (PyTorch) approximates complex, non-linear force laws using an expanded, configurable basis set.
    2.  **Symbolic Distillation:** A `SymbolicRegressor` (gplearn) extracts clean, human-readable, and mathematically interpretable formulas from the neural network's learned weights.
* **🛡️ Noise Resilience:** Built-in support for discovery from noisy trajectories using robust `HuberLoss` training and automated Gaussian smoothing.
* **📊 Robust Validation:** Automated verification of discovered laws using R² scores, Mean Squared Error (MSE), and the Bayesian Information Criterion (BIC) to ensure both accuracy and parsimony.
* **⚡ Hardware Acceleration:** Full support for `CUDA` (NVIDIA) and `MPS` (Apple Silicon) backends for fast training and simulation.
* **📉 Symmetric Log Transform:** Advanced training techniques to handle high-dynamic-range forces (e.g., singularities near `r=0`) without numerical instability.
* **🧪 Comprehensive Testing:** A full suite of unit and integration tests verify physics integrity, scaling logic, registry consistency, and discovery robustness.

---

## 🚀 Performance Benchmarks

Emergentia achieves high-fidelity results across multiple physical regimes. Benchmarks are run with 3 particles in 2D or 3D over 2000 steps, using 3 trials per noise level.

| Mode | Basis Functions | Target Law Example | Success Rate (0.01 noise) | R² (0.01 noise) |
| :--- | :--- | :--- | :--- | :--- |
| **Spring** | `['1', 'r']` | F = -k(r - r₀) | >99% | >0.99 |
| **Lennard-Jones** | `['1', '1/r^7', '1/r^13']` | F = 48ε(σ¹²/r¹³ - σ⁶/r⁷) | ~95% | >0.95 |
| **Morse** | `['1', 'exp(-r)']` | F = 2De·a·(e^(-a(r-re)) - e^(-2a(r-re))) | >99% | >0.99 |
| **Gravity** | `['1', '1/r^2']` | F = -G/r² | >99% | >0.99 |
| **Buckingham** | `['1', '1/r^7', 'exp(-r)']` | F = AB·e^(-Br) - 6C/r⁷ | ~90% | >0.90 |
| **Yukawa** | `['1/r', '1/r^2', 'exp(-r)/r']` | F = A·e^(-Br)·(B/r + 1/r²) | ~90% | >0.90 |
| **Mixed** | `['1', 'r', '1/r^2']` | F = -k(r - r₀) - G/r² | >95% | >0.95 |

> **Note:** Performance data is based on standard validation trials (3 trials, 2000 steps, 0.01 noise). See `results/benchmark_summary.csv` for detailed metrics.

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

# Run all tests
pytest tests/
```

### 📂 Project Structure
* `emergentia/`: Core package containing the discovery logic.
    * `simulator.py`: Modular physics simulation using Velocity Verlet integration.
    * `models.py`: `DiscoveryNet` architecture and `TrajectoryScaler`.
    * `engine.py`: The `DiscoveryPipeline` linking neural training to symbolic regression.
    * `registry.py`: Centralized physical basis functions (Torch, NumPy, SymPy).
    * `utils.py`: Statistical verification and symbolic utility functions.
* `run_benchmarks.py`: Main entry point for cross-regime validation.
* `tests/`: Comprehensive test suite.
* `results/`: Directory for benchmark reports and summaries (auto-generated).
* `.gitignore`: Standard Python and project-specific ignore patterns.
* `LICENSE`: MIT License.

---

## 📜 License
Distributed under the **MIT License**. See `LICENSE` for more information.