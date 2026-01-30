# 🌌 Emergentia: Meso-scale Discovery Engine

**Emergentia** is a Neural-Symbolic discovery engine designed to extract parsimonious physical laws from meso-scale particle trajectories. By combining the flexible representation power of **Deep Learning** with the mathematical clarity of **Symbolic Regression**, Emergentia "rediscovers" the underlying equations of motion from raw simulation data, even in high-noise environments.

---

## ✨ Key Features

* 
**🧩 Modular Physics Engine:** A plug-and-play architecture for physical potentials. Easily define new laws (e.g., Harmonic, Lennard-Jones, Morse, Gravity) by extending the base `Potential` class.


* 
**🧠 Neural-Symbolic Pipeline:** 1.  **Neural Mapping:** A `DiscoveryNet` (PyTorch) approximates force laws using an expanded basis set, such as .
2.  **Symbolic Distillation:** A `SymbolicRegressor` (gplearn) extracts clean, human-readable mathematical formulas from the neural weights.


* 
**🛡️ Noise Resilience:** Built-in support for discovery from noisy trajectories using robust `HuberLoss` training and automated Gaussian smoothing.


* 
**📊 Robust Validation:** Automated verification of discovered laws using  scores, Mean Squared Error (MSE), and the Bayesian Information Criterion (BIC) to ensure both accuracy and parsimony.


* 
**⚡ Hardware Acceleration:** Full support for `CUDA` (NVIDIA) and `MPS` (Apple Silicon) backends.


* 
**📉 Symmetric Log Transform:** Advanced training techniques to handle high-dynamic-range forces (e.g.,  "walls") without numerical instability.



---

## 🚀 Performance Benchmarks

Emergentia achieves high-fidelity results across multiple physical regimes:

| Mode | Basis Functions | Target Law Example | Resilience |
| --- | --- | --- | --- |
| **Spring** |  |  | High |
| **LJ** |  |  | Medium |
| **Morse** |  |  | High |
| **Gravity** |  |  | High |

> 
> **Note:** Performance data is based on standard validation trials.
> 
> 

---

## 🛠 Installation

Emergentia requires **Python 3.9+**. Install the core dependencies via pip:

```bash
pip install torch numpy sympy gplearn pandas scipy pytest

```

---

## 💻 Usage

### 🧪 Running Benchmarks

To evaluate the engine across all supported potentials (Gravity, LJ, Morse, and Mixed):

```bash
python run_benchmarks.py

```

### 🔍 Running Tests

Verify the internal scaling, physics integrity, and registry consistency:

```bash
# Test trajectory scaling logic
pytest tests/test_scaling.py

# Verify Hamiltonian conservation and 3D discovery flow
pytest tests/test_physics_integrity.py

```

---

## 🏗 Project Structure

* 
`emergentia/`: Core package containing the discovery logic.


* 
`simulator.py`: Modular physics simulation using Velocity Verlet integration.


* 
`models.py`: `DiscoveryNet` architecture and `TrajectoryScaler`.


* 
`engine.py`: The `DiscoveryPipeline` linking neural training to symbolic regression.


* 
`registry.py`: Centralized physical basis functions (Torch, NumPy, SymPy).


* 
`utils.py`: Statistical verification and symbolic utility functions.




* 
`run_benchmarks.py`: Main entry point for cross-regime validation.



---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.