# Emergentia: Meso-scale Discovery Engine

Emergentia is a Neural-Symbolic discovery engine designed to extract parsimonious physical laws from meso-scale particle trajectories. By combining the flexible representation power of Neural Networks with the mathematical clarity of Symbolic Regression, Emergentia can "rediscover" the underlying equations of motion from raw simulation data, even in the presence of noise.

## 🚀 Key Features

*   **Modular Physics Engine:** A plug-and-play architecture for physical potentials. Easily define new laws (e.g., Harmonic, Lennard-Jones, Morse) by extending the base `Potential` class.
*   **Neural-Symbolic Pipeline:** 
    1.  **Neural Mapping:** A `DiscoveryNet` (PyTorch) approximates force laws using an expanded basis set $[r, 1/r, \exp(-r)]$.
    2.  **Symbolic Distillation:** A `SymbolicRegressor` (gplearn) extracts clean, human-readable mathematical formulas from the neural weights.
*   **Noise Resilience:** Built-in support for discovery from noisy trajectories using robust `HuberLoss` training and automated Gaussian smoothing considerations.
*   **Robust Statistical Validation:** Automated verification of discovered laws using curve-fitting, $R^2$ scores, and Bayesian Information Criterion (BIC) to ensure parsimony and accuracy.
*   **Hardware Acceleration:** Full support for `CUDA` (NVIDIA) and `MPS` (Apple Silicon) backends.
*   **Symmetric Log Transform:** Advanced training techniques to handle high-dynamic-range forces (e.g., $1/r^{13}$ walls) without numerical instability.

## 🛠 Installation

Ensure you have Python 3.9+ and the following dependencies:

```bash
pip install torch numpy sympy gplearn pandas scipy pytest
```

## 📈 Discovery Performance

Emergentia achieves high-fidelity results across multiple physical regimes and noise levels:

| Mode | Basis | Target Law | Discovery MSE | Noise Resilience |
| :--- | :--- | :--- | :--- | :--- |
| **Spring** | $r$ | $F = -k(r - r_0)$ | ~10⁻⁵ | High |
| **LJ** | $r^{-7}, r^{-13}$ | $F = A/r^{13} - B/r^{7}$ | ~10⁻³ | Medium |
| **Morse** | $\exp(-r)$ | $F = 2Da(e^{-a(r-r_e)} - e^{-2a(r-r_e)})$ | ~10⁻⁴ | High |

## 💻 Usage

### Run Benchmarks
To evaluate the engine across all supported potentials and noise levels:

```bash
python run_benchmarks.py
```

### Run Tests
To verify the internal scaling and normalization logic:

```bash
pytest tests/test_scaling.py
```

## 🏗 Project Structure

*   `emergentia/`: Core package.
    *   `simulator.py`: Modular physics simulation engine and potential definitions.
    *   `models.py`: `DiscoveryNet` architecture and `TrajectoryScaler`.
    *   `engine.py`: The `DiscoveryPipeline` linking neural training to symbolic regression.
    *   `utils.py`: Statistical verification and symbolic utility functions.
*   `run_benchmarks.py`: Main entry point for cross-regime validation.
*   `tests/`: Unit test suite.
*   `LICENSE`: Project licensing information.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.