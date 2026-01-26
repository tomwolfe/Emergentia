# Emergentia: Meso-scale Discovery Engine

Emergentia is a Neural-Symbolic discovery engine designed to extract parsimonious physical laws from meso-scale particle trajectories. By combining the flexible representation power of Neural Networks with the mathematical clarity of Symbolic Regression, Emergentia can "rediscover" the underlying equations of motion from raw simulation data.

## 🚀 Key Features

*   **Neural-Symbolic Pipeline:** Uses a two-stage process:
    1.  **Neural Mapping:** A `DiscoveryNet` (PyTorch) learns to approximate force laws with high precision.
    2.  **Symbolic Distillation:** A `SymbolicRegressor` (gplearn) extracts clean, human-readable mathematical formulas from the neural weights.
*   **Physics-Informed Features:** Incorporates distance-based basis functions (e.g., $r^{-7}$, $r^{-13}$) to guide the discovery of molecular potentials.
*   **Multi-Mode Simulation:** Built-in high-fidelity simulators for:
    *   **Spring Dynamics:** Linear restorative forces.
    *   **Lennard-Jones Potential:** Complex van der Waals interactions with high-dynamic-range repulsive walls.
*   **Hardware Acceleration:** Full support for `CUDA` (NVIDIA) and `MPS` (Apple Silicon) backends.
*   **Log-Transformed Training:** Advanced training techniques to handle force magnitudes across 6 orders of magnitude without numerical instability.

## 🛠 Installation

Ensure you have Python 3.9+ and the following dependencies:

```bash
pip install torch numpy sympy gplearn pandas matplotlib
```

## 📈 Discovery Performance

Emergentia currently achieves high-fidelity results across multiple regimes:

| Mode | Target Law | Discovery MSE | Coefficient Accuracy |
| :--- | :--- | :--- | :--- |
| **Spring** | $F = -k(r - r_0)$ | ~10⁻⁵ | >99.9% |
| **LJ (Repulsive)** | $F = 48(1/r^{13})$ | ~10⁻³ (log-space) | >95% |

## 💻 Usage

To run the discovery engine on all supported modes:

```bash
python discovery_core.py
```

The script will generate a `Final Summary Table` in the console and log detailed experiment data to `experiment_results.csv`.

## 🏗 Project Structure

*   `discovery_core.py`: The main engine containing the simulator, neural architecture, and symbolic pipeline.
*   `test_single_mode.py`: Utility for isolated testing of specific physics regimes.
*   `LICENSE`: Project licensing information.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
