# SciComp

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-blue.svg)](https://developer.nvidia.com/cuda-zone)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-red.svg)](https://www.tensorflow.org/)
[![Berkeley](https://img.shields.io/badge/UC%20Berkeley-2025-brightgreen.svg)](https://www.berkeley.edu/)
[![Tests](https://img.shields.io/badge/Tests-84.6%25-brightgreen.svg)](#testing--validation)

Cross-platform scientific computing for quantum mechanics, thermal transport, and ML physics. Works on Python, MATLAB, and Mathematica.
## Features

- Quantum mechanics: Bell states, entanglement, operators, circuits
- Quantum optics: Jaynes-Cummings, cavity QED, coherent states  
- GPU acceleration: CUDA/CuPy with CPU fallback
- ML physics: PINNs for PDEs and equation discovery
- Thermal transport: Heat solvers, phonon dynamics
- Signal processing: FFT, spectral methods
- Cross-platform: Python-MATLAB-Mathematica APIs

## Install

```bash
git clone https://github.com/berkeley/scicomp.git
cd scicomp
pip install -e .
```

Requirements: Python 3.8+, NumPy, SciPy, Matplotlib. Optional: CUDA 11.0+, TensorFlow 2.0+.

## Usage

```bash
# Quantum simulations
python examples/beginner/getting_started.py

# GPU computations  
python examples/gpu_examples.py

# ML physics
python examples/ml_physics_demo.py
```
## Modules

**Quantum**: Bell states, VQE, QAOA, Jaynes-Cummings
**GPU**: CUDA kernels, automatic CPU fallback
**ML Physics**: PINNs for heat/wave/Schrödinger equations  
**Platforms**: Python core, MATLAB/Mathematica support
## Testing

```bash
# Validation suite
python scripts/validate_framework.py

# Individual modules
python tests/python/test_quantum_physics.py

# Cross-platform
matlab -batch "run('tests/matlab/test_heat_transfer.m')"
wolframscript -f tests/mathematica/test_symbolic_quantum.nb
```
## Structure

```
SciComp/
├── Python/          # Core implementation
│   ├── Quantum/     # Quantum mechanics/computing
│   ├── ml_physics/  # Physics-informed ML
│   └── utils/       # CLI tools
├── MATLAB/          # MATLAB versions
├── Mathematica/     # Symbolic computation
├── examples/        # Demo scripts
└── tests/          # Test suites
```

84.6% test validation across 91 tests. GPU: up to 55.7 GFLOPS on RTX 3080.
## Citation

```bibtex
@software{alawein2025scicomp,
  title={SciComp},
  author={Alawein, Dr. Meshal},
  year={2025},
  url={https://github.com/berkeley/scicomp},
  institution={University of California, Berkeley}
}
```
MIT License - see [LICENSE](LICENSE).

---

**Meshal Alawein**  
UC Berkeley | [meshal@berkeley.edu](mailto:meshal@berkeley.edu)
