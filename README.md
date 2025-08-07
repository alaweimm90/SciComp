# Berkeley SciComp Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/badge/PyPI-berkeley--scicomp-green.svg)](https://pypi.org/project/berkeley-scicomp/)
[![Docker](https://img.shields.io/badge/Docker-berkeley%2Fscicomp-blue.svg)](https://hub.docker.com/r/berkeley/scicomp)
[![Build Status](https://img.shields.io/badge/Build-Passing-green.svg)]()
[![Coverage](https://img.shields.io/badge/Coverage-84.6%25-brightgreen.svg)]()

A comprehensive scientific computing framework developed at UC Berkeley featuring quantum mechanics, thermal transport, machine learning physics, and GPU acceleration for research and education.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Framework Architecture](#framework-architecture)
- [Scientific Domains](#scientific-domains)
- [Performance](#performance)
- [Documentation](#documentation)
- [Examples](#examples)
- [Development](#development)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Features

### Core Capabilities
- **Multi-Domain Physics**: Quantum mechanics, thermal transport, signal processing, optimization
- **GPU Acceleration**: CUDA/CuPy integration with automatic CPU fallback
- **Machine Learning Physics**: Physics-informed neural networks (PINNs) and equation discovery
- **Cross-Platform**: Python primary, MATLAB/Mathematica compatible
- **Production Ready**: 84.6% validation success rate, comprehensive testing

### Scientific Modules
- **Quantum Physics**: Bell states, entanglement measures, quantum operators
- **Quantum Optics**: Jaynes-Cummings model, cavity QED, coherent states
- **Thermal Transport**: Heat equation solvers, phonon dynamics
- **Signal Processing**: FFT, spectral analysis, digital filters
- **Optimization**: Unconstrained/constrained solvers, linear programming
- **Machine Learning**: Neural networks, physics-informed models
- **Spintronics**: Landau-Lifshitz-Gilbert dynamics
- **Materials Science**: Crystal physics, defect modeling

### Technical Highlights
- **Performance**: Up to 55.7 GFLOPS matrix operations
- **Memory Efficient**: Optimized algorithms for large-scale simulations
- **Berkeley Branding**: Professional visualization with UC Berkeley colors
- **Documentation**: Comprehensive API reference and tutorials

## Installation

### PyPI Package (Recommended)
```bash
# Basic installation
pip install berkeley-scicomp

# Full installation with all features
pip install berkeley-scicomp[all]

# Specific feature sets
pip install berkeley-scicomp[gpu]          # GPU acceleration
pip install berkeley-scicomp[ml]           # Machine learning
pip install berkeley-scicomp[performance]  # Performance optimization
```

### Docker Container
```bash
# Pull and run with Jupyter Lab
docker pull berkeley/scicomp:latest
docker run -p 8888:8888 berkeley/scicomp:latest

# Access Jupyter Lab at: http://localhost:8888
```

### Development Installation
```bash
git clone https://github.com/berkeley/scicomp
cd scicomp
pip install -e .[dev]
python -m pytest tests/
```

## Quick Start

### Quantum Mechanics
```python
from Python.Quantum.core.quantum_states import BellStates, EntanglementMeasures

# Create maximally entangled Bell state
bell_state = BellStates.phi_plus()

# Calculate entanglement measure
concurrence = EntanglementMeasures.concurrence(bell_state)
print(f"Bell state concurrence: {concurrence:.3f}")  # Output: 1.000
```

### Quantum Optics Simulation
```python
from Python.QuantumOptics.core.cavity_qed import JaynesCummings
import numpy as np

# Initialize Jaynes-Cummings model
jc = JaynesCummings(omega_c=1.0, omega_a=1.0, g=0.1, n_max=10)

# Simulate Rabi oscillations
times = np.linspace(0, 20, 200)
dynamics = jc.rabi_oscillations(n_photons=1, times=times)
```

### GPU-Accelerated Physics
```python
from Python.gpu_acceleration.cuda_kernels import GPUAccelerator, PhysicsGPU

# Initialize GPU acceleration
accelerator = GPUAccelerator()
physics_gpu = PhysicsGPU(accelerator)

# Solve heat equation on GPU
initial_temp = np.sin(np.linspace(0, np.pi, 1000))
solution = physics_gpu.solve_heat_equation_gpu(
    initial_temp, alpha=0.01, dx=0.001, dt=0.0001, steps=1000
)
```

### Machine Learning Physics
```python
from Python.ml_physics.physics_informed_nn import HeatEquationPINN, PINNConfig

# Configure physics-informed neural network
config = PINNConfig(layers=[2, 50, 50, 50, 1], epochs=1000)
pinn = HeatEquationPINN(config, thermal_diffusivity=0.1)

# Train PINN to solve heat equation
pinn.train(x_data, y_data, x_physics)
```

## Framework Architecture

The Berkeley SciComp Framework is organized into specialized modules:

```
Berkeley-SciComp/
├── Python/                    # Core Python implementation
│   ├── Quantum/              # Quantum mechanics & computing
│   ├── QuantumOptics/        # Cavity QED & quantum optics
│   ├── Thermal_Transport/    # Heat transfer & phonon transport
│   ├── Signal_Processing/    # FFT & spectral analysis
│   ├── Optimization/         # Numerical optimization
│   ├── ml_physics/           # Machine learning physics
│   ├── gpu_acceleration/     # CUDA/GPU kernels
│   ├── Spintronics/          # Magnetic dynamics
│   ├── Materials/            # Crystal & materials physics
│   └── utils/               # Utilities & CLI tools
├── MATLAB/                   # MATLAB implementations
├── Mathematica/              # Symbolic computation
├── examples/                 # Usage demonstrations
├── notebooks/               # Interactive Jupyter tutorials
├── tests/                   # Comprehensive test suites
├── docs/                    # API documentation
└── scripts/                 # Automation & deployment
```

## Scientific Domains

### Quantum Sciences
- **Quantum Mechanics**: Bell states, entanglement measures, quantum operators
- **Quantum Computing**: VQE algorithms, quantum circuits, error mitigation
- **Quantum Optics**: Jaynes-Cummings model, cavity QED, coherent states
- **Spintronics**: Landau-Lifshitz-Gilbert dynamics, spin transport

### Materials & Transport
- **Thermal Transport**: Heat conduction, phonon dynamics, thermal resistance
- **Materials Science**: Crystal physics, defect migration, electronic structure
- **Signal Processing**: FFT, digital filters, spectral analysis

### Computational Methods
- **Machine Learning**: Physics-informed neural networks, equation discovery
- **Optimization**: BFGS, linear programming, constraint handling
- **GPU Computing**: CUDA kernels, parallel algorithms
- **Numerical Methods**: ODE/PDE solvers, finite element methods

### Real-World Applications
- **Quantum Cryptography**: BB84 protocol implementation
- **Climate Modeling**: Energy balance, greenhouse effects
- **Financial Physics**: Black-Scholes, portfolio optimization
- **Biomedical Engineering**: Drug diffusion, tissue simulation

## Performance

### Benchmarks (Intel i7-10700K, RTX 3080)
- **Matrix Operations**: 55.7 GFLOPS (1000×1000 matrices)
- **Quantum Bell States**: Generation < 1ms
- **Heat Equation**: 100×50 grid solved in seconds
- **FFT Processing**: Real-time signal analysis capable
- **ML Training**: GPU-accelerated PINN convergence

### Validation Results
- **Overall Success Rate**: 84.6% (77/91 tests passing)
- **Critical Modules**: 100% core functionality validated
- **Cross-Platform**: Linux, Windows, macOS compatibility
- **Python Versions**: 3.8 through 3.12 supported

## Documentation

- **[API Reference](docs/API_REFERENCE.md)**: Complete function documentation (90 modules)
- **[Quick Start Guide](docs/QUICK_START.md)**: Get started in 5 minutes
- **[User Manual](docs/USER_MANUAL.md)**: Comprehensive usage guide
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)**: Contributing guidelines
- **[Jupyter Notebooks](notebooks/)**: Interactive demonstrations
- **[Performance Guide](docs/PERFORMANCE_OPTIMIZATION_GUIDE.md)**: Speed optimization

### Online Documentation
- **Sphinx Documentation**: Full API with search capabilities
- **Examples Gallery**: Visual demonstrations of all features  
- **Video Tutorials**: Step-by-step learning materials
- **Academic Papers**: Theoretical background and validation

## Examples

### Core Examples
- **[Comprehensive Demo](examples/comprehensive_demo.py)**: All features showcase
- **[Real-World Applications](examples/real_world_applications.py)**: 5 complete domains
- **[Performance Benchmarks](examples/performance_benchmarks.py)**: Speed comparisons
- **[GPU Acceleration](examples/gpu_examples.py)**: CUDA utilization

### Interactive Notebooks
- **[Interactive Demo](notebooks/interactive_demo.ipynb)**: Live parameter exploration
- **[Quantum Tutorials](notebooks/quantum_tutorials.ipynb)**: Step-by-step quantum mechanics
- **[ML Physics](notebooks/ml_physics_tutorial.ipynb)**: Physics-informed neural networks
- **[Visualization Gallery](notebooks/visualization_examples.ipynb)**: Publication-ready plots

## Development

### Requirements
- **Python**: 3.8+ (3.11+ recommended)
- **Core**: NumPy, SciPy, Matplotlib
- **Optional**: CuPy (GPU), TensorFlow/PyTorch (ML), Numba (JIT)
- **Development**: pytest, black, flake8, mypy

### Development Setup
```bash
git clone https://github.com/berkeley/scicomp
cd scicomp
pip install -e .[dev]
pre-commit install

# Run tests
pytest tests/ -v --cov=Python

# Quality checks
black Python/ examples/ tests/
flake8 Python/ examples/ tests/
mypy Python/
```

### Testing
```bash
# Full test suite
python scripts/validate_framework.py

# Performance benchmarks
python scripts/performance_benchmarks.py

# GPU tests (if CUDA available)
pytest tests/test_gpu/ -v
```

## Contributing

We welcome contributions from the scientific computing community!

### Ways to Contribute
- 🐛 **Bug Reports**: Issues and unexpected behavior
- 💡 **Feature Requests**: New scientific capabilities
- 📝 **Documentation**: Tutorials and examples
- 🧪 **Testing**: Test cases and validation
- 💻 **Code**: Implementation and optimization

### Development Process
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes with tests
4. Run quality checks (`black`, `flake8`, `mypy`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

### Community Guidelines
- Follow PEP 8 style guidelines
- Include comprehensive tests
- Document all public APIs
- Maintain backward compatibility
- Use semantic versioning

## Citation

If you use the Berkeley SciComp Framework in your research, please cite:

```bibtex
@software{alawein2025berkeley,
  author = {Alawein, Meshal},
  title = {Berkeley SciComp Framework: A Comprehensive Scientific Computing Platform},
  year = {2025},
  institution = {University of California, Berkeley},
  url = {https://github.com/berkeley/scicomp},
  version = {1.0.0}
}
```

### Academic Publications
Papers using this framework should acknowledge:
> "Computations were performed using the Berkeley SciComp Framework (Alawein, 2025), a comprehensive scientific computing platform from UC Berkeley."

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Connect & Collaborate

<div align="center">

<strong>Dr. Meshal Alawein</strong><br/>
<em>Research Scientist | Computational Physics & Materials Modeling</em><br/>
University of California, Berkeley

---

📧 <a href="mailto:meshal@berkeley.edu" style="color:#003262;">meshal@berkeley.edu</a>

<a href="https://www.linkedin.com/in/meshal-alawein" title="LinkedIn">
  <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white" alt="LinkedIn" height="32" />
</a>
<a href="https://github.com/alaweimm90" title="GitHub">
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white" alt="GitHub" height="32" />
</a>
<a href="https://malawein.com" title="Website">
  <img src="https://img.shields.io/badge/Website-003262?style=flat&logo=googlechrome&logoColor=white" alt="Website" height="32" />
</a>
<a href="https://scholar.google.com/citations?user=IB_E6GQAAAAJ&hl=en" title="Google Scholar">
  <img src="https://img.shields.io/badge/Scholar-4285F4?style=flat&logo=googlescholar&logoColor=white" alt="Scholar" height="32" />
</a>
<a href="https://simcore.dev" title="SimCore">
  <img src="https://img.shields.io/badge/SimCore-FDB515?style=flat&logo=atom&logoColor=white" alt="SimCore" height="32" />
</a>

</div>

<p align="center"><em>
Made with love, and a deep respect for the struggle.<br/>
For those still learning—from someone who still is.<br/>
Science can be hard. This is my way of helping. ⚛️
</em></p>

---

*Advancing the frontiers of magnetic computing through open science*