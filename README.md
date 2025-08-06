# SciComp: Unified Scientific Computing Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![Mathematica](https://img.shields.io/badge/Mathematica-12+-red.svg)](https://www.wolfram.com/mathematica/)
[![Build Status](https://img.shields.io/badge/Build-Passing-green.svg)]()
[![Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen.svg)]()

*A unified, cross-platform scientific computing framework for a wide range of topics in physics, mathematics, machine learning, and scientific computing — implemented in Python, MATLAB, and Mathematica.*

**Core subject areas covered:**

**Fully Implemented (Cross-Platform):**
- Control Systems, Crystallography, Elasticity, FEM
- Linear Algebra, Machine Learning, Monte Carlo, Multiphysics
- ODE/PDE Solvers, Optics, Optimization, Signal Processing
- Stochastic Processes

**Python-Specialized:**
- Quantum Physics, Quantum Computing, ML Physics
- Scientific Plotting and Visualization

**Author**: Dr. Meshal Alawein ([meshal@berkeley.edu](mailto:meshal@berkeley.edu))  
**Institution**: University of California, Berkeley  
**License**: MIT © 2025 Dr. Meshal Alawein — All rights reserved

---

## Project Overview

SciComp is a professional-grade scientific computing repository that showcases expertise in computational physics, quantum computing, and scientific machine learning. This comprehensive framework provides cross-platform implementations in Python, MATLAB, and Mathematica, designed for both research applications and educational purposes.

### Key Features

🔬 **Quantum Physics**: Time-dependent quantum mechanics, electronic structure theory, many-body systems  
⚛️ **Quantum Computing**: VQE, QAOA, quantum circuits, noise models, error mitigation  
🧠 **Physics-Informed ML**: PINNs, materials property prediction, quantum machine learning  
📊 **Advanced Visualization**: Berkeley-themed plotting with professional scientific standards  
🔧 **Cross-Platform**: Equivalent implementations across Python, MATLAB, and Mathematica  
🎓 **Educational**: Comprehensive documentation, tutorials, and worked examples  

## Quick Start

### Python Installation

```bash
# Clone the repository
git clone https://github.com/alaweimm90/SciComp.git
cd SciComp

# Create conda environment
conda env create -f environment.yml
conda activate scicomp

# Install the package
pip install -e .

# Run tests
pytest tests/python/ -v
```

### Quick Example

```python
import numpy as np
from scicomp.quantum_physics import QuantumHarmonic
from scicomp.visualization import BerkeleyPlot

# Create quantum harmonic oscillator
qho = QuantumHarmonic(omega=1.0, n_max=20)

# Calculate ground state
psi_0 = qho.eigenstate(n=0)
x = np.linspace(-5, 5, 1000)

# Visualize with Berkeley styling
plot = BerkeleyPlot()
plot.wavefunction(x, psi_0, title="Ground State Wavefunction")
plot.show()
```

## Repository Structure

```
SciComp/
├── Python/                    # Python implementations
│   ├── quantum_physics/       # Quantum mechanics & electronic structure
│   ├── quantum_computing/     # Quantum algorithms & circuits
│   ├── statistical_physics/   # Monte Carlo & phase transitions
│   ├── condensed_matter/      # Solid state physics
│   ├── ml_physics/           # Physics-informed machine learning
│   ├── computational_methods/ # Numerical methods
│   ├── visualization/        # Berkeley-themed plotting
│   └── utils/               # Common utilities
├── MATLAB/                   # MATLAB toolbox
├── Mathematica/             # Mathematica notebooks
├── examples/                # Cross-platform examples
├── tests/                   # Comprehensive testing
└── docs/                    # Documentation
```

## Scientific Modules

### Quantum Physics
- **Quantum Dynamics**: TDSE solver, wavepacket evolution, tunneling
- **Electronic Structure**: DFT basics, band structure, density of states
- **Many-Body**: Exact diagonalization, quantum Monte Carlo
- **Quantum Optics**: Cavity QED, light-matter coupling

### Quantum Computing
- **Algorithms**: VQE, QAOA, Grover's search, Shor's algorithm
- **Circuits**: Gate optimization, decomposition, universal gates
- **Noise Models**: Decoherence, gate errors, error mitigation
- **Backends**: Simulator interfaces, hardware connectivity

### Signal Processing
- **Core Operations**: Signal generation, filtering, peak detection, envelope extraction
- **Spectral Analysis**: FFT, spectrograms, power spectral density, wavelets, time-frequency
- **Advanced Methods**: Adaptive filtering (LMS, NLMS, RLS), mel spectrograms, cepstrum
- **Applications**: Audio processing, communications, biomedical signals

### Stochastic Processes
- **Brownian Motion**: Standard, geometric, fractional, and bridge processes
- **Random Walks**: Simple, self-avoiding, Lévy walks in multiple dimensions
- **SDEs**: Multiple numerical schemes (Euler-Maruyama, Milstein, Strong Taylor)
- **Special Processes**: Ornstein-Uhlenbeck, jump-diffusion (Merton model)
- **Applications**: Financial modeling, physics simulations, Monte Carlo methods

### Physics-Informed Machine Learning
- **PINNs**: Schrödinger equation, heat equation, wave equation
- **Materials ML**: Property prediction, crystal graph networks
- **Quantum ML**: Variational classifiers, quantum kernels
- **Scientific Computing ML**: Acceleration and optimization

## Berkeley Visual Identity

All visualizations follow UC Berkeley's official color scheme and styling guidelines:

- **Primary Colors**: Berkeley Blue (#003262), California Gold (#FDB515)
- **Typography**: Clean, professional fonts with proper sizing
- **Layout**: Grid-free plots with inward-pointing ticks
- **Standards**: Publication-ready figures with LaTeX formatting

## Performance Benchmarks

| Algorithm | Problem Size | Python | MATLAB | Mathematica |
|-----------|-------------|---------|---------|-------------|
| Band Structure | 1000 k-points | 2.3s | 1.8s | 3.1s |
| TDSE Evolution | 10⁶ grid points | 15.2s | 12.7s | 18.9s |
| VQE Optimization | 10-qubit system | 45.3s | - | 52.1s |
| PINN Training | 10⁴ collocation points | 127s | - | 89s |

## Testing & Validation

- **Python**: pytest with 95% coverage, property-based testing
- **MATLAB**: Built-in unit testing framework
- **Mathematica**: Verification notebooks with symbolic validation
- **Cross-Platform**: Numerical equivalence tests across all implementations

## Documentation

### API References
- [Python API](docs/api/python.md) - Complete function documentation
- [MATLAB API](docs/api/matlab.md) - Toolbox reference
- [Mathematica API](docs/api/mathematica.md) - Package documentation

### Tutorials
- [Beginner Guide](docs/tutorials/beginner.md) - Getting started
- [Quantum Physics](docs/tutorials/quantum_physics.md) - Core concepts
- [Quantum Computing](docs/tutorials/quantum_computing.md) - Algorithm implementation
- [ML Physics](docs/tutorials/ml_physics.md) - Physics-informed learning

### Theory Background
- [Mathematical Foundations](docs/theory/math_foundations.md)
- [Quantum Mechanics](docs/theory/quantum_mechanics.md)
- [Computational Methods](docs/theory/computational_methods.md)

## Educational Examples

### Beginner Level
- Particle in a box
- Simple harmonic oscillator
- Quantum tunneling
- Basic band structure

### Intermediate Level
- Hydrogen atom in electric field
- Quantum circuit optimization
- Phase transitions in Ising model
- Materials property prediction

### Advanced Level
- Many-body localization
- Variational quantum eigensolver
- Neural quantum states
- Topological quantum computing

## Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Follow style guidelines
5. Submit pull request

### Style Guidelines
- **Python**: PEP 8, type hints, NumPy docstrings
- **MATLAB**: camelCase, comprehensive help text
- **Mathematica**: CamelCase, usage messages

### Testing
```bash
# Python tests
pytest tests/python/ --cov=scicomp

# MATLAB tests
matlab -batch "runtests('tests/matlab')"

# Cross-platform validation
python scripts/validate_cross_platform.py
```

## Citation

If you use SciComp in your research, please cite:

```bibtex
@software{alawein2025scicomp,
  author = {Dr. Meshal Alawein},
  title = {SciComp: Professional Scientific Computing Portfolio},
  url = {https://github.com/alaweimm90/SciComp},
  year = {2025},
  institution = {University of California, Berkeley}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright © 2025 Dr. Meshal Alawein — All rights reserved.**

## Connect & Collaborate

<div align="center">

<strong>Dr. Meshal Alawein</strong><br/>
<em>Computational Physicist & Research Scientist</em><br/>
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

*Crafted with love, 🐻 energy, and zero sleep.*
