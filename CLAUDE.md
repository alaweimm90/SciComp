# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

SciComp is a comprehensive, cross-platform scientific computing portfolio showcasing expertise in computational physics, quantum computing, and physics-informed machine learning. The repository provides equivalent implementations across Python, MATLAB, and Mathematica platforms.

**Author**: Meshal Alawein (meshal@berkeley.edu)  
**Institution**: University of California, Berkeley  
**License**: MIT © 2025 Meshal Alawein — All rights reserved

## Quick Start Commands

### Python Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate scicomp

# Install the package in development mode
pip install -e .

# Run tests
pytest tests/python/ -v

# Run type checking
mypy Python/

# Format code
black Python/
flake8 Python/
```

### MATLAB Setup
```matlab
% Add SciComp to MATLAB path
addpath(genpath('MATLAB/'));

% Run startup script
startup

% Run tests
runtests('tests/matlab/')
```

### Cross-Platform Validation
```bash
# Validate implementations across platforms
python scripts/validate_cross_platform.py

# Run performance benchmarks
python scripts/benchmark_performance.py
```

## Repository Architecture

### Core Python Package Structure
```
Python/
├── quantum_physics/        # Quantum mechanics & dynamics
│   ├── quantum_dynamics/    # TDSE solvers, wavepackets
│   ├── electronic_structure/ # DFT, band structure
│   ├── many_body/          # Exact diagonalization, QMC
│   └── quantum_optics/     # Cavity QED, light-matter
├── quantum_computing/      # Quantum algorithms & circuits
│   ├── algorithms/         # VQE, QAOA, Grover, Shor
│   ├── circuits/           # Gate optimization
│   ├── noise_models/       # Error models & mitigation
│   └── backends/           # Simulator interfaces
├── ml_physics/            # Physics-informed ML
│   ├── pinns/             # Physics-informed neural networks
│   ├── materials_ml/      # Property prediction
│   ├── quantum_ml/        # Quantum machine learning
│   └── scientific_computing_ml/ # ML acceleration
├── visualization/         # Berkeley-themed plotting
│   ├── berkeley_style/    # UC Berkeley color scheme
│   ├── interactive/       # Plotly, Bokeh dashboards
│   ├── quantum_viz/       # Bloch spheres, circuits
│   └── materials_viz/     # Crystal structures
└── utils/                # Common utilities
    ├── constants.py       # Physical constants (CODATA 2018)
    ├── units.py          # Unit conversions
    ├── file_io.py        # Data I/O (HDF5, NumPy, etc.)
    └── parallel.py       # Parallelization helpers
```

### Key Implementation Features

#### Berkeley Visual Identity
All visualizations follow UC Berkeley's official brand guidelines:
- **Primary Colors**: Berkeley Blue (#003262), California Gold (#FDB515)
- **Styling**: Publication-quality figures, inward-pointing ticks, no grids
- **Typography**: Professional fonts with proper LaTeX formatting

#### Cross-Platform Compatibility
- **Python**: 3.9+ with NumPy-style docstrings, type hints
- **MATLAB**: R2020b+ with camelCase functions, Live Scripts
- **Mathematica**: Version 12+ with usage messages, interactive notebooks

#### Scientific Computing Standards
- **Documentation**: Comprehensive docstrings with mathematical formulations
- **Testing**: pytest with 95% coverage, property-based testing
- **Performance**: Optimized algorithms with parallelization support
- **Accuracy**: Numerical precision validation across platforms

## Development Workflows

### Adding New Physics Modules
1. Create module in appropriate subdirectory (e.g., `Python/quantum_physics/`)
2. Include comprehensive docstrings with physics background
3. Add unit tests in `tests/python/`
4. Create equivalent MATLAB and Mathematica implementations
5. Add examples to `examples/` directory
6. Update documentation in `docs/`

### Running Simulations
```python
# Example: Quantum harmonic oscillator
from scicomp.quantum_physics import QuantumHarmonic
from scicomp.visualization import BerkeleyPlot

qho = QuantumHarmonic(omega=1.0)
psi_0 = qho.eigenstate(n=0)

plot = BerkeleyPlot()
plot.wavefunction(qho.x, psi_0, title="Ground State")
plot.save_figure("ground_state.png")
```

### VQE Quantum Computing Example
```python
from scicomp.quantum_computing import VQE, HardwareEfficientAnsatz
import numpy as np

# Define Hamiltonian (Pauli-Z on single qubit)
hamiltonian = np.array([[1, 0], [0, -1]])

# Create ansatz and run VQE
ansatz = HardwareEfficientAnsatz(num_qubits=1, num_layers=2)
vqe = VQE(hamiltonian, ansatz)
result = vqe.optimize()

print(f"Ground state energy: {result['optimal_energy']:.6f}")
```

## Git LFS Configuration

The repository uses Git LFS for large scientific data files:
- `.hdf5`, `.h5` - HDF5 data files
- `.nc` - NetCDF files 
- `.mat` - MATLAB data files
- `.npy` - NumPy array files
- Files in `data/examples/` and `results/` directories

## Common Tasks

### Testing
- **Python**: `pytest tests/python/ --cov=scicomp`
- **MATLAB**: `runtests('tests/matlab/')`
- **Cross-platform**: `python scripts/validate_cross_platform.py`

### Code Quality
- **Type checking**: `mypy Python/`
- **Formatting**: `black Python/` and `flake8 Python/`
- **Documentation**: `sphinx-build docs/ docs/_build/`

### Performance Benchmarking
- **Single platform**: `python benchmarks/benchmark_module.py`
- **Cross-platform**: `python scripts/benchmark_cross_platform.py`

### Configuration Files
- **Berkeley Theme**: `config/berkeley_theme.json` - UC Berkeley colors and styling
- **Physics Constants**: `config/physics_constants.json` - CODATA 2018 values
- **Platform Settings**: `config/platform_settings.json` - Platform-specific configs

## Permissions

The repository has Claude Code permissions configured in `.claude/settings.local.json` allowing:
- `Bash(find:*)` - File searching operations
- `Bash(ls:*)` - Directory listing operations

## Educational Examples

### Beginner Level
- Particle in a box (`examples/beginner/particle_in_box.py`)
- Harmonic oscillator (`examples/beginner/harmonic_oscillator.py`)
- Quantum tunneling (`examples/beginner/tunneling.py`)

### Intermediate Level
- Band structure calculations (`examples/intermediate/band_structure.py`)
- VQE optimization (`examples/intermediate/vqe_h2.py`)
- PINN for heat equation (`examples/intermediate/heat_pinn.py`)

### Advanced Level
- Many-body localization (`examples/advanced/mbl_study.py`)
- Quantum error correction (`examples/advanced/qec_codes.py`)
- Materials inverse design (`examples/advanced/inverse_design.py`)

## Research Applications

This codebase supports research in:
- **Quantum Many-Body Physics**: Exact diagonalization, quantum Monte Carlo
- **Electronic Structure Theory**: DFT, tight-binding models, strain engineering
- **Quantum Computing**: Variational algorithms, error mitigation, NISQ devices
- **Materials Science**: Property prediction, inverse design, high-throughput screening
- **Physics-Informed ML**: PINNs for quantum systems, uncertainty quantification

For questions or contributions, contact Meshal Alawein at meshal@berkeley.edu.