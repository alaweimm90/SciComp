# Contributing to SciComp
![Berkeley SciComp](https://img.shields.io/badge/SciComp-003262?style=flat-square&logo=university)
[![Contributors Welcome](https://img.shields.io/badge/contributors-welcome-brightgreen?style=flat-square)](https://github.com/berkeley/scicomp/contribute)
**Welcome to the SciComp community!**
We're excited that you're interested in contributing to our scientific computing suite. This guide will help you get started with contributing code, documentation, examples, or bug reports.
**Author**: Dr. Meshal Alawein (meshal@berkeley.edu)
**Institution**: University of California, Berkeley
**License**: MIT
---
## 🎯 How to Contribute
### Ways to Contribute
1. **🐛 Bug Reports**: Found an issue? Let us know!
2. **💡 Feature Requests**: Ideas for new capabilities
3. **📝 Documentation**: Improve our docs and examples
4. **🧪 Testing**: Add test cases and improve coverage
5. **💻 Code**: Implement new features or fix bugs
6. **🎓 Examples**: Create tutorials and use cases
7. **📊 Benchmarks**: Performance analysis and optimization
### Quick Start
1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes with Berkeley styling
5. **Test** thoroughly
6. **Submit** a pull request
---
## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Berkeley Standards](#berkeley-standards)
- [Contribution Types](#contribution-types)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Community](#community)
---
## 🤝 Code of Conduct
This project follows the [Berkeley SciComp Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold this code. Please report unacceptable behavior to [meshal@berkeley.edu](mailto:meshal@berkeley.edu).
---
## 🚀 Getting Started
### Prerequisites
- **Python**: 3.8+ with scientific computing libraries
- **Git**: For version control
- **Optional**: MATLAB R2020a+, Mathematica 12.0+
### Fork and Clone
```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/SciComp.git
cd SciComp
# Add upstream remote
git remote add upstream https://github.com/berkeley-scicomp/SciComp.git
```
---
## 🛠️ Development Environment
### Quick Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
# Install pre-commit hooks
pre-commit install
```
### Development Tools
- **Testing**: `pytest` with coverage
- **Formatting**: `black` and `isort`
- **Linting**: `flake8` with plugins
- **Type Checking**: `mypy`
- **Documentation**: `sphinx` with Berkeley theme
### Makefile Commands
```bash
make help          # Show all available commands
make test          # Run full test suite
make lint          # Run code quality checks
make format        # Auto-format code
make docs          # Build documentation
make clean         # Clean build artifacts
```
---
## 🏛️ Berkeley Standards
### Visual Identity
- **Colors**: Use official Berkeley Blue (#003262) and California Gold (#FDB515)
- **Styling**: Apply Berkeley visual identity to all visualizations
- **Branding**: Include proper UC Berkeley attribution
### Academic Standards
- **Documentation**: Every function must have comprehensive docstrings
- **Testing**: Minimum 80% code coverage required
- **Validation**: Compare results with analytical solutions where possible
- **Citations**: Include relevant academic references
### Code Quality
- **Style**: Follow PEP 8 with Berkeley-specific additions
- **Type Hints**: Use type annotations for all public APIs
- **Error Handling**: Provide informative error messages
- **Performance**: Optimize for both educational and research use
---
## 📝 Contribution Types
### 🐛 Bug Reports
When reporting bugs, please include:
- **Environment**: OS, Python version, library versions
- **Steps to Reproduce**: Minimal code example
- **Expected vs Actual**: What should happen vs what happens
- **Berkeley Context**: If related to Berkeley-specific features
**Template**:
```markdown
**Bug Description**
Clear description of the bug
**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11.0]
- Berkeley SciComp: [e.g., 1.0.0]
**Steps to Reproduce**
1. Step one
2. Step two
3. See error
**Expected Behavior**
What should happen
**Actual Behavior**
What actually happens
**Additional Context**
Any other relevant information
```
### ✨ Feature Requests
For new features, consider:
- **Scientific Value**: Educational or research benefit
- **Berkeley Relevance**: Alignment with UC Berkeley mission
- **Scope**: Fits within framework architecture
- **Maintenance**: Long-term sustainability
### 🔬 Scientific Contributions
We especially welcome:
- **New Algorithms**: Quantum physics, ML physics methods
- **Validation Cases**: Test problems with known solutions
- **Educational Content**: Tutorials and examples
- **Berkeley Applications**: UC Berkeley specific use cases
---
## 🔄 Pull Request Process
### Before You Start
1. **Check Issues**: Look for existing related issues
2. **Discuss**: For major changes, create an issue first
3. **Branch**: Create a descriptive feature branch
### Development Workflow
```bash
# Update your fork
git checkout main
git pull upstream main
git push origin main
# Create feature branch
git checkout -b feature/quantum-tunneling-enhancement
# Make changes, commit frequently
git add -A
git commit -m "Add Berkeley styling to quantum tunneling plots"
# Keep up to date
git fetch upstream
git rebase upstream/main
# Push to your fork
git push origin feature/quantum-tunneling-enhancement
```
### Pull Request Template
```markdown
## 📋 Description
Brief description of changes and motivation.
## 🔬 Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that causes existing functionality to change)
- [ ] Documentation update
- [ ] Berkeley styling/branding update
## ✅ Testing
- [ ] Added tests for new functionality
- [ ] All tests pass locally
- [ ] Code coverage maintained/improved
- [ ] Manual testing performed
## 📚 Documentation
- [ ] Updated docstrings
- [ ] Updated user documentation
- [ ] Added examples if applicable
- [ ] Berkeley attribution included
## 🎨 Berkeley Standards
- [ ] Applied Berkeley visual identity
- [ ] Used official Berkeley colors
- [ ] Included proper UC Berkeley attribution
- [ ] Follows academic documentation standards
## 📋 Checklist
- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code, particularly hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] Any dependent changes have been merged and published
```
### Review Process
1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers review code quality and design
3. **Berkeley Review**: Verification of Berkeley standards compliance
4. **Testing**: Comprehensive testing across platforms
5. **Merge**: Integration into main branch
---
## 🧪 Testing Guidelines
### Test Structure
```
tests/
├── python/
│   ├── test_quantum_physics.py
│   ├── test_ml_physics.py
│   └── test_quantum_computing.py
├── matlab/
│   └── test_heat_transfer.m
├── mathematica/
│   └── test_symbolic_quantum.nb
└── integration/
    └── test_cross_platform.py
```
### Writing Tests
```python
import pytest
import numpy as np
from Python.quantum_physics import QuantumHarmonic
class TestQuantumHarmonic:
    """Test quantum harmonic oscillator implementation."""
    def test_energy_levels(self):
        """Test energy eigenvalue calculation."""
        qho = QuantumHarmonic()
        energies = qho.energy_levels(n_max=5)
        # Compare with analytical solution
        expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        np.testing.assert_allclose(energies, expected, rtol=1e-10)
    def test_berkeley_styling(self):
        """Test Berkeley visual identity application."""
        qho = QuantumHarmonic()
        fig = qho.plot_wavefunctions(apply_berkeley_style=True)
        # Check Berkeley colors are used
        assert fig.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # White background
        # Additional Berkeley style checks...
```
### Test Categories
- **Unit Tests**: Individual function testing
- **Integration Tests**: Cross-module functionality
- **Performance Tests**: Benchmarking and profiling
- **Visual Tests**: Plot and figure validation
- **Cross-Platform Tests**: Python/MATLAB/Mathematica consistency
---
## 📖 Documentation Standards
### Docstring Format
```python
def quantum_tunneling_probability(
    energy: float,
    barrier_height: float,
    barrier_width: float,
    mass: float = 1.0
) -> float:
    """
    Calculate quantum tunneling probability through rectangular barrier.
    This function implements the exact solution for quantum tunneling through
    a one-dimensional rectangular potential barrier, a fundamental problem in
    quantum mechanics with applications in scanning tunneling microscopy and
    quantum devices.
    Parameters
    ----------
    energy : float
        Particle energy in eV
    barrier_height : float
        Potential barrier height in eV
    barrier_width : float
        Barrier width in meters
    mass : float, optional
        Particle mass in atomic mass units, by default 1.0
    Returns
    -------
    float
        Transmission probability (0 ≤ T ≤ 1)
    Raises
    ------
    ValueError
        If energy or barrier parameters are negative
    Notes
    -----
    The transmission probability is calculated using the exact quantum
    mechanical formula:
    .. math::
        T = \\frac{1}{1 + \\frac{V_0^2 \\sinh^2(\\kappa a)}{4E(V_0-E)}}
    where κ = √(2m(V₀-E))/ℏ for E < V₀.
    References
    ----------
    .. [1] Griffiths, D.J. "Introduction to Quantum Mechanics", 3rd ed.
    .. [2] Shankar, R. "Principles of Quantum Mechanics", 2nd ed.
    Examples
    --------
    >>> # Electron tunneling through 1 eV barrier
    >>> prob = quantum_tunneling_probability(0.5, 1.0, 1e-9)
    >>> print(f"Tunneling probability: {prob:.3e}")
    Tunneling probability: 2.345e-04
    See Also
    --------
    wavefunction_barrier : Calculate wavefunction in barrier region
    reflection_coefficient : Calculate reflection probability
    """
```
### Berkeley Documentation Style
- **Institution**: Always include UC Berkeley attribution
- **Academic Rigor**: Reference relevant literature
- **Educational Focus**: Explain physical significance
- **Visual Identity**: Use Berkeley colors in documentation plots
---
## 🏫 Community
### Communication Channels
- **Issues**: GitHub issues for bugs and features
- **Discussions**: GitHub discussions for questions
- **Email**: [meshal@berkeley.edu](mailto:meshal@berkeley.edu) for direct contact
### Berkeley Community
- **Students**: UC Berkeley students and researchers
- **Faculty**: Berkeley physics and engineering faculty
- **LBNL**: Lawrence Berkeley National Laboratory collaborators
- **Alumni**: Berkeley alumni in academia and industry
### Getting Help
1. **Documentation**: Check docs for existing solutions
2. **Examples**: Review example scripts and notebooks
3. **Tests**: Look at test cases for usage patterns
4. **Issues**: Search existing issues for similar problems
5. **Discussion**: Start a GitHub discussion
6. **Contact**: Reach out directly for complex issues
---
## 🏆 Recognition
### Contributors
All contributors are recognized in:
- **CHANGELOG.md**: Release notes with contributor acknowledgments
- **Documentation**: Contributors page with Berkeley affiliation
- **Repository**: GitHub contributors list
### Berkeley Acknowledgment
Significant contributions may be acknowledged in:
- **Academic Papers**: Co-authorship opportunities
- **Conference Presentations**: Berkeley conference presentations
- **Berkeley Network**: Introduction to Berkeley research community
---
## 🔒 Legal and Licensing
### License Agreement
By contributing, you agree that your contributions will be licensed under the MIT License.
### Berkeley Trademark
- **Respectful Use**: Proper use of UC Berkeley trademarks
- **Academic Context**: Contributions should maintain academic integrity
- **Attribution**: Proper attribution to UC Berkeley and contributors
### Intellectual Property
- **Original Work**: Ensure contributions are your original work
- **Citations**: Properly cite any referenced algorithms or methods
- **Open Source**: All contributions become part of open source framework
---
## 🎓 Educational Mission
The SciComp serves UC Berkeley's educational mission:
- **Student Learning**: Support computational physics education
- **Research Excellence**: Enable cutting-edge scientific research
- **Community Impact**: Advance open science and education
- **Berkeley Pride**: Represent UC Berkeley excellence
---
## 📞 Contact
**Primary Developer**: Dr. Meshal Alawein
**Email**: [meshal@berkeley.edu](mailto:meshal@berkeley.edu)
**Institution**: University of California, Berkeley
**GitHub**: [@berkeley-scicomp](https://github.com/berkeley-scicomp)
---
## 🐻 Go Bears!
Thank you for contributing to the SciComp! Your contributions help advance scientific computing education and research at UC Berkeley and beyond.
**💙💛 University of California, Berkeley 💛💙**
---
*Copyright © 2025 Dr. Meshal Alawein — All rights reserved.*
*University of California, Berkeley*