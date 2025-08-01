#!/usr/bin/env python3
"""
Berkeley SciComp Framework Setup
================================

Professional setup configuration for the UC Berkeley Scientific Computing
Framework, implementing best practices for Python package distribution
and Berkeley academic standards.

Author: Dr. Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT

Copyright © 2025 Dr. Meshal Alawein — All rights reserved.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure minimum Python version
if sys.version_info < (3, 8):
    raise RuntimeError("Berkeley SciComp requires Python 3.8 or higher")

# Get the long description from README
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = here / filename
    if requirements_path.exists():
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [
                line.strip() 
                for line in f 
                if line.strip() and not line.startswith('#')
            ]
    return []

# Package metadata
PACKAGE_NAME = "berkeley-scicomp"
VERSION = "1.0.0"
DESCRIPTION = "UC Berkeley Scientific Computing Framework"
AUTHOR = "Dr. Meshal Alawein"
AUTHOR_EMAIL = "meshal@berkeley.edu"
URL = "https://github.com/berkeley-scicomp/SciComp"
LICENSE = "MIT"

# Berkeley SciComp Framework classification
CLASSIFIERS = [
    # Development Status
    "Development Status :: 5 - Production/Stable",
    
    # Intended Audience
    "Intended Audience :: Science/Research",
    "Intended Audience :: Education",
    "Intended Audience :: Developers",
    
    # Topic Classification
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Education",
    
    # License
    "License :: OSI Approved :: MIT License",
    
    # Programming Language
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    
    # Operating System
    "Operating System :: OS Independent",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: MacOS",
    
    # Natural Language
    "Natural Language :: English",
    
    # Environment
    "Environment :: Console",
    "Environment :: X11 Applications",
    "Environment :: MacOS X",
    "Environment :: Win32 (MS Windows)",
]

# Keywords for package discovery
KEYWORDS = [
    "scientific-computing", "quantum-physics", "machine-learning",
    "physics-informed-neural-networks", "quantum-computing", 
    "numerical-methods", "berkeley", "education", "research",
    "engineering", "mathematics", "visualization", "hpc"
]

# Package requirements
INSTALL_REQUIRES = read_requirements("requirements-berkeley.txt")

# Optional dependencies for different use cases
EXTRAS_REQUIRE = {
    "quantum": [
        "qiskit>=0.45.0",
        "cirq>=1.2.0",
        "pennylane>=0.32.0",
    ],
    "gpu": [
        "tensorflow-gpu>=2.13.0",
        "jax[cuda11_pip]>=0.4.0",
        "cupy-cuda11x>=12.0.0",
    ],
    "hpc": [
        "mpi4py>=3.1.4",
        "petsc4py>=3.19.0",
        "slepc4py>=3.19.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.5.0",
    ],
    "docs": [
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
        "jupyter-book>=0.15.0",
    ],
    "jupyter": [
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0",
        "ipywidgets>=8.0.0",
        "plotly>=5.15.0",
    ]
}

# Add 'all' option for installing everything
EXTRAS_REQUIRE["all"] = list(set(
    dep for deps in EXTRAS_REQUIRE.values() for dep in deps
))

# Entry points for command-line interfaces
ENTRY_POINTS = {
    "console_scripts": [
        "berkeley-scicomp=Python.utils.cli:main",
        "berkeley-test=tests.run_all_tests:main",
        "berkeley-demo=examples.run_demos:main",
    ],
}

# Package data and resources
PACKAGE_DATA = {
    "Python": [
        "*.json",
        "*.yaml", 
        "*.toml",
        "data/*.csv",
        "data/*.h5",
        "data/*.npy",
    ],
    "assets": [
        "*.py",
        "*.m", 
        "*.wl",
        "*.md",
        "*.mplstyle",
        "*.json",
    ],
    "docs": [
        "*.md",
        "*.rst",
        "*.html",
        "theory/*.md",
        "examples/*.md",
    ],
}

# Additional data files
DATA_FILES = [
    ("berkeley_scicomp/assets", [
        "assets/berkeley_style.py",
        "assets/berkeley_style.m",
        "assets/BerkeleyStyle.wl",
        "assets/STYLE_GUIDE.md",
    ]),
    ("berkeley_scicomp/docs", [
        "docs/README.md",
        "docs/theory/quantum_mechanics_theory.md",
        "docs/theory/ml_physics_theory.md",
        "docs/theory/computational_methods.md",
        "docs/theory/engineering_applications.md",
    ]),
]

# Project URLs
PROJECT_URLS = {
    "Homepage": URL,
    "Documentation": "https://berkeley-scicomp.readthedocs.io/",
    "Repository": URL,
    "Bug Tracker": f"{URL}/issues",
    "Changelog": f"{URL}/blob/main/CHANGELOG.md",
    "UC Berkeley": "https://www.berkeley.edu/",
    "Berkeley Physics": "https://physics.berkeley.edu/",
    "LBNL": "https://www.lbl.gov/",
}

def get_version():
    """Get version from __init__.py file."""
    version_file = here / "Python" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    return line.split('=')[1].strip().strip('"\'')
    return VERSION

# Custom commands for Berkeley-specific operations
class BerkeleyCommand:
    """Base class for Berkeley-specific setup commands."""
    
    @staticmethod
    def create_berkeley_config():
        """Create Berkeley configuration directory."""
        config_dir = Path.home() / ".berkeley_scicomp"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "config.json"
        if not config_file.exists():
            import json
            config = {
                "berkeley_colors": True,
                "watermark": True,
                "institution": "University of California, Berkeley",
                "author": "Dr. Meshal Alawein",
                "default_dpi": 300,
                "figure_format": "png"
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
    
    @staticmethod
    def setup_berkeley_environment():
        """Set up Berkeley computing environment."""
        # Create necessary directories
        data_dir = Path.home() / "berkeley_scicomp_data"
        data_dir.mkdir(exist_ok=True)
        
        # Create examples directory
        examples_dir = Path.home() / "berkeley_scicomp_examples"
        examples_dir.mkdir(exist_ok=True)
        
        print("Berkeley SciComp environment configured successfully!")
        print(f"Configuration: {Path.home() / '.berkeley_scicomp'}")
        print(f"Data directory: {data_dir}")
        print(f"Examples directory: {examples_dir}")

def run_berkeley_setup():
    """Run Berkeley-specific setup operations."""
    try:
        BerkeleyCommand.create_berkeley_config()
        BerkeleyCommand.setup_berkeley_environment()
    except Exception as e:
        print(f"Warning: Berkeley setup encountered an issue: {e}")

# Main setup configuration
setup(
    # Basic package information
    name=PACKAGE_NAME,
    version=get_version(),
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    url=URL,
    project_urls=PROJECT_URLS,
    license=LICENSE,
    
    # Package discovery and structure
    packages=find_packages(include=["Python*", "assets*"]),
    package_data=PACKAGE_DATA,
    data_files=DATA_FILES,
    include_package_data=True,
    
    # Dependencies
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    
    # Metadata
    classifiers=CLASSIFIERS,
    keywords=", ".join(KEYWORDS),
    
    # Entry points
    entry_points=ENTRY_POINTS,
    
    # Options
    zip_safe=False,  # Enable development installs
    platforms=["any"],
    
    # Additional configuration
    options={
        "bdist_wheel": {
            "universal": False,  # Python 3 only
        },
        "egg_info": {
            "tag_build": "",
            "tag_date": False,
        },
    },
)

# Run Berkeley-specific setup
if __name__ == "__main__":
    # Check if we're in install mode
    if "install" in sys.argv or "develop" in sys.argv:
        run_berkeley_setup()