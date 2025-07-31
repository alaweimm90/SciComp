#!/usr/bin/env python3
"""
SciComp: Professional Scientific Computing Portfolio

Setup script for installing the SciComp Python package.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_file(filename):
    """Read file contents and return as string."""
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

# Define package requirements
INSTALL_REQUIRES = [
    'numpy>=1.21.0',
    'scipy>=1.7.0',
    'matplotlib>=3.4.0',
    'pandas>=1.3.0',
    'h5py>=3.3.0',
    'tqdm>=4.61.0',
    'joblib>=1.0.0',
    'sympy>=1.8.0',
    'seaborn>=0.11.0',
    'plotly>=5.0.0',
    'bokeh>=2.3.0',
    'scikit-learn>=1.0.0',
    'tensorflow>=2.8.0',
    'torch>=1.11.0',
    'qiskit>=0.34.0',
    'pennylane>=0.22.0',
    'dwave-ocean-sdk>=4.0.0',
    'pymatgen>=2022.0.0',
    'ase>=3.22.0',
    'networkx>=2.6.0',
    'numba>=0.56.0',
    'opt-einsum>=3.3.0',
    'sparse>=0.13.0',
]

EXTRAS_REQUIRE = {
    'dev': [
        'pytest>=6.2.0',
        'pytest-cov>=2.12.0',
        'pytest-xdist>=2.3.0',
        'black>=21.6b0',
        'flake8>=3.9.0',
        'mypy>=0.910',
        'pre-commit>=2.13.0',
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=0.5.0',
        'nbsphinx>=0.8.0',
        'jupyter>=1.0.0',
        'ipykernel>=6.0.0',
        'memory-profiler>=0.58.0',
        'line-profiler>=3.3.0',
    ],
    'gpu': [
        'cupy>=9.0.0',
        'tensorflow-gpu>=2.8.0',
        'torch-gpu>=1.11.0',
    ],
    'quantum': [
        'qiskit[visualization]>=0.34.0',
        'pennylane[default.qubit]>=0.22.0',
        'cirq>=0.13.0',
        'forest-benchmarking>=0.8.0',
    ],
    'ml': [
        'pytorch-lightning>=1.6.0',
        'wandb>=0.12.0',
        'optuna>=2.10.0',
        'hyperopt>=0.2.5',
        'ray[tune]>=1.13.0',
    ],
}

# Add 'all' extra that includes everything
EXTRAS_REQUIRE['all'] = list(set(sum(EXTRAS_REQUIRE.values(), [])))

setup(
    name='scicomp',
    version='1.0.0',
    description='Professional Scientific Computing Portfolio',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    author='Meshal Alawein',
    author_email='meshal@berkeley.edu',
    url='https://github.com/alaweimm90/SciComp',
    license='MIT',
    packages=find_packages(where='Python'),
    package_dir={'': 'Python'},
    python_requires='>=3.9',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Education',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=[
        'scientific computing', 'quantum physics', 'quantum computing',
        'machine learning', 'computational physics', 'education',
        'research', 'berkeley', 'materials science', 'simulation'
    ],
    project_urls={
        'Documentation': 'https://scicomp.readthedocs.io/',
        'Source': 'https://github.com/alaweimm90/SciComp',
        'Tracker': 'https://github.com/alaweimm90/SciComp/issues',
        'Website': 'https://malawein.com',
        'LinkedIn': 'https://www.linkedin.com/in/meshal-alawein',
        'Research': 'https://simcore.dev',
    },
    include_package_data=True,
    package_data={
        'scicomp': [
            'config/*.json',
            'data/*.csv',
            'data/*.npy',
        ],
    },
    entry_points={
        'console_scripts': [
            'scicomp=scicomp.cli:main',
            'scicomp-benchmark=scicomp.benchmarks:main',
            'scicomp-validate=scicomp.validation:main',
        ],
    },
    zip_safe=False,
)