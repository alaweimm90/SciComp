#!/usr/bin/env python3
"""
SciComp: Professional Scientific Computing Portfolio

A comprehensive, cross-platform scientific computing framework for quantum physics,
quantum computing, and physics-informed machine learning.

Author: Dr. Meshal Alawein (meshal@berkeley.edu)  
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Dr. Meshal Alawein — All rights reserved.
"""

__version__ = '1.0.0'
__author__ = 'Dr. Meshal Alawein'
__email__ = 'meshal@berkeley.edu'
__institution__ = 'University of California, Berkeley'
__license__ = 'MIT'
__website__ = 'https://malawein.com'
__github__ = 'https://github.com/alaweimm90'
__linkedin__ = 'https://www.linkedin.com/in/meshal-alawein'
__research__ = 'https://simcore.dev'

# Import main modules
from . import quantum_physics
from . import quantum_computing
from . import statistical_physics
from . import condensed_matter
from . import ml_physics
from . import computational_methods
from . import visualization
from . import utils

__all__ = [
    'quantum_physics',
    'quantum_computing', 
    'statistical_physics',
    'condensed_matter',
    'ml_physics',
    'computational_methods',
    'visualization',
    'utils'
]