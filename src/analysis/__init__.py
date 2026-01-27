"""
Analysis Package for Cell-Free Massive MIMO Networks

This package contains modular analysis scripts for:
- Network evaluation and visualization
- Circuit power sensitivity analysis
- User scaling analysis

All scripts support both Statistical and Ray Tracing (RT) channel models.
"""

from . import visualization

__all__ = [
    'visualization',
]

__version__ = '1.0.0'
