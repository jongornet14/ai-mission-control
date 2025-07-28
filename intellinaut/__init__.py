"""
Intellinaut: A Python package for RL training
"""

__version__ = "0.1.0"
__author__ = "Jonathan Gornet"
__email__ = "jonathan.gornet@gmail.com"

from . import algorithms, cli, workers, environments, logging, training, optimizers

__all__ = [
    "__version__",
    "algorithms",
    "cli",
    "workers",
    "environments",
    "logging",
    "training",
    "optimizers",
]
