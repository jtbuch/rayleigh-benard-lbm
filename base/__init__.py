"""Base functionality for ShittyBird."""

from .simulation import run_simulation
from . import lbm

__all__ = ["run_simulation", "lbm"]