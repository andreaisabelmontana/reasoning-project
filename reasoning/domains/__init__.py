"""Concrete problem domains implementing the :class:`~reasoning.problem.Problem` interface."""

from .eight_puzzle import EightPuzzle
from .missionaries import MissionariesAndCannibals

__all__ = ["EightPuzzle", "MissionariesAndCannibals"]
