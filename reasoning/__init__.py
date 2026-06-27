"""Single-agent automated reasoning: model a problem as a state space, then search.

A problem is defined by an initial state, a goal test, a successor function and a
step cost. Search algorithms (BFS, uniform-cost, A*, IDA*) explore the resulting
state-action graph and return an optimal solution path plus the number of nodes
expanded, so different strategies can be compared on the same instance.
"""

from .problem import Problem, Node
from .search import bfs, uniform_cost, astar, ida_star, SearchResult
from .domains.eight_puzzle import EightPuzzle
from .domains.missionaries import MissionariesAndCannibals

__all__ = [
    "Problem",
    "Node",
    "bfs",
    "uniform_cost",
    "astar",
    "ida_star",
    "SearchResult",
    "EightPuzzle",
    "MissionariesAndCannibals",
]
