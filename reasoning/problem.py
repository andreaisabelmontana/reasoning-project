"""The problem interface shared by every domain and every search algorithm.

A `Problem` is the classical formulation of single-agent reasoning:

  * an *initial state*,
  * a *goal test* that recognises a solved state,
  * a *successor function* yielding (action, next_state) pairs, and
  * a *step cost* for taking an action between two states.

A search algorithm only ever talks to a problem through these four methods, so the
same algorithm solves any domain that implements the interface. `Node` is a thin
wrapper used to build the search tree and to recover the solution path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Iterable, List, Optional, Tuple


# A state may be any hashable, immutable value (e.g. a tuple). Actions are arbitrary.
State = Hashable
Action = Any


class Problem:
    """Abstract base class for a single-agent state-space problem.

    Subclasses must provide ``initial`` and override ``goal_test`` and
    ``successors``. ``step_cost`` defaults to 1 (every action costs the same),
    which makes BFS, uniform-cost and A* coincide on optimal *length*.
    """

    def __init__(self, initial: State, goal: Optional[State] = None) -> None:
        self.initial = initial
        self.goal = goal

    def goal_test(self, state: State) -> bool:
        """Return True if ``state`` satisfies the goal."""
        if self.goal is None:
            raise NotImplementedError("goal_test must be overridden or a goal supplied")
        return state == self.goal

    def successors(self, state: State) -> Iterable[Tuple[Action, State]]:
        """Yield ``(action, next_state)`` pairs reachable from ``state``."""
        raise NotImplementedError

    def step_cost(self, state: State, action: Action, next_state: State) -> float:
        """Cost of applying ``action`` in ``state`` to reach ``next_state``."""
        return 1.0

    def heuristic(self, state: State) -> float:
        """Admissible estimate of the cost from ``state`` to a goal.

        The default heuristic (0) is admissible for every problem and reduces A*
        to uniform-cost search. Domains override this with something informative.
        """
        return 0.0


@dataclass(order=True)
class Node:
    """A node in the search tree.

    Ordering is by ``f`` then ``g`` so a node can be pushed onto a priority queue
    directly; ``state``, ``action`` and ``parent`` are excluded from comparison.
    """

    f: float
    g: float = field(compare=True)
    state: State = field(compare=False)
    parent: Optional["Node"] = field(default=None, compare=False)
    action: Action = field(default=None, compare=False)

    def path(self) -> List["Node"]:
        """Return the list of nodes from the root to this node, inclusive."""
        node: Optional[Node] = self
        out: List[Node] = []
        while node is not None:
            out.append(node)
            node = node.parent
        out.reverse()
        return out

    def actions(self) -> List[Action]:
        """Return the sequence of actions taken from the root to this node."""
        return [n.action for n in self.path() if n.action is not None]

    def states(self) -> List[State]:
        """Return the sequence of states from the root to this node."""
        return [n.state for n in self.path()]
