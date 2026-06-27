"""Search algorithms over the :class:`~reasoning.problem.Problem` interface.

Every algorithm returns a :class:`SearchResult` carrying the solution path, its
cost, and the number of nodes *expanded* (popped from the frontier and had their
successors generated). Nodes-expanded is the standard yardstick for comparing how
hard each strategy had to work on the same instance.

  * ``bfs``           — breadth-first; optimal when every step cost is equal.
  * ``uniform_cost``  — Dijkstra over the state space; optimal for any non-negative cost.
  * ``astar``         — best-first on f = g + h; optimal when h is admissible (graph
                        search additionally needs consistency, which all heuristics
                        here satisfy).
  * ``ida_star``      — iterative-deepening A*; same optimal cost as A* but linear memory.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .problem import Action, Node, Problem, State


@dataclass
class SearchResult:
    """Outcome of a search."""

    found: bool
    actions: List[Action]
    states: List[State]
    cost: float
    nodes_expanded: int

    @property
    def length(self) -> int:
        """Number of actions in the solution (0 if no/empty solution)."""
        return len(self.actions)

    def __bool__(self) -> bool:
        return self.found


def _failure(nodes_expanded: int) -> SearchResult:
    return SearchResult(False, [], [], float("inf"), nodes_expanded)


def _success(node: Node, nodes_expanded: int) -> SearchResult:
    return SearchResult(True, node.actions(), node.states(), node.g, nodes_expanded)


def bfs(problem: Problem) -> SearchResult:
    """Breadth-first graph search. Optimal in path length for uniform step costs."""
    start = Node(f=0.0, g=0.0, state=problem.initial)
    if problem.goal_test(start.state):
        return _success(start, 0)

    frontier: deque[Node] = deque([start])
    reached: Dict[State, None] = {start.state: None}
    expanded = 0

    while frontier:
        node = frontier.popleft()
        expanded += 1
        for action, succ in problem.successors(node.state):
            if succ in reached:
                continue
            cost = problem.step_cost(node.state, action, succ)
            child = Node(f=0.0, g=node.g + cost, state=succ, parent=node, action=action)
            if problem.goal_test(succ):
                return _success(child, expanded)
            reached[succ] = None
            frontier.append(child)
    return _failure(expanded)


def uniform_cost(problem: Problem) -> SearchResult:
    """Uniform-cost (Dijkstra) search. Optimal for any non-negative step cost."""
    return _best_first(problem, lambda g, h: g)


def astar(problem: Problem) -> SearchResult:
    """A* search on f = g + h. Optimal when the heuristic is admissible/consistent."""
    return _best_first(problem, lambda g, h: g + h)


def _best_first(problem: Problem, f_of: Callable[[float, float], float]) -> SearchResult:
    """Generic best-first graph search parameterised by the evaluation function.

    ``f_of(g, h)`` maps the path cost and heuristic to the priority. With
    ``f_of = g`` this is uniform-cost; with ``f_of = g + h`` it is A*.
    """
    start_h = problem.heuristic(problem.initial)
    start = Node(f=f_of(0.0, start_h), g=0.0, state=problem.initial)

    frontier: List[Node] = [start]
    # Best known g for each state; lets us skip stale/duplicate frontier entries.
    best_g: Dict[State, float] = {start.state: 0.0}
    expanded = 0

    while frontier:
        node = heapq.heappop(frontier)
        if node.g > best_g.get(node.state, float("inf")):
            continue  # a cheaper route to this state was already processed
        if problem.goal_test(node.state):
            return _success(node, expanded)
        expanded += 1
        for action, succ in problem.successors(node.state):
            g = node.g + problem.step_cost(node.state, action, succ)
            if g < best_g.get(succ, float("inf")):
                best_g[succ] = g
                h = problem.heuristic(succ)
                child = Node(f=f_of(g, h), g=g, state=succ, parent=node, action=action)
                heapq.heappush(frontier, child)
    return _failure(expanded)


def ida_star(problem: Problem, max_bound: Optional[float] = None) -> SearchResult:
    """Iterative-deepening A*. Optimal like A* but uses memory linear in depth.

    Each iteration runs a cost-bounded depth-first search; the bound starts at
    ``h(initial)`` and rises to the smallest f-value that exceeded the previous
    bound, until the goal is found.

    Plain IDA* keeps no closed set, so on an *unsolvable* instance it would deepen
    forever. ``max_bound`` caps the cost bound: once it is exceeded the search
    reports failure. A domain that knows an upper bound on any optimal solution
    (e.g. the 8-puzzle's diameter of 31) can supply it; otherwise the caller may
    pass one explicitly. With ``max_bound=None`` behaviour is the textbook one.
    """
    expanded = 0
    if max_bound is None:
        max_bound = getattr(problem, "max_solution_cost", None)
    cap = float("inf") if max_bound is None else float(max_bound)

    def dfs(node: Node, bound: float, on_path: set) -> tuple[Optional[Node], float]:
        nonlocal expanded
        f = node.g + problem.heuristic(node.state)
        if f > bound:
            return None, f
        if problem.goal_test(node.state):
            return node, f
        expanded += 1
        next_bound = float("inf")
        for action, succ in problem.successors(node.state):
            if succ in on_path:  # avoid cycles within the current DFS branch
                continue
            g = node.g + problem.step_cost(node.state, action, succ)
            child = Node(f=0.0, g=g, state=succ, parent=node, action=action)
            on_path.add(succ)
            found, t = dfs(child, bound, on_path)
            on_path.discard(succ)
            if found is not None:
                return found, t
            next_bound = min(next_bound, t)
        return None, next_bound

    root = Node(f=0.0, g=0.0, state=problem.initial)
    bound = problem.heuristic(root.state)
    while bound != float("inf") and bound <= cap:
        found, t = dfs(root, bound, {root.state})
        if found is not None:
            return _success(found, expanded)
        bound = t
    return _failure(expanded)


ALGORITHMS: Dict[str, Callable[[Problem], SearchResult]] = {
    "BFS": bfs,
    "UCS": uniform_cost,
    "A*": astar,
    "IDA*": ida_star,
}
