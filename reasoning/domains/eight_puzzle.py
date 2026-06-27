"""The 8-puzzle: a 3x3 sliding-tile puzzle (generalised to any N x N).

A state is a flat tuple of length ``side*side`` holding the tiles ``1..n`` and a
single blank ``0``. The goal is tiles in order with the blank last, e.g. for the
8-puzzle ``(1, 2, 3, 4, 5, 6, 7, 8, 0)``. The only action is sliding a tile into
the blank, encoded by the direction the *blank* moves: 'U', 'D', 'L', 'R'.

Two admissible heuristics are provided:

  * **misplaced tiles** — count of tiles not in their goal cell; admissible because
    every misplaced tile needs at least one move.
  * **Manhattan distance** — sum over tiles of the grid distance to their goal cell;
    admissible because a tile can close at most one unit of its distance per move,
    and it dominates misplaced-tiles, so it expands fewer nodes in A*.
"""

from __future__ import annotations

import random
from typing import Iterable, List, Optional, Sequence, Tuple

from ..problem import Problem

# Moves are named by where the blank goes; the value is the change in (row, col).
_MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
# Applying a move and then its inverse returns to the same state — used when scrambling.
_INVERSE = {"U": "D", "D": "U", "L": "R", "R": "L"}


class EightPuzzle(Problem):
    """Sliding-tile puzzle on a ``side x side`` board (default 3x3 = 8-puzzle)."""

    def __init__(self, initial: Sequence[int], side: int = 3, heuristic: str = "manhattan"):
        initial = tuple(initial)
        if len(initial) != side * side:
            raise ValueError(f"state must have {side * side} entries for a {side}x{side} board")
        if sorted(initial) != list(range(side * side)):
            raise ValueError("state must be a permutation of 0..side*side-1")
        if heuristic not in ("manhattan", "misplaced", "zero"):
            raise ValueError("heuristic must be 'manhattan', 'misplaced' or 'zero'")
        self.side = side
        self.heuristic_name = heuristic
        # Diameter of the puzzle graph: the longest optimal solution. Known exactly
        # for small boards (3x3 = 31, 2x2 = 6). Any deeper bound means unsolvable,
        # which lets IDA* terminate on infeasible instances.
        self.max_solution_cost = {2: 6, 3: 31, 4: 80}.get(side)
        goal = tuple(list(range(1, side * side)) + [0])
        super().__init__(initial=initial, goal=goal)
        # Precompute each tile's goal (row, col) for the Manhattan heuristic.
        self._goal_pos = {tile: divmod(i, side) for i, tile in enumerate(goal)}

    # -- problem interface -------------------------------------------------
    def successors(self, state: Tuple[int, ...]) -> Iterable[Tuple[str, Tuple[int, ...]]]:
        side = self.side
        blank = state.index(0)
        br, bc = divmod(blank, side)
        for name, (dr, dc) in _MOVES.items():
            nr, nc = br + dr, bc + dc
            if 0 <= nr < side and 0 <= nc < side:
                ni = nr * side + nc
                lst = list(state)
                lst[blank], lst[ni] = lst[ni], lst[blank]
                yield name, tuple(lst)

    def heuristic(self, state: Tuple[int, ...]) -> float:
        if self.heuristic_name == "zero":
            return 0.0
        if self.heuristic_name == "misplaced":
            return float(self._misplaced(state))
        return float(self._manhattan(state))

    # -- heuristics (exposed for testing) ---------------------------------
    def _misplaced(self, state: Tuple[int, ...]) -> int:
        return sum(1 for i, t in enumerate(state) if t != 0 and t != self.goal[i])

    def _manhattan(self, state: Tuple[int, ...]) -> int:
        total = 0
        side = self.side
        for i, tile in enumerate(state):
            if tile == 0:
                continue
            r, c = divmod(i, side)
            gr, gc = self._goal_pos[tile]
            total += abs(r - gr) + abs(c - gc)
        return total

    # -- solvability & construction ---------------------------------------
    @staticmethod
    def is_solvable(state: Sequence[int], side: int = 3) -> bool:
        """Return True if ``state`` can reach the ordered goal.

        For odd board width the parity of inversions must be even. For even width,
        the rule also depends on the blank's row. This lets tests assert that an
        unsolvable instance is genuinely unsolvable rather than merely hard.
        """
        tiles = [t for t in state if t != 0]
        inversions = sum(
            1
            for i in range(len(tiles))
            for j in range(i + 1, len(tiles))
            if tiles[i] > tiles[j]
        )
        if side % 2 == 1:
            return inversions % 2 == 0
        blank_row_from_bottom = side - (list(state).index(0) // side)
        if blank_row_from_bottom % 2 == 0:
            return inversions % 2 == 1
        return inversions % 2 == 0

    @classmethod
    def scrambled(cls, k: int, side: int = 3, seed: Optional[int] = None,
                  heuristic: str = "manhattan") -> "EightPuzzle":
        """Build an instance by applying ``k`` random moves from the solved board.

        The walk never immediately undoes its previous move, so the shortest
        solution is *at most* ``k`` (it can be shorter if the walk looped back).
        This gives tests a known upper bound on the optimal length.
        """
        rng = random.Random(seed)
        goal = tuple(list(range(1, side * side)) + [0])
        state = goal
        last: Optional[str] = None
        problem = cls(goal, side=side, heuristic=heuristic)
        for _ in range(k):
            moves = [
                (name, succ)
                for name, succ in problem.successors(state)
                if last is None or name != _INVERSE[last]
            ]
            name, state = rng.choice(moves)
            last = name
        return cls(state, side=side, heuristic=heuristic)

    # -- display -----------------------------------------------------------
    def render(self, state: Tuple[int, ...]) -> str:
        side = self.side
        width = len(str(side * side - 1))
        rows = []
        for r in range(side):
            cells = state[r * side:(r + 1) * side]
            rows.append(" ".join("_".rjust(width) if c == 0 else str(c).rjust(width) for c in cells))
        return "\n".join(rows)
