"""Missionaries and cannibals: a classic river-crossing constraint puzzle.

``m`` missionaries and ``c`` cannibals must cross a river using a boat that holds
at most ``capacity`` people and cannot cross empty. On either bank, if missionaries
are present they must not be outnumbered by cannibals, or they are eaten.

A state is ``(m_left, c_left, boat)`` where ``boat`` is 1 if the boat is on the
starting (left) bank and 0 if on the far (right) bank. The classic instance is
3 missionaries, 3 cannibals, boat capacity 2; its shortest solution is 11 crossings.

An admissible heuristic estimates the remaining crossings from how many people are
still on the left bank: ferrying ``p`` people across with a boat of capacity ``cap``
needs at least ``ceil((p - (boat_on_left? 0 : 1)) ... )`` round trips. We use the
simple lower bound ``ceil(people_left / capacity)`` rounded for the return trips,
which never overestimates because each crossing moves at most ``capacity`` people
one way.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Tuple

from ..problem import Problem

State = Tuple[int, int, int]  # (missionaries_left, cannibals_left, boat_on_left)


class MissionariesAndCannibals(Problem):
    """River-crossing puzzle parameterised by population and boat capacity."""

    def __init__(self, missionaries: int = 3, cannibals: int = 3, capacity: int = 2):
        if capacity < 1:
            raise ValueError("boat capacity must be at least 1")
        self.m = missionaries
        self.c = cannibals
        self.capacity = capacity
        # An optimal path never repeats a state, so the count of distinct states is
        # a safe upper bound on solution length -- enough for IDA* to give up on an
        # unsolvable instance instead of deepening forever.
        self.max_solution_cost = (missionaries + 1) * (cannibals + 1) * 2
        super().__init__(initial=(missionaries, cannibals, 1), goal=(0, 0, 0))

    # -- safety & legality -------------------------------------------------
    def _safe(self, ml: int, cl: int) -> bool:
        """No bank may have missionaries outnumbered by cannibals."""
        mr, cr = self.m - ml, self.c - cl
        if not (0 <= ml <= self.m and 0 <= cl <= self.c):
            return False
        if ml > 0 and cl > ml:
            return False
        if mr > 0 and cr > mr:
            return False
        return True

    def _boat_loads(self) -> List[Tuple[int, int]]:
        """All non-empty (missionaries, cannibals) loads the boat can carry."""
        loads = []
        for mm in range(self.capacity + 1):
            for cc in range(self.capacity + 1):
                if 1 <= mm + cc <= self.capacity:
                    loads.append((mm, cc))
        return loads

    # -- problem interface -------------------------------------------------
    def successors(self, state: State) -> Iterable[Tuple[str, State]]:
        ml, cl, boat = state
        direction = -1 if boat == 1 else 1  # moving people from left to right reduces left counts
        for mm, cc in self._boat_loads():
            # Can only load people from the bank the boat is on.
            if boat == 1:
                if mm > ml or cc > cl:
                    continue
            else:
                if mm > self.m - ml or cc > self.c - cl:
                    continue
            nml = ml + direction * mm
            ncl = cl + direction * cc
            if self._safe(nml, ncl):
                action = f"{'L->R' if boat == 1 else 'R->L'} {mm}M {cc}C"
                yield action, (nml, ncl, 1 - boat)

    def heuristic(self, state: State) -> float:
        """Lower bound on remaining crossings to empty the left bank.

        Each crossing carries at most ``capacity`` people one way, but the boat
        must return (one person back) between deliveries, so net throughput is at
        most ``capacity - 1`` people per round trip — except the final crossing,
        which needs no return. ``2*ceil((people-capacity)/(capacity-1)) + 1`` is a
        standard admissible bound; with capacity 1 we fall back to a trivial bound.
        """
        ml, cl, boat = state
        people = ml + cl
        if people == 0:
            return 0.0
        cap = self.capacity
        if cap <= 1:
            # Each person needs its own crossing; admissible trivial bound.
            return float(people)
        if people <= cap and boat == 1:
            return 1.0
        # Round trips to whittle the bank down to one final boatload, plus that load.
        trips = 2 * math.ceil((people - cap) / (cap - 1)) + 1
        return float(trips)

    def render(self, state: State) -> str:
        ml, cl, boat = state
        mr, cr = self.m - ml, self.c - cl
        left = f"{'M'*ml}{'C'*cl}".ljust(self.m + self.c) or "-"
        right = f"{'M'*mr}{'C'*cr}".ljust(self.m + self.c) or "-"
        b = "(boat)" if boat == 1 else "      "
        return f"L[{left}] {b}~~~~{'      ' if boat == 1 else '(boat)'} [{right}]R"

    @property
    def optimal_crossings_known(self) -> "int | None":
        """The textbook optimal for the canonical 3M/3C/cap-2 instance (11)."""
        if (self.m, self.c, self.capacity) == (3, 3, 2):
            return 11
        return None
