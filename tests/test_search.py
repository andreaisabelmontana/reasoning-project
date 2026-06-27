"""Correctness tests for the domains, heuristics and search algorithms.

These assert the properties that make the project trustworthy: every algorithm
returns a *valid* plan, all four agree on the same *optimal cost*, the heuristics
are *admissible* on sampled states, A* expands *fewer* nodes than BFS, and
*unsolvable* instances are reported as failures.
"""

from __future__ import annotations

import random

import pytest

from reasoning import (
    EightPuzzle,
    MissionariesAndCannibals,
    astar,
    bfs,
    ida_star,
    uniform_cost,
)
from reasoning.search import ALGORITHMS

ALGOS = [bfs, uniform_cost, astar, ida_star]


# --------------------------------------------------------------------------
# Plan validity: replaying a returned plan must reach a goal state.
# --------------------------------------------------------------------------
def _replay(problem, result):
    """Apply each action in order from the initial state and return the final state."""
    state = problem.initial
    for action in result.actions:
        succ_map = {a: s for a, s in problem.successors(state)}
        assert action in succ_map, f"action {action!r} not legal in {state!r}"
        state = succ_map[action]
    return state


@pytest.mark.parametrize("algo", ALGOS)
def test_eight_puzzle_plan_is_valid(algo):
    problem = EightPuzzle.scrambled(k=14, seed=3)
    result = algo(problem)
    assert result.found
    final = _replay(problem, result)
    assert problem.goal_test(final)
    # Reported cost must equal the number of moves (uniform step cost of 1).
    assert result.cost == result.length


@pytest.mark.parametrize("algo", ALGOS)
def test_missionaries_plan_is_valid(algo):
    problem = MissionariesAndCannibals(3, 3, 2)
    result = algo(problem)
    assert result.found
    final = _replay(problem, result)
    assert problem.goal_test(final)


# --------------------------------------------------------------------------
# Optimality: a k-scramble solves in <= k, and all algorithms agree on the cost.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(6))
def test_scramble_solves_within_bound(seed):
    k = 12
    problem = EightPuzzle.scrambled(k=k, seed=seed)
    result = astar(problem)
    assert result.found
    # The non-backtracking random walk gives an upper bound of k on the optimum.
    assert result.length <= k


@pytest.mark.parametrize("seed", range(6))
def test_all_algorithms_agree_on_optimal_cost(seed):
    problem = EightPuzzle.scrambled(k=15, seed=seed)
    costs = {name: fn(problem).cost for name, fn in ALGORITHMS.items()}
    reference = costs["BFS"]
    for name, cost in costs.items():
        assert cost == reference, f"{name} cost {cost} != BFS {reference}"


def test_missionaries_optimal_is_eleven():
    problem = MissionariesAndCannibals(3, 3, 2)
    costs = {name: fn(problem).cost for name, fn in ALGORITHMS.items()}
    assert set(costs.values()) == {11}
    assert problem.optimal_crossings_known == 11


# --------------------------------------------------------------------------
# Efficiency: a good heuristic makes A* expand fewer nodes than BFS.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_astar_expands_fewer_nodes_than_bfs(seed):
    problem = EightPuzzle.scrambled(k=18, seed=seed)
    a = astar(problem)
    b = bfs(problem)
    assert a.found and b.found
    assert a.cost == b.cost
    assert a.nodes_expanded < b.nodes_expanded


def test_manhattan_dominates_misplaced():
    """Manhattan never expands more nodes than misplaced-tiles (it dominates)."""
    base = EightPuzzle.scrambled(k=20, seed=11)
    man = EightPuzzle(base.initial, heuristic="manhattan")
    mis = EightPuzzle(base.initial, heuristic="misplaced")
    rm = astar(man)
    rs = astar(mis)
    assert rm.cost == rs.cost
    assert rm.nodes_expanded <= rs.nodes_expanded


# --------------------------------------------------------------------------
# Admissibility: heuristic never exceeds the true optimal cost-to-go.
# --------------------------------------------------------------------------
def test_eight_puzzle_heuristics_are_admissible():
    """On sampled states, h(s) <= true optimal distance for both heuristics."""
    rng = random.Random(0)
    for _ in range(40):
        k = rng.randint(2, 16)
        seed = rng.randint(0, 10_000)
        scrambled = EightPuzzle.scrambled(k=k, seed=seed)
        # True optimal distance via an admissible-and-consistent A* (Manhattan).
        true_cost = astar(EightPuzzle(scrambled.initial, heuristic="manhattan")).cost
        for hname in ("manhattan", "misplaced", "zero"):
            p = EightPuzzle(scrambled.initial, heuristic=hname)
            h0 = p.heuristic(p.initial)
            assert h0 <= true_cost, f"{hname} overestimates: {h0} > {true_cost}"


def test_missionaries_heuristic_is_admissible():
    """Sample reachable states and check h never exceeds the true cost-to-go."""
    problem = MissionariesAndCannibals(3, 3, 2)
    # Enumerate reachable states by BFS over the full graph.
    from collections import deque

    seen = {problem.initial}
    queue = deque([problem.initial])
    while queue:
        s = queue.popleft()
        for _, succ in problem.successors(s):
            if succ not in seen:
                seen.add(succ)
                queue.append(succ)
    for s in seen:
        sub = MissionariesAndCannibals(3, 3, 2)
        sub.initial = s
        true_cost = astar(sub).cost
        assert problem.heuristic(s) <= true_cost


# --------------------------------------------------------------------------
# Unsolvability: detect instances with no path to the goal.
# --------------------------------------------------------------------------
def test_unsolvable_eight_puzzle_is_detected():
    # Swapping tiles 1 and 2 from the goal flips inversion parity -> unsolvable.
    bad = (2, 1, 3, 4, 5, 6, 7, 8, 0)
    assert not EightPuzzle.is_solvable(bad)
    problem = EightPuzzle(bad)
    # Graph-search algorithms detect it by exhausting the reachable half of the space.
    for algo in (bfs, uniform_cost, astar):
        result = algo(problem)
        assert not result.found
        assert result.cost == float("inf")


def test_unsolvable_ida_star_terminates():
    """IDA* (no closed set) must still report failure on an unsolvable instance.

    Uses the 2x2 puzzle (diameter 6) so the bound-capped IDA* terminates quickly;
    swapping two tiles flips inversion parity and makes it unreachable.
    """
    bad = (2, 1, 3, 0)  # 2x2: tiles 1,2 swapped from the (1,2,3,0) goal
    assert not EightPuzzle.is_solvable(bad, side=2)
    result = ida_star(EightPuzzle(bad, side=2))
    assert not result.found
    assert result.cost == float("inf")


def test_solvable_classifier_agrees_with_search():
    """is_solvable must match whether search actually finds a solution.

    Validated on the smaller 2x2 (3-puzzle): its state space is only 24 states, so
    we can exhaustively confirm the inversion-parity classifier against real search
    on *every* configuration -- a stronger check than sampling the 8-puzzle.
    """
    import itertools

    for state in itertools.permutations(range(4)):  # all 24 states of the 2x2 puzzle
        solvable = EightPuzzle.is_solvable(state, side=2)
        found = bfs(EightPuzzle(state, side=2)).found
        assert solvable == found, f"mismatch on {state}: classifier={solvable}, search={found}"


def test_unsolvable_missionaries_capacity_one():
    """With a 1-seat boat and missionaries present, 3M/3C cannot cross safely."""
    problem = MissionariesAndCannibals(3, 3, 1)
    result = bfs(problem)
    assert not result.found
