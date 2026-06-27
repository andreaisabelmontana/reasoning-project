"""Comparison harness: run several algorithms on one instance and tabulate them.

The point is to make the efficiency gap concrete — every algorithm finds an
optimal-cost solution, but the number of nodes expanded differs by orders of
magnitude, and a good heuristic (Manhattan) lets A*/IDA* expand far fewer nodes
than uninformed BFS or uniform-cost search.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from .problem import Problem
from .search import ALGORITHMS, SearchResult


def run_all(problem: Problem) -> Dict[str, SearchResult]:
    """Run every registered algorithm on ``problem`` and return their results."""
    return {name: fn(problem) for name, fn in ALGORITHMS.items()}


def format_table(results: Dict[str, SearchResult]) -> str:
    """Render a fixed-width comparison table of the results."""
    header = f"{'algorithm':<10}{'solved':<8}{'cost':<8}{'len':<6}{'expanded':>10}"
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        cost = "-" if not r.found else f"{r.cost:g}"
        length = "-" if not r.found else str(r.length)
        lines.append(
            f"{name:<10}{('yes' if r.found else 'no'):<8}{cost:<8}{length:<6}{r.nodes_expanded:>10}"
        )
    return "\n".join(lines)


def timed_run_all(problem: Problem) -> Tuple[Dict[str, SearchResult], Dict[str, float]]:
    """Like :func:`run_all` but also return wall-clock milliseconds per algorithm."""
    results: Dict[str, SearchResult] = {}
    times: Dict[str, float] = {}
    for name, fn in ALGORITHMS.items():
        t0 = time.perf_counter()
        results[name] = fn(problem)
        times[name] = (time.perf_counter() - t0) * 1000.0
    return results, times
