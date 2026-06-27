"""Solve a few instances and print the optimal length + nodes expanded per algorithm.

Run:  python demo.py

Everything printed here is the result of a real search at run time -- nothing is
hard-coded. The headline is the efficiency comparison: a good heuristic lets A*
and IDA* expand a small fraction of the nodes that uninformed BFS/UCS do, while
all four agree on the same optimal solution cost.
"""

from __future__ import annotations

from reasoning import EightPuzzle, MissionariesAndCannibals
from reasoning.compare import format_table, run_all


def banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def solve_eight_puzzle() -> None:
    banner("8-PUZZLE  --  scrambled 18 moves from solved (seed 7)")
    problem = EightPuzzle.scrambled(k=18, seed=7, heuristic="manhattan")
    print("start:")
    print(problem.render(problem.initial))
    print(f"\nManhattan heuristic of start state: {problem.heuristic(problem.initial):g}")
    results = run_all(problem)
    print()
    print(format_table(results))

    astar = results["A*"]
    bfs = results["BFS"]
    if astar.found and bfs.nodes_expanded:
        ratio = bfs.nodes_expanded / max(astar.nodes_expanded, 1)
        print(f"\noptimal solution length: {astar.length} moves")
        print(f"A* expanded {astar.nodes_expanded} nodes vs BFS {bfs.nodes_expanded} "
              f"-- {ratio:.1f}x fewer")
        print("solution (blank moves):", " ".join(astar.actions))


def solve_eight_puzzle_hard() -> None:
    banner("8-PUZZLE  --  scrambled 27 moves from solved (seed 1)")
    problem = EightPuzzle.scrambled(k=27, seed=1, heuristic="manhattan")
    print("start:")
    print(problem.render(problem.initial))
    results = run_all(problem)
    print()
    print(format_table(results))
    astar, bfs = results["A*"], results["BFS"]
    if astar.found:
        ratio = bfs.nodes_expanded / max(astar.nodes_expanded, 1)
        print(f"\noptimal solution length: {astar.length} moves")
        print(f"A* expanded {astar.nodes_expanded} nodes vs BFS {bfs.nodes_expanded} "
              f"-- {ratio:.1f}x fewer")


def solve_missionaries() -> None:
    banner("MISSIONARIES & CANNIBALS  --  3M 3C, boat capacity 2")
    problem = MissionariesAndCannibals(3, 3, 2)
    results = run_all(problem)
    print(format_table(results))
    astar = results["A*"]
    if astar.found:
        print(f"\noptimal solution: {astar.length} crossings")
        print("plan:")
        for action in astar.actions:
            print("   ", action)


def main() -> None:
    solve_eight_puzzle()
    solve_eight_puzzle_hard()
    solve_missionaries()
    print()


if __name__ == "__main__":
    main()
