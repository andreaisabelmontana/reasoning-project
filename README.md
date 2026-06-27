# reasoning-project

Single-agent automated reasoning as state-space search. Model a puzzle's rules as
states and actions, then search for an optimal solution — the foundation of how
machines plan. Pure Python, standard library only.

- **Live page:** https://andreaisabelmontana.github.io/reasoning-project/
- **Index of all my builds:** https://andreaisabelmontana.github.io/coursework-rebuilds/

## What it does

A problem is defined by four things — an initial state, a goal test, a successor
function, and a step cost. Any domain that implements this interface can be solved
by any of the search algorithms. The point of the project is to compare those
algorithms on the *same* instance: they all return an optimal-cost solution, but
the number of nodes they expand differs by orders of magnitude.

## Domains

- **8-puzzle** (`reasoning/domains/eight_puzzle.py`) — the 3×3 sliding-tile puzzle
  (generalised to any N×N). State is a flat tuple; the action slides a tile into
  the blank. Includes an inversion-parity solvability test and a `scrambled(k)`
  constructor that walks `k` non-backtracking moves from the solved board, giving a
  known upper bound of `k` on the optimal length.
- **Missionaries and cannibals** (`reasoning/domains/missionaries.py`) — a
  river-crossing constraint puzzle. `m` missionaries and `c` cannibals cross with a
  boat of capacity `cap`; missionaries must never be outnumbered on either bank. The
  classic 3M/3C/cap-2 instance has a known optimal of 11 crossings.

## Search algorithms (`reasoning/search.py`)

| Algorithm | Strategy | Optimality |
|-----------|----------|------------|
| **BFS** | breadth-first graph search | optimal in length when step costs are equal |
| **UCS** | uniform-cost (Dijkstra) | optimal for any non-negative step cost |
| **A\*** | best-first on `f = g + h` | optimal when `h` is admissible/consistent |
| **IDA\*** | iterative-deepening A\* | same optimal cost as A\*, memory linear in depth |

Each returns the solution path, its cost, and the number of nodes expanded.

## Heuristics and admissibility

For the 8-puzzle:

- **Misplaced tiles** — count of tiles not in their goal cell. Admissible: every
  misplaced tile needs at least one move.
- **Manhattan distance** — sum of each tile's grid distance to its goal cell.
  Admissible (a move closes at most one unit of distance) and it *dominates*
  misplaced-tiles, so A\* with Manhattan expands no more nodes — usually far fewer.

For missionaries and cannibals, a lower bound on the remaining crossings derived
from how many people are still on the near bank.

Admissibility is not assumed — it is tested. The suite scrambles many random
states, computes the true optimal cost-to-go with a consistent A\*, and asserts the
heuristic never exceeds it. For missionaries it does the same over *every* reachable
state.

## Efficiency comparison (real `demo.py` output)

A good heuristic turns a broad blind search into a near-beeline. All algorithms find
the same optimal cost; only the work differs.

**8-puzzle, scrambled 18 moves (seed 7) — optimal length 12:**

| algorithm | cost | nodes expanded |
|-----------|------|----------------|
| BFS  | 12 | 821 |
| UCS  | 12 | 1204 |
| A\*  | 12 | 33 |
| IDA\* | 12 | 28 |

A\* expands **24.9×** fewer nodes than BFS.

**8-puzzle, scrambled 27 moves (seed 1) — optimal length 23:**

| algorithm | cost | nodes expanded |
|-----------|------|----------------|
| BFS  | 23 | 80,439 |
| UCS  | 23 | 119,498 |
| A\*  | 23 | 1,193 |
| IDA\* | 23 | 784 |

A\* expands **67.4×** fewer nodes than BFS; IDA\* fewer still.

**Missionaries & cannibals, 3M/3C/cap-2 — optimal 11 crossings:** all four
algorithms agree on the 11-crossing plan (the state space is tiny: BFS expands 13).

## Run it

```bash
pip install -r requirements.txt   # pytest only; the package itself is stdlib
python demo.py                    # solve the instances above, print the comparison
python -m pytest -q               # run the test suite
```

The package has no runtime dependencies — `requirements.txt` lists only `pytest`.

## Tests

```
$ python -m pytest -q
.................................                                        [100%]
33 passed in 7.37s
```

The 33 tests check: every algorithm returns a *valid* plan (replaying the actions
reaches the goal); BFS, UCS, A\* and IDA\* agree on the same *optimal cost*; a
`k`-scramble solves in ≤ `k`; the missionaries optimum is 11; A\* expands fewer nodes
than BFS; Manhattan dominates misplaced-tiles; both 8-puzzle heuristics and the
missionaries heuristic are *admissible* on sampled/enumerated states; the
inversion-parity solvability test agrees with real search on all 24 configurations
of the 2×2 puzzle; and *unsolvable* instances (parity-flipped 8-puzzle, capacity-1
missionaries) are reported as failures by every algorithm.

## Layout

```
reasoning/
  problem.py              problem interface + search-tree Node
  search.py               BFS, UCS, A*, IDA*  -> SearchResult(path, cost, nodes_expanded)
  compare.py              run-all harness + comparison table
  domains/
    eight_puzzle.py       N x N sliding-tile puzzle + heuristics + solvability
    missionaries.py       river-crossing puzzle + heuristic
demo.py                   solves the instances above, prints the comparison
tests/test_search.py      33 tests
```

## License

MIT — see [LICENSE](LICENSE).
