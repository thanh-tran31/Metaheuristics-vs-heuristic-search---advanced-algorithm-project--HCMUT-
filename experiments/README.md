# Pathfinding experiments

## Quick start

Run from the repository root with Python 3.10 or later. The tools use the standard library.

```sh
python run_tests.py
python -m experiments.benchmark
python -m experiments.benchmark --full
```

Open `experiments/results/smoke/benchmark_report.md` in Markdown preview for the tables and figures. Open `benchmark_results.csv` beside it for the individual attempts. Full results go to `experiments/results/full/`.

Use `python run_tests.py --verbose` for individual method names. Use `--algorithm astar`, `--algorithm theta_star` or `--algorithm aco` on a benchmark command to run one solver. A single-algorithm run has its own directory, such as `experiments/results/smoke-aco/`. The longer `--profile smoke|full` and `--algorithms astar theta_star aco` forms are also supported. Choose one form for each option.

## Sources and ownership

| File or directory | Responsibility |
|---|---|
| `src/astar/`, `src/theta_star/`, `src/aco/` | Algorithm implementations owned by their maintainers. |
| `tests/test_astar.py`, `tests/test_aco.py`, `src/theta_star/unit_test.py` | Original maintainer suites for algorithm behavior, including the Theta* fixture beside its suite. |
| `tests/test_integration.py` | Exercises the public solvers with independent grids, MovingAI references, test-side Dijkstra and geometric validation. Requires ACO success on the previously verified random-map record with seed 0. |
| `run_tests.py` | Discovers the repository tests and the original Theta* suite, reports each suite and returns the combined exit status. |
| `src/common/parser.py`, `data/movingai/` | MovingAI parser and original map/scenario inputs. Coordinates are (column, row). |
| `src/common/validation.py` | Independently checks endpoints, bounds, traversability and legal grid steps or any-angle line of sight, then sums Euclidean segment lengths. |
| `experiments/benchmark.py` | Scenario selection, serial runner, two-stage aggregation, SVGs and report generation. |
| `experiments/configs/selected_scenarios.json` | Fixed 24-record manifest used for the comparative study. |

## Methodology

**Scenario selection.** Within each map, scan scenario files in filename order and records in file order, retaining the first occurrence of each ordered start/goal pair. Sort by grid reference length, filename and zero-based index. For N distinct pairs, select ranks `(2*i+1)*N//24` for i from 0 through 11. To check the saved selection without writing it:

```sh
python -c "from experiments.benchmark import load_scenarios, select_scenarios; assert select_scenarios() == load_scenarios('full')"
```

**Execution.** Smoke runs the first manifest record from each map twice per algorithm (12 attempts). Full runs all 24 records with five repetitions each for A* and Theta* and 30 ACO seeds per record (960 attempts). Every call gets a fresh grid. The serial nanosecond clock surrounds only the public solve call. Parser setup, ACO construction, external validation, aggregation and drawing occur outside it. A* uses octile, weight 1 and high-g tie-breaking. Theta* uses its defaults. Each ACO instance uses 12 ants, 15 rounds, budget 512, alpha 1, beta 2, rho 0.2 and q 10. Seeds restart at zero on each scenario.

**CSV.** Map, scenario file, zero-based index, algorithm and one-based repetition identify an attempt. The file records endpoints, seed, solver milliseconds, outcome, independent validity and length, grid reference, available comparisons and diagnostics. Outcomes are `found`, `none`, `invalid_path` and `error`. Blank means unavailable. A valid singleton has length zero. A* supplies expanded nodes, heap insertions and evaluated neighbor candidates. ACO ants and rounds are configuration values. Theta* counters are unavailable.

**Success and quality.** Success divides valid successes by attempts. Path validity divides valid paths by returned path results, excluding absent results and exceptions. Coverage counts scenarios solved at least once. Grid gap is the signed percentage difference from a positive grid reference for A* and ACO. Theta* and ACO also receive a signed difference from the same scenario’s median positive valid A* baseline when available. Theta* can use any-angle shortcuts.

**Aggregation.** Compute mean, median, sample SD (at least two values), and interpolated p90 at rank `0.9*(n-1)` within each scenario. Runtime distinguishes all timed attempts from valid ones. The report averages available scenario statistics with equal scenario weights. Main quality comparisons use only scenarios solved by every selected algorithm, so harder instances cannot disappear from just one algorithm’s mean. Use the accompanying full-collection success and coverage percentages to interpret exclusions.

**Reproducibility.** The representative figures show the first measured repetition of the first selected scenario per map, including failures. Metadata records the selected IDs, repetitions, seeds, parameters, timing policy, UTC timestamp, revision, Python and machine details, and SHA-256 hashes of sources and inputs. Rerunning the same selection replaces its named artifacts. Use `--output-dir PATH` to retain a separate run. This specifies an exact directory. The runner rejects directories whose existing metadata identifies a different profile or algorithm set.
