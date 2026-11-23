import copy
import heapq
import itertools
import math
import unittest
from collections import Counter
from src.aco import ACOSolver, solve
from src.aco.solver import edge, heuristic, neighbors, octile, path_cost
from math import sqrt
from random import Random
from time import perf_counter


GRID = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
START, GOAL = (0, 0), (4, 4)


def independent_neighbors(grid, u):
    # Independent validator: do not call the implementation's neighbors helper.
    x, y = u
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < len(grid[0]) and 0 <= ny < len(grid)):
            continue
        if grid[ny][nx] != 0:
            continue
        if dx and dy and (grid[y][nx] or grid[ny][x]):
            continue
        yield (nx, ny), math.hypot(dx, dy)


def reference_distance(grid, start, goal):
    queue = [(0.0, start)]
    dist = {start: 0.0}
    while queue:
        cost, u = heapq.heappop(queue)
        if cost != dist[u]:
            continue
        if u == goal:
            return cost
        for v, length in independent_neighbors(grid, u):
            candidate = cost + length
            if candidate < dist.get(v, math.inf):
                dist[v] = candidate
                heapq.heappush(queue, (candidate, v))
    return math.inf


# Original implementation retained here for seeded regression checks.
_REFERENCE_DIAG = sqrt(2)

def _reference_octile(a, b):
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy) + (_REFERENCE_DIAG - 1) * min(dx, dy)

def _reference_neighbors(grid, u):
    x, y = u
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            nx, ny = x + dx, y + dy
            if dx == dy == 0:
                continue
            if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])):
                continue
            if grid[ny][nx]:
                continue
            if dx and dy and (grid[y][nx] or grid[ny][x]):
                continue
            yield (nx, ny)

def _reference_edge(a, b):
    return (a, b) if a < b else (b, a)

def original_solve(grid, start, goal, *, seed=0, ants=12, rounds=15,
          budget=512, alpha=1, beta=2, rho=0.2, stats=None):
    t0 = perf_counter()
    if stats is not None:
        stats.update(evaluations=0, improvements=[])
    if start == goal:
        if stats is not None:
            stats["improvements"].append((perf_counter() - t0, 0))
        return [start]

    rng = Random(seed)
    bonus = Counter()
    best, best_cost = None, float("inf")
    for _ in range(rounds):
        deposit = Counter()
        for _ in range(ants):
            path, seen = [start], {start}
            for _ in range(budget):
                u = path[-1]
                options = [v for v in _reference_neighbors(grid, u) if v not in seen]
                if stats is not None:
                    stats["evaluations"] += len(options)
                if not options:
                    path.pop()
                    if not path:
                        break
                    continue

                weights = []
                for v in options:
                    tau = 1 + bonus[_reference_edge(u, v)]
                    eta = 1 / (1 + _reference_octile(v, goal))
                    weights.append(tau**alpha * eta**beta)
                v = rng.choices(options, weights=weights, k=1)[0]
                path.append(v)
                seen.add(v)
                if v == goal:
                    cost = sum(_reference_octile(a, b) for a, b in zip(path, path[1:]))
                    for a, b in zip(path, path[1:]):
                        deposit[_reference_edge(a, b)] += 10 / cost
                    if cost < best_cost:
                        best, best_cost = path, cost
                        if stats is not None:
                            stats["improvements"].append((perf_counter() - t0, cost))
                    break
        for e in bonus:
            bonus[e] *= 1 - rho
        bonus.update(deposit)
    return best


class SolverTests(unittest.TestCase):
    def assert_valid_path(self, grid, start, goal, path):
        self.assertIsNotNone(path)
        self.assertEqual((path[0], path[-1]), (start, goal))
        self.assertEqual(len(path), len(set(path)))
        measured = 0.0
        for a, b in zip(path, path[1:]):
            choices = dict(independent_neighbors(grid, a))
            self.assertIn(b, choices)
            measured += choices[b]
        self.assertAlmostEqual(measured, path_cost(path))

    def test_octile_and_eta(self):
        self.assertAlmostEqual(octile((1, 2), GOAL), 1 + 2 * math.sqrt(2))
        self.assertAlmostEqual(heuristic((1, 2), GOAL), 1 / (2 + 2 * math.sqrt(2)))
        self.assertEqual(heuristic(GOAL, GOAL), 1.0)

    def test_no_corner_cutting(self):
        self.assertNotIn((1, 1), list(neighbors([[0, 1], [0, 0]], (0, 0))))
        self.assertNotIn((1, 1), list(neighbors([[0, 0], [1, 0]], (0, 0))))
        self.assertIn((1, 1), list(neighbors([[0, 0], [0, 0]], (0, 0))))

    def test_evaporation_only_affects_bonus(self):
        e = edge((0, 0), (1, 0))
        bonus = Counter({e: 2.0})
        ACOSolver(rho=0.2).evaporate_and_deposit(bonus, Counter({e: 5.0}))
        self.assertAlmostEqual(bonus[e], 6.6)
        self.assertAlmostEqual(1 + bonus[e], 7.6)
        ACOSolver(rho=1).evaporate_and_deposit(bonus, Counter())
        self.assertEqual(1 + bonus[e], 1.0)

    def test_start_equals_goal(self):
        self.assertEqual(ACOSolver(ants=0, budget=0).solve([[0]], (0, 0), (0, 0)), [(0, 0)])

    def test_small_budget_can_fail_on_connected_map(self):
        self.assertEqual(reference_distance([[0, 0, 0]], (0, 0), (2, 0)), 2.0)
        self.assertIsNone(ACOSolver(ants=1, rounds=1, budget=1).solve([[0, 0, 0]], (0, 0), (2, 0)))

    def test_unique_corridor(self):
        grid = [[0, 0, 0], [1, 1, 0], [0, 0, 0]]
        path = ACOSolver(ants=1, rounds=1, budget=20).solve(grid, (0, 0), (0, 2))
        self.assert_valid_path(grid, (0, 0), (0, 2), path)
        self.assertEqual(path_cost(path), 6.0)
        self.assertEqual(reference_distance(grid, (0, 0), (0, 2)), 6.0)

    def test_grid_unchanged_and_repeat_seed(self):
        before = copy.deepcopy(GRID)
        solver = ACOSolver(seed=7)
        first = solver.solve(GRID, START, GOAL)
        self.assert_valid_path(GRID, START, GOAL, first)
        self.assertEqual(first, solver.solve(GRID, START, GOAL))
        self.assertEqual(GRID, before)

    def test_refactor_matches_original_30_seeds(self):
        cases = [(GRID, START, GOAL), ([[0] * 4 for _ in range(4)], (0, 0), (3, 3)),
                 ([[0, 1], [1, 0]], (0, 0), (1, 1))]
        for grid, start, goal in cases:
            for seed in range(30):
                with self.subTest(seed=seed, grid=grid):
                    kwargs = dict(seed=seed, ants=3, rounds=3, budget=30)
                    actual = ACOSolver(**kwargs).solve(grid, start, goal)
                    self.assertEqual(actual, original_solve(grid, start, goal, **kwargs))
                    if actual is not None:
                        self.assert_valid_path(grid, start, goal, actual)

    def test_all_128_small_maps_reachability(self):
        # Keep start/goal free; check all 2^7 obstacle layouts on the 3x3 grid.
        cells = [(x, y) for y in range(3) for x in range(3) if (x, y) not in ((0, 0), (2, 2))]
        for mask in itertools.product((0, 1), repeat=7):
            grid = [[0] * 3 for _ in range(3)]
            for (x, y), value in zip(cells, mask):
                grid[y][x] = value
            with self.subTest(mask=mask):
                path = ACOSolver(ants=1, rounds=1, budget=18).solve(grid, (0, 0), (2, 2))
                exists = math.isfinite(reference_distance(grid, (0, 0), (2, 2)))
                self.assertEqual(path is not None, exists)
                if path is not None:
                    self.assert_valid_path(grid, (0, 0), (2, 2), path)

    def test_wrapper_matches_default_instance(self):
        self.assertEqual(solve(GRID, START, GOAL), ACOSolver().solve(GRID, START, GOAL))

    def test_instances_keep_separate_configuration(self):
        grid = [[0, 0, 0]]
        limited = ACOSolver(ants=1, rounds=1, budget=1)
        sufficient = ACOSolver(ants=1, rounds=1, budget=2)
        self.assertIsNone(limited.solve(grid, (0, 0), (2, 0)))
        self.assertEqual(sufficient.solve(grid, (0, 0), (2, 0)), [(0, 0), (1, 0), (2, 0)])
        self.assertIsNone(limited.solve(grid, (0, 0), (2, 0)))

    def test_reusing_instance_starts_fresh_search(self):
        solver = ACOSolver(seed=7, ants=3, rounds=3, budget=30)
        first = solver.solve(GRID, START, GOAL)
        solver.solve([[0]], (0, 0), (0, 0))
        self.assertEqual(first, solver.solve(GRID, START, GOAL))

    def test_invalid_input_and_parameters(self):
        for grid, start, goal in [([], (0, 0), (0, 0)), ([[0], [0, 0]], (0, 0), (0, 0)),
                                  ([[2]], (0, 0), (0, 0)), ([[1]], (0, 0), (0, 0)),
                                  ([[0]], (-1, 0), (0, 0))]:
            with self.assertRaises(ValueError):
                solve(grid, start, goal)
        for kwargs in (dict(rho=1.1), dict(beta=-1), dict(ants=1.5), dict(alpha=float("nan")), dict(q=-1), dict(budget=True)):
            with self.assertRaises(ValueError):
                ACOSolver(**kwargs)


if __name__ == "__main__":
    unittest.main()
