"""Exercise the public solvers on equivalent inputs using independent oracles."""
import math
import unittest
from copy import deepcopy
from pathlib import Path

from src import astar, theta_star
from src.aco import ACOSolver
from src.common.parser import MovingAIParser
from src.common.validation import validate_path
from tests.test_aco import reference_distance

DATA = Path(__file__).resolve().parents[1] / "data" / "movingai"
# First record in scenario file 1, independently known from the datasets.
MAPS = (
    ("random", "random-32-32-20", 819, (5, 16), (31, 24), 31.31370850),
    ("maze", "maze-32-32-2", 666, (15, 2), (1, 27), 64.31370850),
)


def first_record(family, name):
    directory = DATA / family
    return MovingAIParser(directory / "map" / f"{name}.map",
                          directory / "scenarios" / f"{name}-random-1.scen")


class TestParserIntegration(unittest.TestCase):
    def test_real_maps_and_records(self):
        for family, name, free, start, goal, length in MAPS:
            with self.subTest(map=name):
                parser = first_record(family, name)
                grid, s, g = parser.get_input(0)
                self.assertEqual((s, g), (start, goal))
                self.assertEqual((len(grid), set(map(len, grid))), (32, {32}))
                self.assertEqual(sum(row.count(0) for row in grid), free)
                self.assertEqual(sum(row.count(1) for row in grid), 1024 - free)
                self.assertEqual((grid[s[1]][s[0]], grid[g[1]][g[0]]), (0, 0))
                self.assertAlmostEqual(parser.get_scenario_info(0)["optimal_length"], length)
                grid[s[1]][s[0]] = 1
                self.assertEqual(parser.get_input(0)[0][s[1]][s[0]], 0)


class TestAlgorithmIntegration(unittest.TestCase):
    def checked_solve(self, algorithm, solve, inputs, *, allow_none=False):
        grid, start, goal = inputs
        original = deepcopy(grid)
        path = solve(grid, start, goal)
        self.assertEqual(grid, original)
        result = validate_path(original, start, goal, path, algorithm=algorithm)
        if allow_none and path is None:
            self.assertEqual(result.status, "none")
        else:
            self.assertIsNotNone(path)
            self.assertTrue(result.is_valid, result.errors)
            self.assertEqual((path[0], path[-1]), (start, goal))
        return path, result.path_length

    def assert_distance(self, actual, expected):
        self.assertTrue(math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-6),
                        (actual, expected))

    def test_real_paths(self):
        for family, name, *_ in MAPS:
            parser = first_record(family, name)
            optimal = reference_distance(*parser.get_input(0))
            with self.subTest(map=name, algorithm="astar"):
                _, length = self.checked_solve("astar", astar.solve, parser.get_input(0))
                self.assert_distance(length, optimal)
                self.assert_distance(length, parser.get_scenario_info(0)["optimal_length"])
            with self.subTest(map=name, algorithm="theta_star"):
                path, length = self.checked_solve("theta_star", theta_star.solve, parser.get_input(0))
                self.assertGreaterEqual(length + 1e-9, math.dist(path[0], path[-1]))
                self.assert_distance(length, sum(math.dist(a, b) for a, b in zip(path, path[1:])))
            for seed in (0, 7):
                with self.subTest(map=name, algorithm="aco", seed=seed):
                    solver = ACOSolver(seed=seed)
                    optional = (family, seed) != ("random", 0)
                    path, length = self.checked_solve("aco", solver.solve, parser.get_input(0),
                                                      allow_none=optional)
                    if length is not None:
                        self.assertGreaterEqual(length + 1e-6, optimal)
                    solver.solve([[0]], (0, 0), (0, 0))
                    for replay in (solver, ACOSolver(seed=seed)):
                        again, _ = self.checked_solve("aco", replay.solve, parser.get_input(0),
                                                      allow_none=optional)
                        self.assertEqual(again, path)

    def test_astar_against_independent_dijkstra(self):
        solver = astar.AStarSolver()
        self.assertIs(solver.heuristic_fn, astar.octile_distance)
        self.assertEqual((solver.weight, solver.tie_breaking), (1, "high_g"))
        for grid, start, goal in (
            ([[0] * 6 for _ in range(3)], (5, 0), (0, 2)),
            ([[0, 1, 0, 0], [0, 0, 0, 1]], (3, 0), (0, 1)),
            ([[0, 1, 0], [0, 0, 0]], (0, 0), (2, 0)),
            ([[0, 0], [0, 0]], (0, 0), (1, 1)),
            ([[0, 1], [0, 0]], (0, 0), (1, 1)),
            ([[0, 1], [1, 0]], (0, 0), (1, 1)),
            ([[0]], (0, 0), (0, 0)),
        ):
            with self.subTest(grid=grid, start=start, goal=goal):
                optimal = reference_distance(grid, start, goal)
                _, length = self.checked_solve("astar", astar.solve, (grid, start, goal),
                                               allow_none=math.isinf(optimal))
                if math.isfinite(optimal):
                    self.assert_distance(length, optimal)
                else:
                    self.assertIsNone(length)

    def test_theta_segments_and_disconnection(self):
        for grid, goal, expected in (([[0] * 3 for _ in range(2)], (2, 1), math.sqrt(5)),
                                     ([[0, 1, 0]], (2, 0), None)):
            with self.subTest(goal=goal):
                path, length = self.checked_solve("theta_star", theta_star.solve,
                                                  (grid, (0, 0), goal), allow_none=expected is None)
                if expected is None:
                    self.assertIsNone(path)
                else:
                    self.assertEqual(path, [(0, 0), goal])
                    self.assert_distance(length, expected)

    def test_boolean_waypoint(self):
        grid = [[0, 0, 0]]
        for algorithm in ("astar", "theta_star", "aco"):
            with self.subTest(algorithm=algorithm):
                for x, expected in ((1, "valid"), (True, "invalid")):
                    path = [(0, 0), (x, 0), (2, 0)]
                    self.assertEqual(validate_path(grid, path[0], path[-1], path,
                                                   algorithm=algorithm).status, expected)
