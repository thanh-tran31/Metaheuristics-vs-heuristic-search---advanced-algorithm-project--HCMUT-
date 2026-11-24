"""Unit tests for the 8-connected A* algorithm and Octile heuristic."""

from __future__ import annotations

import copy
import math
import unittest

from src.astar import (
    SQRT2,
    check_admissibility_and_consistency,
    compute_heuristic_matrix,
    compute_path_cost,
    get_last_stats,
    is_valid_grid_path,
    octile_distance,
    solve,
    solve_with_stats,
)


class TestAStarContract(unittest.TestCase):
    """Test suite verifying strict compliance with the Project Contract."""

    def test_contract_example_grid(self) -> None:
        """Test the exact example from IO_of_Algo_and_Repo_Format.md."""
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        grid_copy = copy.deepcopy(grid)
        start = (0, 0)
        goal = (4, 4)

        path = solve(grid, start, goal)

        # 1. Input grid must not be mutated
        self.assertEqual(grid, grid_copy)

        # 2. Path must be valid and connect start to goal
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual(path[0], start)
        self.assertEqual(path[-1], goal)
        self.assertTrue(is_valid_grid_path(grid, path, start, goal))

        # 3. Cost must match Dijkstra exact shortest path
        dijkstra_res = solve_with_stats(grid, start, goal, heuristic="dijkstra")
        self.assertAlmostEqual(compute_path_cost(path), dijkstra_res.cost, places=9)

    def test_start_equals_goal(self) -> None:
        """When start == goal, return [(x, y)] with cost 0."""
        grid = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        path = solve(grid, (1, 2), (1, 2))
        self.assertEqual(path, [(1, 2)])
        stats = get_last_stats()
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.cost, 0.0)
        self.assertEqual(stats.expanded_nodes, 0)

    def test_non_square_grid_xy_indexing(self) -> None:
        """Verify (x, y) = (column, row) indexing on a wide rectangular grid (7x3)."""
        grid = [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0],
        ]
        start = (0, 2)  # bottom-left: col 0, row 2
        goal = (6, 0)   # top-right: col 6, row 0

        path = solve(grid, start, goal)
        self.assertIsNotNone(path)
        self.assertTrue(is_valid_grid_path(grid, path, start, goal))

    def test_pure_diagonal_and_cardinal_costs(self) -> None:
        """On an obstacle-free grid, A* cost equals exact Octile distance."""
        grid = [[0] * 8 for _ in range(6)]
        start = (1, 1)
        goal = (6, 4)  # dx = 5, dy = 3 -> 3 diagonals + 2 cardinals

        res = solve_with_stats(grid, start, goal)
        self.assertIsNotNone(res.path)
        expected_cost = 3 * SQRT2 + 2 * 1.0
        self.assertAlmostEqual(res.cost, expected_cost, places=9)
        self.assertAlmostEqual(octile_distance(start, goal), expected_cost, places=9)
        self.assertAlmostEqual(compute_path_cost(res.path), expected_cost, places=9)

    def test_no_corner_cutting_single_obstacle(self) -> None:
        """Diagonal step must be blocked if one adjacent cardinal cell is an obstacle."""
        # Moving from (0, 0) to (1, 1) when (1, 0) is blocked:
        # Must go (0, 0) -> (0, 1) -> (1, 1) with cost 2.0, NOT diagonal sqrt(2).
        grid = [
            [0, 1],
            [0, 0],
        ]
        res = solve_with_stats(grid, (0, 0), (1, 1))
        self.assertEqual(res.path, [(0, 0), (0, 1), (1, 1)])
        self.assertAlmostEqual(res.cost, 2.0, places=9)

    def test_no_corner_cutting_diagonal_pinch_unreachable(self) -> None:
        """Diagonal pinch [[0, 1], [1, 0]] must be completely impassable."""
        grid = [
            [0, 1],
            [1, 0],
        ]
        path = solve(grid, (0, 0), (1, 1))
        self.assertIsNone(path)

    def test_unreachable_goal_returns_none(self) -> None:
        """When goal is walled off, solve() must return None."""
        grid = [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        self.assertIsNone(solve(grid, (0, 0), (3, 0)))

    def test_invalid_inputs_handled_safely(self) -> None:
        """Out-of-bounds or obstacle start/goal should return None."""
        grid = [
            [0, 1],
            [0, 0],
        ]
        self.assertIsNone(solve(grid, (1, 0), (1, 1)))  # start on obstacle
        self.assertIsNone(solve(grid, (0, 0), (1, 0)))  # goal on obstacle
        self.assertIsNone(solve(grid, (-1, 0), (1, 1)))  # negative coordinate
        self.assertIsNone(solve(grid, (0, 0), (2, 1)))  # out of bounds
        self.assertIsNone(solve([], (0, 0), (0, 0)))    # empty grid


class TestHeuristicsAndOptimality(unittest.TestCase):
    """Test suite verifying mathematical properties of heuristics and A* optimality."""

    def setUp(self) -> None:
        self.maze_grid = [
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ]

    def test_admissibility_and_consistency(self) -> None:
        """Octile, Euclidean, Chebyshev, and Zero must be admissible and consistent."""
        goal = (9, 7)
        for h_name in ("octile", "euclidean", "chebyshev", "zero"):
            admissible, consistent = check_admissibility_and_consistency(
                self.maze_grid, goal, heuristic=h_name
            )
            self.assertTrue(admissible, f"{h_name} should be admissible")
            self.assertTrue(consistent, f"{h_name} should be consistent")

        # Manhattan is NOT admissible on 8-connected grids with diagonal moves
        open_grid = [[0] * 5 for _ in range(5)]
        admissible_l1, consistent_l1 = check_admissibility_and_consistency(
            open_grid, (4, 4), heuristic="manhattan"
        )
        self.assertFalse(admissible_l1)
        self.assertFalse(consistent_l1)

    def test_astar_matches_dijkstra_optimal_cost_with_fewer_expansions(self) -> None:
        """A* with Octile must match Dijkstra's optimal cost while expanding <= nodes."""
        open_obstacle_grid = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
        start = (1, 2)
        goal = (10, 6)

        res_octile = solve_with_stats(open_obstacle_grid, start, goal, heuristic="octile")
        res_euclid = solve_with_stats(open_obstacle_grid, start, goal, heuristic="euclidean")
        res_dijkstra = solve_with_stats(open_obstacle_grid, start, goal, heuristic="dijkstra")

        self.assertAlmostEqual(res_octile.cost, res_dijkstra.cost, places=9)
        self.assertAlmostEqual(res_euclid.cost, res_dijkstra.cost, places=9)
        self.assertLess(res_octile.expanded_nodes, res_dijkstra.expanded_nodes)
        self.assertLessEqual(res_octile.expanded_nodes, res_euclid.expanded_nodes)

    def test_heuristic_matrix_for_hybrid_aco(self) -> None:
        """Verify compute_heuristic_matrix in octile and exact modes."""
        goal = (9, 7)
        octile_mat = compute_heuristic_matrix(self.maze_grid, goal, mode="octile")
        exact_mat = compute_heuristic_matrix(self.maze_grid, goal, mode="exact")

        self.assertEqual(octile_mat[goal[1]][goal[0]], 0.0)
        self.assertEqual(exact_mat[goal[1]][goal[0]], 0.0)
        # Obstacle cells must have inf distance
        self.assertTrue(math.isinf(octile_mat[0][3]))
        self.assertTrue(math.isinf(exact_mat[0][3]))
        # Exact cost from (0, 0) to goal must match A* optimal path cost
        res = solve_with_stats(self.maze_grid, (0, 0), goal)
        self.assertAlmostEqual(exact_mat[0][0], res.cost, places=9)


if __name__ == "__main__":
    unittest.main()

