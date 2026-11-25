"""Unit tests for 8-connected A*, 1-based Heap structures, and Admissible/Consistent properties."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import unittest

from src.astar import (
    HEAP_MODES,
    SQRT2,
    AStarSolver,
    BinaryMinHeap,
    FastIntMinHeap,
    IndexedMinHeap,
    analyze_heuristic_properties,
    check_admissibility_and_consistency,
    compute_heuristic_matrix,
    compute_path_cost,
    get_last_stats,
    is_valid_grid_path,
    left_child_index,
    octile_distance,
    parent_index,
    right_child_index,
    solve,
    solve_with_stats,
)
from src.common.parser import MovingAIParser

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "movingai"


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
        start, goal = (0, 0), (4, 4)

        path = solve(grid, start, goal)

        self.assertEqual(grid, grid_copy)
        self.assertIsNotNone(path)
        assert path is not None
        self.assertEqual((path[0], path[-1]), (start, goal))
        self.assertTrue(is_valid_grid_path(grid, path, start, goal))

        dijkstra_res = solve_with_stats(grid, start, goal, heuristic="dijkstra")
        self.assertAlmostEqual(compute_path_cost(path), dijkstra_res.cost, places=9)

    def test_start_equals_goal(self) -> None:
        """When start == goal, return [(x, y)] with cost 0."""
        grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        path = solve(grid, (1, 2), (1, 2))
        self.assertEqual(path, [(1, 2)])
        stats = get_last_stats()
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.cost, 0.0)
        self.assertEqual(stats.expanded_nodes, 0)

    def test_solver_wrapper_compatibility(self) -> None:
        """Ensure AStarSolver wrapper remains compatible with integration tests."""
        solver = AStarSolver()
        self.assertIs(solver.heuristic_fn, octile_distance)
        self.assertEqual((solver.weight, solver.tie_breaking), (1.0, "high_g"))
        path = solver.solve([[0, 0], [0, 0]], (0, 0), (1, 1))
        self.assertEqual(path, [(0, 0), (1, 1)])

    def test_non_square_grid_xy_indexing(self) -> None:
        """Verify (x, y) = (column, row) indexing on a wide rectangular grid (7x3)."""
        grid = [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0, 0, 0],
        ]
        start, goal = (0, 2), (6, 0)
        path = solve(grid, start, goal)
        self.assertTrue(is_valid_grid_path(grid, path, start, goal))

    def test_pure_diagonal_and_cardinal_costs(self) -> None:
        """On an obstacle-free grid, A* cost equals exact Octile distance."""
        grid = [[0] * 8 for _ in range(6)]
        start, goal = (1, 1), (6, 4)
        res = solve_with_stats(grid, start, goal)
        expected_cost = 3 * SQRT2 + 2 * 1.0
        self.assertAlmostEqual(res.cost, expected_cost, places=9)
        self.assertAlmostEqual(octile_distance(start, goal), expected_cost, places=9)
        self.assertAlmostEqual(compute_path_cost(res.path), expected_cost, places=9)

    def test_no_corner_cutting_single_and_pinch(self) -> None:
        """Verify single corner block forces detour and diagonal pinch returns None."""
        res = solve_with_stats([[0, 1], [0, 0]], (0, 0), (1, 1))
        self.assertEqual(res.path, [(0, 0), (0, 1), (1, 1)])
        self.assertAlmostEqual(res.cost, 2.0, places=9)
        self.assertIsNone(solve([[0, 1], [1, 0]], (0, 0), (1, 1)))

    def test_unreachable_and_invalid_inputs(self) -> None:
        """Unreachable goals and out-of-bounds/blocked inputs return None safely."""
        walled = [
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0],
        ]
        self.assertIsNone(solve(walled, (0, 0), (3, 0)))

        grid = [[0, 1], [0, 0]]
        self.assertIsNone(solve(grid, (1, 0), (1, 1)))
        self.assertIsNone(solve(grid, (0, 0), (1, 0)))
        self.assertIsNone(solve(grid, (-1, 0), (1, 1)))
        self.assertIsNone(solve(grid, (0, 0), (2, 1)))
        self.assertIsNone(solve([], (0, 0), (0, 0)))
        self.assertIsNone(solve([[True, 0], [0, 0]], (0, 0), (1, 1)))  # type: ignore[list-item]


class TestOneBasedHeapDesign(unittest.TestCase):
    """Verify 1-based Heap indexing (`i // 2`, `2 * i`, `2 * i + 1`) matching lecture slides."""

    def test_one_based_index_formulas_and_slide_tree(self) -> None:
        """Verify exact 1..15 array & tree relations from the reference Heap diagram."""
        # Exact 15-element Min-Heap array at indices 1..15 from the reference diagram:
        slide_values = [0, 1, 3, 4, 6, 7, 8, 10, 12, 20, 24, 13, 9, 11, 21]

        heap_std: BinaryMinHeap[int] = BinaryMinHeap()
        heap_fast = FastIntMinHeap()
        for val in slide_values:
            heap_std.push(val)
            heap_fast.push(val)

        # Active items occupy indices 1..15 (len == 15, internal list length == 16)
        self.assertEqual(len(heap_std), 15)
        self.assertEqual(len(heap_std.data), 16)
        self.assertEqual(heap_std.data[1:], slide_values)
        self.assertEqual(heap_fast.data[1:], slide_values)

        # Verify parentIndex = i // 2, leftChildIndex = 2*i, rightChildIndex = 2*i + 1
        self.assertEqual(parent_index(8), 4)
        self.assertEqual(parent_index(9), 4)
        self.assertEqual(parent_index(15), 7)
        self.assertEqual(left_child_index(4), 8)
        self.assertEqual(right_child_index(4), 9)
        self.assertEqual(right_child_index(7), 15)

        for i in range(2, 16):
            p = parent_index(i)
            self.assertLessEqual(heap_std.data[p], heap_std.data[i])

        # Popping all 15 elements must return sorted ascending order
        expected_sorted = sorted(slide_values)
        self.assertEqual([heap_std.pop() for _ in range(15)], expected_sorted)
        self.assertEqual([heap_fast.pop() for _ in range(15)], expected_sorted)

    def test_indexed_min_heap_one_based_decrease_key(self) -> None:
        """Verify 1-based `IndexedMinHeap` slots `1..size`, `pos[v]`, and `decrease_key`."""
        heap = IndexedMinHeap(10)
        self.assertTrue(heap.push_or_decrease(3, 50))
        self.assertTrue(heap.push_or_decrease(1, 30))
        self.assertTrue(heap.push_or_decrease(7, 40))
        self.assertEqual(len(heap), 3)
        # Root is at 1-based index 1 (`node 1` with key 30)
        self.assertEqual(heap.pos[1], 1)
        self.assertEqual(heap.pos[5], 0)  # 0 means absent

        self.assertFalse(heap.push_or_decrease(3, 60))
        self.assertTrue(heap.push_or_decrease(3, 10))
        # Node 3 with new key 10 must sift up to root slot 1!
        self.assertEqual(heap.pos[3], 1)
        self.assertEqual(len(heap), 3)
        self.assertEqual(heap.decrease_key_count, 1)

        popped = [heap.pop_min() for _ in range(3)]
        self.assertEqual(popped, [(3, 10), (1, 30), (7, 40)])
        self.assertFalse(heap)

    def test_all_heap_modes_match_optimal_cost_and_properties(self) -> None:
        """`fast`, `decrease_key`, and `lazy_tuple` must find identical optimal costs."""
        rng = random.Random(2026)
        for _ in range(40):
            w, h = 16, 16
            grid = [[1 if rng.random() < 0.25 else 0 for _ in range(w)] for _ in range(h)]
            free = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == 0]
            start, goal = rng.sample(free, 2)

            results = {
                mode: solve_with_stats(grid, start, goal, heap_mode=mode)
                for mode in HEAP_MODES
            }
            ref_cost = results["lazy_tuple"].cost
            for mode, res in results.items():
                if math.isinf(ref_cost):
                    self.assertIsNone(res.path, f"{mode} should return None")
                else:
                    self.assertTrue(is_valid_grid_path(grid, res.path, start, goal))
                    self.assertAlmostEqual(res.cost, ref_cost, places=9, msg=mode)
                    self.assertLessEqual(res.expanded_nodes, len(free))

            self.assertEqual(results["decrease_key"].stale_pops, 0)
            self.assertLessEqual(
                results["decrease_key"].peak_open_size,
                results["lazy_tuple"].peak_open_size,
            )
            self.assertLessEqual(results["fast"].heap_ops, results["lazy_tuple"].heap_ops)


class TestAdmissibilityConsistencyAndMovingAI(unittest.TestCase):
    """Verify heuristic dominance, Manhattan counterexample, and MovingAI ground truth."""

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

    def test_admissibility_consistency_and_dominance(self) -> None:
        """Verify zero <= chebyshev <= euclidean <= octile <= h* and expansion ordering."""
        goal = (9, 7)
        reports = {
            name: analyze_heuristic_properties(self.maze_grid, goal, heuristic=name)
            for name in ("zero", "chebyshev", "euclidean", "octile")
        }
        for name, rep in reports.items():
            self.assertTrue(rep.is_admissible, f"{name} must be admissible")
            self.assertTrue(rep.is_consistent, f"{name} must be consistent")

        self.assertLess(reports["zero"].mean_tightness_ratio, reports["chebyshev"].mean_tightness_ratio)
        self.assertLessEqual(reports["chebyshev"].mean_tightness_ratio, reports["euclidean"].mean_tightness_ratio)
        self.assertLessEqual(reports["euclidean"].mean_tightness_ratio, reports["octile"].mean_tightness_ratio)

        admissible, consistent = check_admissibility_and_consistency(
            [[0] * 5 for _ in range(5)], (4, 4), heuristic="manhattan"
        )
        self.assertFalse(admissible)
        self.assertFalse(consistent)

    def test_heuristic_matrix_for_hybrid_aco(self) -> None:
        """Verify compute_heuristic_matrix in octile and exact modes."""
        goal = (9, 7)
        octile_mat = compute_heuristic_matrix(self.maze_grid, goal, mode="octile")
        exact_mat = compute_heuristic_matrix(self.maze_grid, goal, mode="exact")

        self.assertEqual(octile_mat[goal[1]][goal[0]], 0.0)
        self.assertEqual(exact_mat[goal[1]][goal[0]], 0.0)
        self.assertTrue(math.isinf(octile_mat[0][3]))
        self.assertTrue(math.isinf(exact_mat[0][3]))
        res = solve_with_stats(self.maze_grid, (0, 0), goal)
        self.assertAlmostEqual(exact_mat[0][0], res.cost, places=9)

    def test_movingai_maze_and_random_scenarios(self) -> None:
        """Cross-check A* against MovingAIParser on maze-32-32-2 and random-32-32-20."""
        for family, stem in (("maze", "maze-32-32-2"), ("random", "random-32-32-20")):
            map_path = DATA_ROOT / family / "map" / f"{stem}.map"
            scen_path = DATA_ROOT / family / "scenarios" / f"{stem}-random-1.scen"
            parser = MovingAIParser(map_path, scen_path)
            grid, _, _ = parser.get_input(0)
            for idx in range(parser.scenario_count):
                info = parser.get_scenario_info(idx)
                for mode in HEAP_MODES:
                    res = solve_with_stats(grid, info["start"], info["goal"], heap_mode=mode)
                    self.assertTrue(is_valid_grid_path(grid, res.path, info["start"], info["goal"]))
                    self.assertAlmostEqual(res.cost, info["optimal_length"], delta=1e-6)


if __name__ == "__main__":
    unittest.main()
