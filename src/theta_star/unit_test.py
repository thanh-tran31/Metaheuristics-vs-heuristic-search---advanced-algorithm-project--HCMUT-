import copy
import heapq
import math
import os
from pathlib import Path
import random
import sys
import unittest
from fractions import Fraction

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.path.join(ROOT, "src"))

from theta_star import solve  

FIXTURE_MAP = os.path.join(ROOT, "theta_small.map")


# --------------------------------------------------------------------------- helpers
def load_movingai_map(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    rows = lines[4:4 + height]
    assert len(rows) == height and all(len(r) == width for r in rows)
    return [[0 if ch in ".GS" else 1 for ch in r] for r in rows]


def segment_touches_blocked(grid, a, b):
    p0 = (Fraction(2 * a[0] + 1, 2), Fraction(2 * a[1] + 1, 2))
    d = (Fraction(b[0] - a[0]), Fraction(b[1] - a[1]))
    for cy in range(min(a[1], b[1]), max(a[1], b[1]) + 1):
        for cx in range(min(a[0], b[0]), max(a[0], b[0]) + 1):
            if grid[cy][cx] == 0:
                continue
            t0, t1 = Fraction(0), Fraction(1)
            ok = True
            for axis, lo in ((0, cx), (1, cy)):
                p, dd = p0[axis], d[axis]
                if dd == 0:
                    if not (lo <= p <= lo + 1):
                        ok = False
                        break
                else:
                    ta, tb = (lo - p) / dd, (lo + 1 - p) / dd
                    if ta > tb:
                        ta, tb = tb, ta
                    t0, t1 = max(t0, ta), min(t1, tb)
                    if t0 > t1:
                        ok = False
                        break
            if ok:
                return True
    return False


def path_cost(path):
    return sum(math.hypot(p[0] - q[0], p[1] - q[1]) for p, q in zip(path, path[1:]))


def assert_valid_path(tc, grid, start, goal, path):
    tc.assertIsInstance(path, list)
    tc.assertEqual(path[0], start)
    tc.assertEqual(path[-1], goal)
    h, w = len(grid), len(grid[0])
    for (x, y) in path:
        tc.assertTrue(0 <= x < w and 0 <= y < h, f"waypoint {(x, y)} ngoài bản đồ")
        tc.assertEqual(grid[y][x], 0, f"waypoint {(x, y)} là vật cản")
    for p, q in zip(path, path[1:]):
        tc.assertNotEqual(p, q, "hai waypoint liên tiếp trùng nhau")
        tc.assertFalse(segment_touches_blocked(grid, p, q),
                       f"đoạn {p}->{q} xuyên/chạm vật cản")


def dijkstra_cost(grid, start, goal):
    """Chi phí tối ưu trên lưới 8 hướng, no corner-cutting (tham chiếu cho A*)."""
    h, w = len(grid), len(grid[0])
    dist = {start: 0.0}
    pq = [(0.0, start)]
    while pq:
        d, (x, y) = heapq.heappop(pq)
        if (x, y) == goal:
            return d
        if d > dist[(x, y)]:
            continue
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < w and 0 <= ny < h) or grid[ny][nx]:
                    continue
                if dx and dy and (grid[y][nx] or grid[ny][x]):
                    continue
                nd = d + math.hypot(dx, dy)
                if nd < dist.get((nx, ny), float("inf")):
                    dist[(nx, ny)] = nd
                    heapq.heappush(pq, (nd, (nx, ny)))
    return None


# --------------------------------------------------------------------------- tests
class TestContractExamples(unittest.TestCase):
    def test_example_from_contract(self):
        grid = [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        start, goal = (0, 0), (4, 4)
        path = solve(grid, start, goal)
        self.assertIsNotNone(path)
        assert_valid_path(self, grid, start, goal, path)
        self.assertLessEqual(path_cost(path), dijkstra_cost(grid, start, goal) + 1e-9)

    def test_start_equals_goal(self):
        grid = [[0, 0], [0, 0]]
        self.assertEqual(solve(grid, (1, 1), (1, 1)), [(1, 1)])

    def test_open_map_straight_line(self):
        grid = [[0] * 8 for _ in range(6)]
        self.assertEqual(solve(grid, (0, 0), (7, 5)), [(0, 0), (7, 5)])

    def test_no_path_returns_none(self):
        grid = [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ]
        self.assertIsNone(solve(grid, (0, 0), (2, 2)))

    def test_no_corner_cutting_diagonal(self):
        grid = [[0, 1], [1, 0]]
        self.assertIsNone(solve(grid, (0, 0), (1, 1)))

    def test_one_side_blocked_diagonal_is_rejected(self):
        grid = [[0, 1], [0, 0]]
        path = solve(grid, (0, 0), (1, 1))
        self.assertEqual(path, [(0, 0), (0, 1), (1, 1)])

    def test_los_not_through_vertex_between_blocked_cells(self):
        # đường chéo dài đi đúng qua đỉnh ô giữa 2 vật cản -> không được nối thẳng
        grid = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ]
        path = solve(grid, (0, 0), (2, 2))
        print(f"path = {path}")
        self.assertIsNotNone(path)
        assert_valid_path(self, grid, (0, 0), (2, 2), path)
        self.assertGreater(len(path), 2)

    def test_any_angle_shortcut(self):
        # Theta* phải rút gọn được: ít điểm hơn đường lưới thuần túy
        grid = [[0] * 10 for _ in range(10)]
        grid[4][4] = 1
        path = solve(grid, (0, 0), (9, 9))
        assert_valid_path(self, grid, (0, 0), (9, 9), path)
        self.assertLessEqual(len(path), 4)


class TestInputHandling(unittest.TestCase):
    def test_grid_not_modified(self):
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        before = copy.deepcopy(grid)
        solve(grid, (0, 0), (2, 2))
        self.assertEqual(grid, before)

    def test_invalid_inputs_raise(self):
        grid = [[0, 1], [0, 0]]
        with self.assertRaises(ValueError):
           solve(grid, (5, 5), (0, 0))        # start ngoài bản đồ
        with self.assertRaises(ValueError):
            solve(grid, (0, 0), (1, 0))        # goal là vật cản
        with self.assertRaises(ValueError):
            solve([[0, 0], [0]], (0, 0), (1, 0))  # không chữ nhật
        with self.assertRaises(ValueError):
            solve([], (0, 0), (0, 0))


class TestFixtureMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = load_movingai_map(FIXTURE_MAP)

    def test_map_size(self):
        self.assertEqual((len(self.grid), len(self.grid[0])), (8, 12))

    def test_open_row_is_direct(self):
        self.assertEqual(solve(self.grid, (0, 0), (11, 0)), [(0, 0), (11, 0)])

    def test_around_obstacles(self):
        queries = [
            ((0, 0), (11, 7)),
            ((0, 7), (11, 0)),
            ((3, 3), (11, 7)),   
            ((3, 4), (0, 0)),
            ((11, 7), (4, 6)),
        ]
        for start, goal in queries:
            with self.subTest(start=start, goal=goal):
                path = solve(self.grid, start, goal)
                print(f"path = {path}")
                self.assertIsNotNone(path)
                assert_valid_path(self, self.grid, start, goal, path)
                cost = path_cost(path)
                self.assertLessEqual(cost, dijkstra_cost(self.grid, start, goal) + 1e-9)
                self.assertGreaterEqual(cost, math.hypot(goal[0] - start[0],
                                                          goal[1] - start[1]) - 1e-9)


class TestRandomGrids(unittest.TestCase):
    def test_random_against_dijkstra(self):
        rng = random.Random(12345)
        checked = 0
        for _ in range(300):
            h, w = rng.randint(3, 12), rng.randint(3, 12)
            grid = [[1 if rng.random() < 0.28 else 0 for _ in range(w)] for _ in range(h)]
            free = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == 0]
            if len(free) < 2:
                continue
            start, goal = rng.sample(free, 2)
            ref = dijkstra_cost(grid, start, goal)
            path = solve(grid, start, goal)
            if ref is None:
                self.assertIsNone(path)
            else:
                self.assertIsNotNone(path)
                assert_valid_path(self, grid, start, goal, path)
                self.assertLessEqual(path_cost(path), ref + 1e-9)
                checked += 1
        self.assertGreater(checked, 50)


if __name__ == "__main__":
    unittest.main(verbosity=2)