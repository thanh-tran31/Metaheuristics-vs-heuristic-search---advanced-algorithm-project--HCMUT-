"""Heuristic functions and heuristic matrix utilities for 2D grid pathfinding."""

from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Callable

Point = tuple[int, int]
Grid = list[list[int]]
HeuristicFn = Callable[[Point, Point], float]

SQRT2: float = math.sqrt(2.0)
EPSILON: float = 1e-9


def octile_distance(a: Point, b: Point) -> float:
    """Compute the Octile distance between two grid coordinates (x, y).

    On an 8-connected grid with cardinal cost 1 and diagonal cost sqrt(2),
    Octile distance is exact in the absence of obstacles, making it both
    admissible and consistent (monotonic).
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if dx >= dy:
        return (dx - dy) + SQRT2 * dy
    return (dy - dx) + SQRT2 * dx


def euclidean_distance(a: Point, b: Point) -> float:
    """Compute the Euclidean (L2) distance between two coordinates (x, y).

    Admissible and consistent on 8-connected grids (since L2 <= Octile).
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev_distance(a: Point, b: Point) -> float:
    """Compute the Chebyshev (L-infinity) distance between two coordinates (x, y).

    Admissible and consistent on 8-connected grids with diagonal cost sqrt(2) >= 1.
    """
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


def manhattan_distance(a: Point, b: Point) -> float:
    """Compute the Manhattan (L1) distance between two coordinates (x, y).

    Note: Manhattan distance is NOT admissible on 8-connected grids because a
    diagonal move reduces (dx + dy) by 2 while costing only sqrt(2) < 2.
    Included for ablation studies.
    """
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def zero_heuristic(a: Point, b: Point) -> float:
    """Return 0.0 (reduces A* to Dijkstra's uniform-cost search)."""
    return 0.0


HEURISTIC_REGISTRY: dict[str, HeuristicFn] = {
    "octile": octile_distance,
    "euclidean": euclidean_distance,
    "chebyshev": chebyshev_distance,
    "manhattan": manhattan_distance,
    "zero": zero_heuristic,
    "dijkstra": zero_heuristic,
}


def get_heuristic_fn(heuristic: str | HeuristicFn = "octile") -> HeuristicFn:
    """Resolve a heuristic name or callable into a heuristic function."""
    if callable(heuristic):
        return heuristic
    key = heuristic.strip().lower()
    if key not in HEURISTIC_REGISTRY:
        valid = ", ".join(sorted(HEURISTIC_REGISTRY))
        raise ValueError(f"Unknown heuristic '{heuristic}'. Valid options: {valid}")
    return HEURISTIC_REGISTRY[key]


def check_admissibility_and_consistency(
    grid: Grid,
    goal: Point,
    heuristic: str | HeuristicFn = "octile",
) -> tuple[bool, bool]:
    """Empirically verify admissibility and consistency of a heuristic on a grid.

    Computes the exact shortest-path distance h*(n) from all reachable cells to
    `goal` via reverse Dijkstra, then checks:
      1. Admissibility: h(u, goal) <= h*(u) + EPSILON for all reachable u.
      2. Consistency: h(u, goal) <= c(u, v) + h(v, goal) + EPSILON for all valid edges (u, v).

    Returns:
        (is_admissible, is_consistent)
    """
    h_fn = get_heuristic_fn(heuristic)
    exact_dist = compute_heuristic_matrix(grid, goal, mode="exact")
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    is_admissible = True
    is_consistent = True

    for y in range(height):
        row = grid[y]
        for x in range(width):
            if row[x] != 0:
                continue
            u = (x, y)
            h_u = h_fn(u, goal)
            h_star_u = exact_dist[y][x]
            if math.isfinite(h_star_u) and h_u > h_star_u + EPSILON:
                is_admissible = False

            # Check consistency across all valid 8-connected outgoing edges
            free_right = x + 1 < width and row[x + 1] == 0
            free_left = x - 1 >= 0 and row[x - 1] == 0
            free_down = y + 1 < height and grid[y + 1][x] == 0
            free_up = y - 1 >= 0 and grid[y - 1][x] == 0

            cardinal_neighbors = []
            if free_right:
                cardinal_neighbors.append((x + 1, y))
            if free_left:
                cardinal_neighbors.append((x - 1, y))
            if free_down:
                cardinal_neighbors.append((x, y + 1))
            if free_up:
                cardinal_neighbors.append((x, y - 1))

            for v in cardinal_neighbors:
                if h_u > 1.0 + h_fn(v, goal) + EPSILON:
                    is_consistent = False

            diagonal_neighbors = []
            if free_right and free_down and grid[y + 1][x + 1] == 0:
                diagonal_neighbors.append((x + 1, y + 1))
            if free_right and free_up and grid[y - 1][x + 1] == 0:
                diagonal_neighbors.append((x + 1, y - 1))
            if free_left and free_down and grid[y + 1][x - 1] == 0:
                diagonal_neighbors.append((x - 1, y + 1))
            if free_left and free_up and grid[y - 1][x - 1] == 0:
                diagonal_neighbors.append((x - 1, y - 1))

            for v in diagonal_neighbors:
                if h_u > SQRT2 + h_fn(v, goal) + EPSILON:
                    is_consistent = False

    return is_admissible, is_consistent


def compute_heuristic_matrix(
    grid: Grid,
    goal: Point,
    mode: str = "octile",
) -> list[list[float]]:
    """Precompute a 2D heuristic matrix H[y][x] towards `goal`.

    Designed to support both A* analysis and the Week 48 extension
    (Hybrid ACO-A*: guiding ants using an A* heuristic distance matrix).

    Args:
        grid: 2D map where 0 = traversable, 1 = obstacle.
        goal: Target coordinate (gx, gy).
        mode:
            - "octile": Geometric Octile distance to `goal` (O(W*H), obstacle-unaware
              except obstacles themselves are set to inf).
            - "euclidean": Geometric Euclidean distance to `goal`.
            - "exact": Exact 8-connected cost-to-go from every cell to `goal`
              computed via reverse Dijkstra respecting obstacles and no corner-cutting.
            - "bfs": Hop count (unweighted 8-connected steps) from `goal`.

    Returns:
        2D list `matrix[y][x]` of floats (`math.inf` for obstacle/unreachable cells).
    """
    if not grid or not grid[0]:
        return []

    height = len(grid)
    width = len(grid[0])
    gx, gy = goal
    mode_key = mode.strip().lower()

    matrix: list[list[float]] = [[math.inf] * width for _ in range(height)]
    if not (0 <= gx < width and 0 <= gy < height) or grid[gy][gx] != 0:
        return matrix

    if mode_key in ("octile", "euclidean", "chebyshev", "manhattan"):
        h_fn = get_heuristic_fn(mode_key)
        for y in range(height):
            row = grid[y]
            out_row = matrix[y]
            for x in range(width):
                if row[x] == 0:
                    out_row[x] = h_fn((x, y), goal)
        return matrix

    if mode_key == "bfs":
        matrix[gy][gx] = 0.0
        queue: deque[Point] = deque([goal])
        while queue:
            x, y = queue.popleft()
            next_dist = matrix[y][x] + 1.0
            row = grid[y]
            free_right = x + 1 < width and row[x + 1] == 0
            free_left = x - 1 >= 0 and row[x - 1] == 0
            free_down = y + 1 < height and grid[y + 1][x] == 0
            free_up = y - 1 >= 0 and grid[y - 1][x] == 0

            neighbors: list[Point] = []
            if free_right:
                neighbors.append((x + 1, y))
            if free_left:
                neighbors.append((x - 1, y))
            if free_down:
                neighbors.append((x, y + 1))
            if free_up:
                neighbors.append((x, y - 1))
            if free_right and free_down and grid[y + 1][x + 1] == 0:
                neighbors.append((x + 1, y + 1))
            if free_right and free_up and grid[y - 1][x + 1] == 0:
                neighbors.append((x + 1, y - 1))
            if free_left and free_down and grid[y + 1][x - 1] == 0:
                neighbors.append((x - 1, y + 1))
            if free_left and free_up and grid[y - 1][x - 1] == 0:
                neighbors.append((x - 1, y - 1))

            for nx, ny in neighbors:
                if math.isinf(matrix[ny][nx]):
                    matrix[ny][nx] = next_dist
                    queue.append((nx, ny))
        return matrix

    if mode_key == "exact":
        matrix[gy][gx] = 0.0
        heap: list[tuple[float, int, int]] = [(0.0, gx, gy)]
        while heap:
            dist_u, x, y = heapq.heappop(heap)
            if dist_u > matrix[y][x] + EPSILON:
                continue

            row = grid[y]
            free_right = x + 1 < width and row[x + 1] == 0
            free_left = x - 1 >= 0 and row[x - 1] == 0
            free_down = y + 1 < height and grid[y + 1][x] == 0
            free_up = y - 1 >= 0 and grid[y - 1][x] == 0

            # Cardinal moves (cost 1.0)
            if free_right and dist_u + 1.0 + EPSILON < matrix[y][x + 1]:
                matrix[y][x + 1] = dist_u + 1.0
                heapq.heappush(heap, (dist_u + 1.0, x + 1, y))
            if free_left and dist_u + 1.0 + EPSILON < matrix[y][x - 1]:
                matrix[y][x - 1] = dist_u + 1.0
                heapq.heappush(heap, (dist_u + 1.0, x - 1, y))
            if free_down and dist_u + 1.0 + EPSILON < matrix[y + 1][x]:
                matrix[y + 1][x] = dist_u + 1.0
                heapq.heappush(heap, (dist_u + 1.0, x, y + 1))
            if free_up and dist_u + 1.0 + EPSILON < matrix[y - 1][x]:
                matrix[y - 1][x] = dist_u + 1.0
                heapq.heappush(heap, (dist_u + 1.0, x, y - 1))

            # Diagonal moves (cost SQRT2, no corner-cutting)
            diag_cost = dist_u + SQRT2
            if free_right and free_down and grid[y + 1][x + 1] == 0:
                if diag_cost + EPSILON < matrix[y + 1][x + 1]:
                    matrix[y + 1][x + 1] = diag_cost
                    heapq.heappush(heap, (diag_cost, x + 1, y + 1))
            if free_right and free_up and grid[y - 1][x + 1] == 0:
                if diag_cost + EPSILON < matrix[y - 1][x + 1]:
                    matrix[y - 1][x + 1] = diag_cost
                    heapq.heappush(heap, (diag_cost, x + 1, y - 1))
            if free_left and free_down and grid[y + 1][x - 1] == 0:
                if diag_cost + EPSILON < matrix[y + 1][x - 1]:
                    matrix[y + 1][x - 1] = diag_cost
                    heapq.heappush(heap, (diag_cost, x - 1, y + 1))
            if free_left and free_up and grid[y - 1][x - 1] == 0:
                if diag_cost + EPSILON < matrix[y - 1][x - 1]:
                    matrix[y - 1][x - 1] = diag_cost
                    heapq.heappush(heap, (diag_cost, x - 1, y - 1))
        return matrix

    raise ValueError(
        f"Unknown heuristic matrix mode '{mode}'. Valid options: octile, euclidean, chebyshev, manhattan, bfs, exact"
    )

