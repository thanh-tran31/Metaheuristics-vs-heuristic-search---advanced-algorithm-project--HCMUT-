"""Heuristic functions, admissibility/consistency analysis, and distance matrices."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Callable, Iterator

from .heap import IndexedMinHeap, PRIORITY_SCALE

Point = tuple[int, int]
Grid = list[list[int]]
HeuristicFn = Callable[[Point, Point], float]

SQRT2: float = math.sqrt(2.0)
EPSILON: float = 1e-9

CONSISTENT_HEURISTICS: frozenset[str] = frozenset(
    {"octile", "euclidean", "chebyshev", "zero", "dijkstra"}
)


@dataclass(frozen=True)
class HeuristicAnalysisReport:
    """Quantitative verification report for admissibility and consistency on a grid."""

    heuristic_name: str
    is_admissible: bool
    is_consistent: bool
    reachable_cells: int
    checked_edges: int
    admissibility_violations: int
    consistency_violations: int
    max_h_overestimate: float
    max_triangle_violation: float
    mean_tightness_ratio: float
    sample_violation: tuple[Point, Point | None, float] | None = None


def iter_neighbors_8(
    grid: Grid,
    x: int,
    y: int,
    width: int,
    height: int,
) -> Iterator[tuple[int, int, float]]:
    """Yield valid 8-connected neighbors `(nx, ny, step_cost)` obeying no corner-cutting.

    Cardinal moves (cost 1.0) are yielded before diagonal moves (cost sqrt(2)).
    """
    row = grid[y]
    free_right = x + 1 < width and row[x + 1] == 0
    free_left = x - 1 >= 0 and row[x - 1] == 0
    free_down = y + 1 < height and grid[y + 1][x] == 0
    free_up = y - 1 >= 0 and grid[y - 1][x] == 0

    if free_right:
        yield x + 1, y, 1.0
    if free_left:
        yield x - 1, y, 1.0
    if free_down:
        yield x, y + 1, 1.0
    if free_up:
        yield x, y - 1, 1.0

    if free_right and free_down and grid[y + 1][x + 1] == 0:
        yield x + 1, y + 1, SQRT2
    if free_right and free_up and grid[y - 1][x + 1] == 0:
        yield x + 1, y - 1, SQRT2
    if free_left and free_down and grid[y + 1][x - 1] == 0:
        yield x - 1, y + 1, SQRT2
    if free_left and free_up and grid[y - 1][x - 1] == 0:
        yield x - 1, y - 1, SQRT2


def octile_distance(a: Point, b: Point) -> float:
    """Compute the Octile distance between two grid coordinates (x, y)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if dx >= dy:
        return (dx - dy) + SQRT2 * dy
    return (dy - dx) + SQRT2 * dx


def euclidean_distance(a: Point, b: Point) -> float:
    """Compute the Euclidean (L2) distance between two coordinates (x, y)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chebyshev_distance(a: Point, b: Point) -> float:
    """Compute the Chebyshev (L-infinity) distance between two coordinates (x, y)."""
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


def manhattan_distance(a: Point, b: Point) -> float:
    """Compute the Manhattan (L1) distance between two coordinates (x, y)."""
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


def analyze_heuristic_properties(
    grid: Grid,
    goal: Point,
    heuristic: str | HeuristicFn = "octile",
) -> HeuristicAnalysisReport:
    """Exhaustively analyze admissibility, consistency, and tightness on `grid`."""
    h_fn = get_heuristic_fn(heuristic)
    name = heuristic if isinstance(heuristic, str) else getattr(heuristic, "__name__", "custom")
    exact_dist = compute_heuristic_matrix(grid, goal, mode="exact")
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0

    reachable_cells = 0
    checked_edges = 0
    admissibility_violations = 0
    consistency_violations = 0
    max_overestimate = 0.0
    max_tri_violation = 0.0
    tightness_sum = 0.0
    tightness_count = 0
    sample_violation: tuple[Point, Point | None, float] | None = None

    gx, gy = goal
    if 0 <= gx < width and 0 <= gy < height and grid[gy][gx] == 0:
        goal_h = abs(h_fn(goal, goal))
        if goal_h > EPSILON:
            consistency_violations += 1
            max_tri_violation = max(max_tri_violation, goal_h)
            sample_violation = (goal, None, goal_h)

    for y in range(height):
        row = grid[y]
        for x in range(width):
            if row[x] != 0:
                continue
            u = (x, y)
            h_u = h_fn(u, goal)
            h_star_u = exact_dist[y][x]
            if math.isfinite(h_star_u):
                reachable_cells += 1
                over = h_u - h_star_u
                if over > max_overestimate:
                    max_overestimate = over
                if over > EPSILON:
                    admissibility_violations += 1
                    if sample_violation is None:
                        sample_violation = (u, None, over)
                if h_star_u > EPSILON:
                    tightness_sum += h_u / h_star_u
                    tightness_count += 1

            for nx, ny, step_cost in iter_neighbors_8(grid, x, y, width, height):
                checked_edges += 1
                v = (nx, ny)
                tri_diff = h_u - (step_cost + h_fn(v, goal))
                if tri_diff > max_tri_violation:
                    max_tri_violation = tri_diff
                if tri_diff > EPSILON:
                    consistency_violations += 1
                    if sample_violation is None:
                        sample_violation = (u, v, tri_diff)

    mean_tightness = tightness_sum / tightness_count if tightness_count > 0 else 1.0
    return HeuristicAnalysisReport(
        heuristic_name=name,
        is_admissible=(admissibility_violations == 0),
        is_consistent=(consistency_violations == 0),
        reachable_cells=reachable_cells,
        checked_edges=checked_edges,
        admissibility_violations=admissibility_violations,
        consistency_violations=consistency_violations,
        max_h_overestimate=max_overestimate,
        max_triangle_violation=max_tri_violation,
        mean_tightness_ratio=mean_tightness,
        sample_violation=sample_violation,
    )


def check_admissibility_and_consistency(
    grid: Grid,
    goal: Point,
    heuristic: str | HeuristicFn = "octile",
) -> tuple[bool, bool]:
    """Empirically verify admissibility and consistency of a heuristic on a grid."""
    report = analyze_heuristic_properties(grid, goal, heuristic=heuristic)
    return report.is_admissible, report.is_consistent


def compute_heuristic_matrix(
    grid: Grid,
    goal: Point,
    mode: str = "octile",
) -> list[list[float]]:
    """Precompute a 2D heuristic matrix H[y][x] towards `goal` using 1-based Heap."""
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
            for nx, ny, _ in iter_neighbors_8(grid, x, y, width, height):
                if math.isinf(matrix[ny][nx]):
                    matrix[ny][nx] = next_dist
                    queue.append((nx, ny))
        return matrix

    if mode_key == "exact":
        num_cells = width * height
        heap = IndexedMinHeap(num_cells)
        matrix[gy][gx] = 0.0
        heap.push_or_decrease(gy * width + gx, 0)
        while heap:
            u_idx, _ = heap.pop_min()
            x = u_idx % width
            y = u_idx // width
            dist_u = matrix[y][x]
            for nx, ny, step_cost in iter_neighbors_8(grid, x, y, width, height):
                next_dist = dist_u + step_cost
                if next_dist + EPSILON < matrix[ny][nx]:
                    matrix[ny][nx] = next_dist
                    heap.push_or_decrease(ny * width + nx, int(next_dist * PRIORITY_SCALE + 0.5))
        return matrix

    raise ValueError(
        f"Unknown heuristic matrix mode '{mode}'. Valid options: "
        "octile, euclidean, chebyshev, manhattan, bfs, exact"
    )
