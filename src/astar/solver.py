"""8-connected A* search implementation adhering to the Project Contract."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import time
from typing import Iterator

from .heuristics import (
    EPSILON,
    SQRT2,
    Grid,
    HeuristicFn,
    Point,
    get_heuristic_fn,
)


@dataclass(frozen=True)
class AStarResult:
    """Detailed result and metrics of an A* search execution.

    Attributes:
        path: Sequence of coordinates [(x0, y0), ..., (xk, yk)] from start to goal,
            or None if no path exists.
        cost: Total path cost (0.0 if start == goal, math.inf if no path).
        expanded_nodes: Number of states popped from OPEN and expanded.
        evaluated_nodes: Number of neighbor state evaluations performed.
        generated_nodes: Number of state insertions into the OPEN priority queue.
        runtime_ms: Execution wall-clock time in milliseconds.
    """

    path: list[Point] | None
    cost: float
    expanded_nodes: int
    evaluated_nodes: int
    generated_nodes: int
    runtime_ms: float


_LAST_STATS: AStarResult | None = None


def get_last_stats() -> AStarResult | None:
    """Return the detailed `AStarResult` from the most recent `solve()` call."""
    return _LAST_STATS


def validate_inputs(
    grid: Grid,
    start: Point,
    goal: Point,
) -> tuple[bool, int, int]:
    """Validate grid geometry and start/goal coordinates without mutating `grid`.

    Returns:
        (is_valid, width, height)
    """
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list) or not grid[0]:
        return False, 0, 0

    height = len(grid)
    width = len(grid[0])
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            return False, 0, 0

    if (
        not isinstance(start, tuple)
        or not isinstance(goal, tuple)
        or len(start) != 2
        or len(goal) != 2
    ):
        return False, width, height

    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height):
        return False, width, height

    if grid[sy][sx] != 0 or grid[gy][gx] != 0:
        return False, width, height

    return True, width, height


def iter_neighbors_8(
    grid: Grid,
    x: int,
    y: int,
    width: int,
    height: int,
) -> Iterator[tuple[int, int, float]]:
    """Yield valid 8-connected neighbors `(nx, ny, step_cost)` obeying no corner-cutting."""
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


class AStarSolver:
    """Configurable 8-connected A* solver.

    Structured so that subclasses (e.g., Theta* in Week 47) can reuse the
    priority queue, closed-set management, and path reconstruction.
    """

    def __init__(
        self,
        heuristic: str | HeuristicFn = "octile",
        weight: float = 1.0,
        tie_breaking: str = "high_g",
    ) -> None:
        if weight < 0.0:
            raise ValueError("Heuristic weight must be non-negative.")
        if tie_breaking not in ("high_g", "low_g", "fifo"):
            raise ValueError("tie_breaking must be one of: 'high_g', 'low_g', 'fifo'.")
        self.heuristic_fn: HeuristicFn = get_heuristic_fn(heuristic)
        self.weight: float = float(weight)
        self.tie_breaking: str = tie_breaking

    def solve_with_stats(
        self,
        grid: Grid,
        start: Point,
        goal: Point,
    ) -> AStarResult:
        """Run A* search and return the path along with detailed search statistics."""
        t0 = time.perf_counter()

        is_valid, width, height = validate_inputs(grid, start, goal)
        if not is_valid:
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            return AStarResult(
                path=None,
                cost=math.inf,
                expanded_nodes=0,
                evaluated_nodes=0,
                generated_nodes=0,
                runtime_ms=runtime_ms,
            )

        if start == goal:
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            return AStarResult(
                path=[start],
                cost=0.0,
                expanded_nodes=0,
                evaluated_nodes=0,
                generated_nodes=1,
                runtime_ms=runtime_ms,
            )

        sx, sy = start
        gx, gy = goal
        num_cells = width * height
        start_idx = sy * width + sx
        goal_idx = gy * width + gx

        g_score: list[float] = [math.inf] * num_cells
        parent: list[int] = [-1] * num_cells
        closed = bytearray(num_cells)

        g_score[start_idx] = 0.0
        h_start = self.heuristic_fn(start, goal)
        f_start = self.weight * h_start

        # Heap entry: (f_rounded, tie_breaker, push_order, node_idx, g_val)
        # Rounding f to 12 decimal places prevents IEEE-754 addition-order drift
        # (1e-16) from overriding intentional tie-breaking between equal-cost paths.
        open_heap: list[tuple[float, float, int, int, float]] = [
            (round(f_start, 12), 0.0, 0, start_idx, 0.0)
        ]

        expanded_nodes = 0
        evaluated_nodes = 0
        generated_nodes = 1
        push_order = 0

        h_fn = self.heuristic_fn
        weight = self.weight
        tie_mode = self.tie_breaking

        while open_heap:
            _, _, _, u_idx, g_u = heapq.heappop(open_heap)

            if closed[u_idx]:
                continue
            closed[u_idx] = 1

            if u_idx == goal_idx:
                path = self._reconstruct_path(parent, start_idx, goal_idx, width)
                runtime_ms = (time.perf_counter() - t0) * 1000.0
                return AStarResult(
                    path=path,
                    cost=g_u,
                    expanded_nodes=expanded_nodes,
                    evaluated_nodes=evaluated_nodes,
                    generated_nodes=generated_nodes,
                    runtime_ms=runtime_ms,
                )

            expanded_nodes += 1
            ux = u_idx % width
            uy = u_idx // width
            row = grid[uy]

            # Inline 8-neighbor check with strict no corner-cutting for maximum speed
            free_right = ux + 1 < width and row[ux + 1] == 0
            free_left = ux - 1 >= 0 and row[ux - 1] == 0
            free_down = uy + 1 < height and grid[uy + 1][ux] == 0
            free_up = uy - 1 >= 0 and grid[uy - 1][ux] == 0

            neighbors: list[tuple[int, int, int, float]] = []
            if free_right:
                neighbors.append((ux + 1, uy, u_idx + 1, 1.0))
            if free_left:
                neighbors.append((ux - 1, uy, u_idx - 1, 1.0))
            if free_down:
                neighbors.append((ux, uy + 1, u_idx + width, 1.0))
            if free_up:
                neighbors.append((ux, uy - 1, u_idx - width, 1.0))

            if free_right and free_down and grid[uy + 1][ux + 1] == 0:
                neighbors.append((ux + 1, uy + 1, u_idx + width + 1, SQRT2))
            if free_right and free_up and grid[uy - 1][ux + 1] == 0:
                neighbors.append((ux + 1, uy - 1, u_idx - width + 1, SQRT2))
            if free_left and free_down and grid[uy + 1][ux - 1] == 0:
                neighbors.append((ux - 1, uy + 1, u_idx + width - 1, SQRT2))
            if free_left and free_up and grid[uy - 1][ux - 1] == 0:
                neighbors.append((ux - 1, uy - 1, u_idx - width - 1, SQRT2))

            for vx, vy, v_idx, step_cost in neighbors:
                if closed[v_idx]:
                    continue
                evaluated_nodes += 1
                tentative_g = g_u + step_cost
                if tentative_g + EPSILON < g_score[v_idx]:
                    g_score[v_idx] = tentative_g
                    parent[v_idx] = u_idx
                    h_v = h_fn((vx, vy), goal)
                    f_v = tentative_g + weight * h_v
                    if tie_mode == "high_g":
                        tie_val = -tentative_g
                    elif tie_mode == "low_g":
                        tie_val = tentative_g
                    else:
                        tie_val = 0.0
                    push_order += 1
                    generated_nodes += 1
                    heapq.heappush(
                        open_heap,
                        (round(f_v, 12), tie_val, push_order, v_idx, tentative_g),
                    )

        runtime_ms = (time.perf_counter() - t0) * 1000.0
        return AStarResult(
            path=None,
            cost=math.inf,
            expanded_nodes=expanded_nodes,
            evaluated_nodes=evaluated_nodes,
            generated_nodes=generated_nodes,
            runtime_ms=runtime_ms,
        )

    @staticmethod
    def _reconstruct_path(
        parent: list[int],
        start_idx: int,
        goal_idx: int,
        width: int,
    ) -> list[Point]:
        """Reconstruct coordinate list from `start` to `goal` using the parent array."""
        path: list[Point] = []
        curr = goal_idx
        while curr != -1:
            path.append((curr % width, curr // width))
            if curr == start_idx:
                break
            curr = parent[curr]
        path.reverse()
        return path


def solve_with_stats(
    grid: Grid,
    start: Point,
    goal: Point,
    *,
    heuristic: str | HeuristicFn = "octile",
    weight: float = 1.0,
    tie_breaking: str = "high_g",
) -> AStarResult:
    """Execute A* and return full `AStarResult` (path, cost, expanded/evaluated nodes, runtime)."""
    global _LAST_STATS
    solver = AStarSolver(heuristic=heuristic, weight=weight, tie_breaking=tie_breaking)
    result = solver.solve_with_stats(grid, start, goal)
    _LAST_STATS = result
    return result


def solve(
    grid: Grid,
    start: Point,
    goal: Point,
    *,
    heuristic: str | HeuristicFn = "octile",
    weight: float = 1.0,
    tie_breaking: str = "high_g",
) -> list[Point] | None:
    """Find an optimal 8-connected path from `start` to `goal` on `grid`.

    Conforms strictly to the Project Contract:
      - Input: `grid` (2D list where 0 = free, 1 = obstacle), `start` (x, y), `goal` (x, y).
      - Output: `list[tuple[int, int]] | None`.
      - Uses 8-connected movement with cardinal cost 1, diagonal cost sqrt(2),
        Octile heuristic, and strict no corner-cutting.
    """
    result = solve_with_stats(
        grid,
        start,
        goal,
        heuristic=heuristic,
        weight=weight,
        tie_breaking=tie_breaking,
    )
    return result.path


def compute_path_cost(path: list[Point] | None) -> float:
    """Compute the total Euclidean/step length of a path."""
    if path is None:
        return math.inf
    if len(path) <= 1:
        return 0.0
    total = 0.0
    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dx == 1 and dy == 1:
            total += SQRT2
        elif dx + dy == 1:
            total += 1.0
        else:
            total += math.hypot(dx, dy)
    return total


def is_valid_grid_path(
    grid: Grid,
    path: list[Point] | None,
    start: Point,
    goal: Point,
) -> bool:
    """Validate that `path` is a legal 8-connected path on `grid` without corner-cutting."""
    if path is None or not path:
        return False
    is_valid, width, height = validate_inputs(grid, start, goal)
    if not is_valid:
        return False
    if path[0] != start or path[-1] != goal:
        return False

    for x, y in path:
        if not (0 <= x < width and 0 <= y < height) or grid[y][x] != 0:
            return False

    for (x1, y1), (x2, y2) in zip(path, path[1:]):
        dx = x2 - x1
        dy = y2 - y1
        if max(abs(dx), abs(dy)) != 1 or (dx == 0 and dy == 0):
            return False
        # Diagonal move must not cut corners
        if abs(dx) == 1 and abs(dy) == 1:
            if grid[y1][x1 + dx] != 0 or grid[y1 + dy][x1] != 0:
                return False

    return True

