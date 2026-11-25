"""8-connected A* search using custom 1-based Binary Min-Heap structures."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

from .heap import (
    HEAP_MODES,
    MAX_TIE_INT,
    PRIORITY_SCALE,
    BinaryMinHeap,
    FastIntMinHeap,
    IndexedMinHeap,
    compute_packing_params,
    pack_priority_key,
)
from .heuristics import (
    CONSISTENT_HEURISTICS,
    EPSILON,
    SQRT2,
    Grid,
    HeuristicFn,
    Point,
    get_heuristic_fn,
    iter_neighbors_8,
    octile_distance,
)


@dataclass(frozen=True)
class AStarResult:
    """Detailed result and metrics of an A* search execution."""

    path: list[Point] | None
    cost: float
    expanded_nodes: int
    evaluated_nodes: int
    generated_nodes: int
    runtime_ms: float
    peak_open_size: int = 0
    stale_pops: int = 0
    heap_ops: int = 0


_STATE = threading.local()


def get_last_stats() -> AStarResult | None:
    """Return the thread-local `AStarResult` from the most recent `solve()` call."""
    return getattr(_STATE, "last_stats", None)


def validate_inputs(
    grid: Grid,
    start: Point,
    goal: Point,
) -> tuple[bool, int, int]:
    """Validate grid geometry and start/goal coordinates without mutating `grid`."""
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
    if type(sx) is not int or type(sy) is not int or type(gx) is not int or type(gy) is not int:
        return False, width, height

    if not (0 <= sx < width and 0 <= sy < height and 0 <= gx < width and 0 <= gy < height):
        return False, width, height

    if grid[sy][sx] != 0 or grid[gy][gx] != 0:
        return False, width, height

    return True, width, height


def _reconstruct_path(
    parent: list[int],
    start_idx: int,
    goal_idx: int,
    width: int,
) -> list[Point]:
    """Reconstruct coordinate list from `start` to `goal` using the 1D parent array."""
    path: list[Point] = []
    curr = goal_idx
    while curr != -1:
        path.append((curr % width, curr // width))
        if curr == start_idx:
            break
        curr = parent[curr]
    path.reverse()
    return path


def _is_consistent_default(heuristic: str | HeuristicFn, h_fn: HeuristicFn) -> bool:
    """Return True if the heuristic is mathematically guaranteed consistent on 8-grids."""
    if isinstance(heuristic, str):
        return heuristic.strip().lower() in CONSISTENT_HEURISTICS
    return h_fn is octile_distance


def _run_astar_fast(
    grid: Grid,
    start: Point,
    goal: Point,
    width: int,
    height: int,
    h_fn: HeuristicFn,
    weight: float,
    tie_breaking: str,
    use_plateau_bypass: bool,
    t0: float,
) -> AStarResult:
    """Execute A* using 1-based `FastIntMinHeap` + optional delta-f=0 plateau stack."""
    sx, sy = start
    gx, gy = goal
    num_cells = width * height
    start_idx = sy * width + sx
    goal_idx = gy * width + gx

    g_score: list[float] = [math.inf] * num_cells
    parent: list[int] = [-1] * num_cells
    closed = bytearray(num_cells)

    idx_bits, idx_mask, f_shift = compute_packing_params(num_cells)
    g_score[start_idx] = 0.0
    f_start = weight * h_fn(start, goal)
    cur_f_int = int(f_start * PRIORITY_SCALE + 0.5)

    open_heap = FastIntMinHeap()
    open_heap.push(pack_priority_key(cur_f_int, 0.0, 0, start_idx, tie_breaking, idx_bits, f_shift))
    hot_stack: list[int] = []

    heap_push = open_heap.push
    heap_pop = open_heap.pop
    is_octile = h_fn is octile_distance and weight == 1.0

    expanded_nodes = 0
    evaluated_nodes = 0
    generated_nodes = 1
    push_order = 0
    peak_open = 1
    stale_pops = 0

    while hot_stack or open_heap:
        open_len = len(hot_stack) + len(open_heap)
        if open_len > peak_open:
            peak_open = open_len

        if hot_stack:
            u_idx = hot_stack.pop()
        else:
            entry = heap_pop()
            u_idx = entry & idx_mask
            cur_f_int = entry >> f_shift

        if closed[u_idx]:
            stale_pops += 1
            continue
        closed[u_idx] = 1

        if u_idx == goal_idx:
            path = _reconstruct_path(parent, start_idx, goal_idx, width)
            return AStarResult(
                path=path,
                cost=g_score[u_idx],
                expanded_nodes=expanded_nodes,
                evaluated_nodes=evaluated_nodes,
                generated_nodes=generated_nodes,
                runtime_ms=(time.perf_counter() - t0) * 1000.0,
                peak_open_size=peak_open,
                stale_pops=stale_pops,
                heap_ops=open_heap.heap_ops,
            )

        expanded_nodes += 1
        g_u = g_score[u_idx]
        ux = u_idx % width
        uy = u_idx // width

        for vx, vy, step_cost in iter_neighbors_8(grid, ux, uy, width, height):
            v_idx = vy * width + vx
            if closed[v_idx]:
                continue
            evaluated_nodes += 1
            tentative_g = g_u + step_cost
            if tentative_g + EPSILON < g_score[v_idx]:
                g_score[v_idx] = tentative_g
                parent[v_idx] = u_idx
                generated_nodes += 1

                if is_octile:
                    dx = abs(vx - gx)
                    dy = abs(vy - gy)
                    h_v = (dx - dy) + SQRT2 * dy if dx >= dy else (dy - dx) + SQRT2 * dx
                    f_v_int = int((tentative_g + h_v) * PRIORITY_SCALE + 0.5)
                else:
                    f_v = tentative_g + weight * h_fn((vx, vy), goal)
                    f_v_int = int(f_v * PRIORITY_SCALE + 0.5)

                if use_plateau_bypass and f_v_int <= cur_f_int + 1:
                    hot_stack.append(v_idx)
                else:
                    push_order += 1
                    if use_plateau_bypass:
                        g_scaled = int(tentative_g * PRIORITY_SCALE + 0.5)
                        tie_int = MAX_TIE_INT - g_scaled if g_scaled < MAX_TIE_INT else 0
                        key = (f_v_int << f_shift) | (tie_int << idx_bits) | v_idx
                    else:
                        key = pack_priority_key(
                            f_v_int,
                            tentative_g,
                            push_order,
                            v_idx,
                            tie_breaking,
                            idx_bits,
                            f_shift,
                        )
                    heap_push(key)

    return AStarResult(
        path=None,
        cost=math.inf,
        expanded_nodes=expanded_nodes,
        evaluated_nodes=evaluated_nodes,
        generated_nodes=generated_nodes,
        runtime_ms=(time.perf_counter() - t0) * 1000.0,
        peak_open_size=peak_open,
        stale_pops=stale_pops,
        heap_ops=open_heap.heap_ops,
    )


def _run_astar_decrease_key(
    grid: Grid,
    start: Point,
    goal: Point,
    width: int,
    height: int,
    h_fn: HeuristicFn,
    weight: float,
    tie_breaking: str,
    t0: float,
) -> AStarResult:
    """Execute A* using 1-based `IndexedMinHeap` with in-place O(log |OPEN|) `decrease_key`."""
    sx, sy = start
    gx, gy = goal
    num_cells = width * height
    start_idx = sy * width + sx
    goal_idx = gy * width + gx

    g_score: list[float] = [math.inf] * num_cells
    parent: list[int] = [-1] * num_cells
    closed = bytearray(num_cells)

    idx_bits, _, f_shift = compute_packing_params(num_cells)
    open_heap = IndexedMinHeap(num_cells)
    is_octile = h_fn is octile_distance and weight == 1.0

    g_score[start_idx] = 0.0
    f_start_int = int((weight * h_fn(start, goal)) * PRIORITY_SCALE + 0.5)
    start_key = pack_priority_key(f_start_int, 0.0, 0, start_idx, tie_breaking, idx_bits, f_shift)
    open_heap.push_or_decrease(start_idx, start_key)

    expanded_nodes = 0
    evaluated_nodes = 0
    generated_nodes = 1
    push_order = 0
    peak_open = 1

    while open_heap:
        if len(open_heap) > peak_open:
            peak_open = len(open_heap)

        u_idx, _ = open_heap.pop_min()
        closed[u_idx] = 1

        if u_idx == goal_idx:
            path = _reconstruct_path(parent, start_idx, goal_idx, width)
            return AStarResult(
                path=path,
                cost=g_score[u_idx],
                expanded_nodes=expanded_nodes,
                evaluated_nodes=evaluated_nodes,
                generated_nodes=generated_nodes,
                runtime_ms=(time.perf_counter() - t0) * 1000.0,
                peak_open_size=peak_open,
                stale_pops=0,
                heap_ops=open_heap.heap_ops,
            )

        expanded_nodes += 1
        g_u = g_score[u_idx]
        ux = u_idx % width
        uy = u_idx // width

        for vx, vy, step_cost in iter_neighbors_8(grid, ux, uy, width, height):
            v_idx = vy * width + vx
            if closed[v_idx]:
                continue
            evaluated_nodes += 1
            tentative_g = g_u + step_cost
            if tentative_g + EPSILON < g_score[v_idx]:
                g_score[v_idx] = tentative_g
                parent[v_idx] = u_idx
                push_order += 1
                generated_nodes += 1
                if is_octile:
                    dx = abs(vx - gx)
                    dy = abs(vy - gy)
                    h_v = (dx - dy) + SQRT2 * dy if dx >= dy else (dy - dx) + SQRT2 * dx
                    f_v_int = int((tentative_g + h_v) * PRIORITY_SCALE + 0.5)
                else:
                    f_v_int = int((tentative_g + weight * h_fn((vx, vy), goal)) * PRIORITY_SCALE + 0.5)
                key = pack_priority_key(
                    f_v_int,
                    tentative_g,
                    push_order,
                    v_idx,
                    tie_breaking,
                    idx_bits,
                    f_shift,
                )
                open_heap.push_or_decrease(v_idx, key)

    return AStarResult(
        path=None,
        cost=math.inf,
        expanded_nodes=expanded_nodes,
        evaluated_nodes=evaluated_nodes,
        generated_nodes=generated_nodes,
        runtime_ms=(time.perf_counter() - t0) * 1000.0,
        peak_open_size=peak_open,
        stale_pops=0,
        heap_ops=open_heap.heap_ops,
    )


def _run_astar_lazy_tuple(
    grid: Grid,
    start: Point,
    goal: Point,
    width: int,
    height: int,
    h_fn: HeuristicFn,
    weight: float,
    tie_breaking: str,
    t0: float,
) -> AStarResult:
    """Baseline A* using textbook 1-based `BinaryMinHeap` with 5-tuples & lazy deletion."""
    sx, sy = start
    gx, gy = goal
    num_cells = width * height
    start_idx = sy * width + sx
    goal_idx = gy * width + gx

    g_score: list[float] = [math.inf] * num_cells
    parent: list[int] = [-1] * num_cells
    closed = bytearray(num_cells)

    g_score[start_idx] = 0.0
    f_start = weight * h_fn(start, goal)
    open_heap: BinaryMinHeap[tuple[float, float, int, int, float]] = BinaryMinHeap()
    open_heap.push((round(f_start, 12), 0.0, 0, start_idx, 0.0))

    expanded_nodes = 0
    evaluated_nodes = 0
    generated_nodes = 1
    push_order = 0
    peak_open = 1
    stale_pops = 0

    while open_heap:
        if len(open_heap) > peak_open:
            peak_open = len(open_heap)

        _, _, _, u_idx, g_u = open_heap.pop()

        if closed[u_idx]:
            stale_pops += 1
            continue
        closed[u_idx] = 1

        if u_idx == goal_idx:
            path = _reconstruct_path(parent, start_idx, goal_idx, width)
            return AStarResult(
                path=path,
                cost=g_u,
                expanded_nodes=expanded_nodes,
                evaluated_nodes=evaluated_nodes,
                generated_nodes=generated_nodes,
                runtime_ms=(time.perf_counter() - t0) * 1000.0,
                peak_open_size=peak_open,
                stale_pops=stale_pops,
                heap_ops=open_heap.heap_ops,
            )

        expanded_nodes += 1
        ux = u_idx % width
        uy = u_idx // width

        for vx, vy, step_cost in iter_neighbors_8(grid, ux, uy, width, height):
            v_idx = vy * width + vx
            if closed[v_idx]:
                continue
            evaluated_nodes += 1
            tentative_g = g_u + step_cost
            if tentative_g + EPSILON < g_score[v_idx]:
                g_score[v_idx] = tentative_g
                parent[v_idx] = u_idx
                f_v = tentative_g + weight * h_fn((vx, vy), goal)
                if tie_breaking == "high_g":
                    tie_val = -tentative_g
                elif tie_breaking == "low_g":
                    tie_val = tentative_g
                else:
                    tie_val = 0.0
                push_order += 1
                generated_nodes += 1
                open_heap.push((round(f_v, 12), tie_val, push_order, v_idx, tentative_g))

    return AStarResult(
        path=None,
        cost=math.inf,
        expanded_nodes=expanded_nodes,
        evaluated_nodes=evaluated_nodes,
        generated_nodes=generated_nodes,
        runtime_ms=(time.perf_counter() - t0) * 1000.0,
        peak_open_size=peak_open,
        stale_pops=stale_pops,
        heap_ops=open_heap.heap_ops,
    )


class AStarSolver:
    """Configurable 8-connected A* solver wrapper (compatible with integration benchmarks)."""

    def __init__(
        self,
        heuristic: str | HeuristicFn = "octile",
        weight: float = 1.0,
        tie_breaking: str = "high_g",
        heap_mode: str = "fast",
    ) -> None:
        if weight < 0.0:
            raise ValueError("Heuristic weight must be non-negative.")
        if tie_breaking not in ("high_g", "low_g", "fifo"):
            raise ValueError("tie_breaking must be one of: 'high_g', 'low_g', 'fifo'.")
        if heap_mode not in HEAP_MODES:
            raise ValueError(f"heap_mode must be one of: {HEAP_MODES}.")
        self._heuristic_spec: str | HeuristicFn = heuristic
        self.heuristic_fn: HeuristicFn = get_heuristic_fn(heuristic)
        self.weight: float = float(weight)
        self.tie_breaking: str = tie_breaking
        self.heap_mode: str = heap_mode

    def solve_with_stats(self, grid: Grid, start: Point, goal: Point) -> AStarResult:
        """Run A* search and return full `AStarResult`."""
        return solve_with_stats(
            grid,
            start,
            goal,
            heuristic=self._heuristic_spec,
            weight=self.weight,
            tie_breaking=self.tie_breaking,
            heap_mode=self.heap_mode,
        )

    def solve(self, grid: Grid, start: Point, goal: Point) -> list[Point] | None:
        """Run A* search and return path or None."""
        return self.solve_with_stats(grid, start, goal).path


def solve_with_stats(
    grid: Grid,
    start: Point,
    goal: Point,
    *,
    heuristic: str | HeuristicFn = "octile",
    weight: float = 1.0,
    tie_breaking: str = "high_g",
    heap_mode: str = "fast",
) -> AStarResult:
    """Execute 8-connected A* and return `AStarResult` with search & heap metrics."""
    t0 = time.perf_counter()

    if weight < 0.0:
        raise ValueError("Heuristic weight must be non-negative.")
    if tie_breaking not in ("high_g", "low_g", "fifo"):
        raise ValueError("tie_breaking must be one of: 'high_g', 'low_g', 'fifo'.")
    if heap_mode not in HEAP_MODES:
        raise ValueError(f"heap_mode must be one of: {HEAP_MODES}.")
    h_fn = get_heuristic_fn(heuristic)

    is_valid, width, height = validate_inputs(grid, start, goal)
    if not is_valid:
        res = AStarResult(
            path=None,
            cost=math.inf,
            expanded_nodes=0,
            evaluated_nodes=0,
            generated_nodes=0,
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
            peak_open_size=0,
            stale_pops=0,
            heap_ops=0,
        )
        _STATE.last_stats = res
        return res

    if start == goal:
        res = AStarResult(
            path=[start],
            cost=0.0,
            expanded_nodes=0,
            evaluated_nodes=0,
            generated_nodes=1,
            runtime_ms=(time.perf_counter() - t0) * 1000.0,
            peak_open_size=1,
            stale_pops=0,
            heap_ops=0,
        )
        _STATE.last_stats = res
        return res

    if heap_mode == "fast":
        use_plateau = (
            weight == 1.0
            and tie_breaking == "high_g"
            and _is_consistent_default(heuristic, h_fn)
        )
        res = _run_astar_fast(
            grid,
            start,
            goal,
            width,
            height,
            h_fn,
            weight,
            tie_breaking,
            use_plateau,
            t0,
        )
    elif heap_mode == "decrease_key":
        res = _run_astar_decrease_key(
            grid, start, goal, width, height, h_fn, weight, tie_breaking, t0
        )
    else:
        res = _run_astar_lazy_tuple(
            grid, start, goal, width, height, h_fn, weight, tie_breaking, t0
        )

    _STATE.last_stats = res
    return res


def solve(
    grid: Grid,
    start: Point,
    goal: Point,
    *,
    heuristic: str | HeuristicFn = "octile",
    weight: float = 1.0,
    tie_breaking: str = "high_g",
    heap_mode: str = "fast",
) -> list[Point] | None:
    """Find an optimal 8-connected path from `start` to `goal` on `grid`."""
    return solve_with_stats(
        grid,
        start,
        goal,
        heuristic=heuristic,
        weight=weight,
        tie_breaking=tie_breaking,
        heap_mode=heap_mode,
    ).path


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
        if abs(dx) == 1 and abs(dy) == 1:
            if grid[y1][x1 + dx] != 0 or grid[y1 + dy][x1] != 0:
                return False

    return True
