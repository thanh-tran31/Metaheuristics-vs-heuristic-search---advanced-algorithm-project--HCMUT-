"""Independent path checks for cell centers on rectangular binary grids."""
from dataclasses import dataclass
from math import dist
from collections.abc import Sequence


@dataclass
class ValidationResult:
    status: str  # valid, invalid, or none
    path_length: float | None
    errors: list[str]

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"


def calculate_path_length(path) -> float | None:
    """Sum Euclidean segment lengths for a checked path, or None for no path."""
    if not path:
        return None
    return sum((dist(a, b) for a, b in zip(path, path[1:])), 0.0)


def _visible(grid, a, b):
    """Exact segment vs closed obstacle squares by separating-axis projection."""
    x, y = a
    dx, dy = b[0] - x, b[1] - y
    # The bounding box already checks the two square axes. On the remaining
    # axis (normal to the segment), a square's projected half-width is
    # (abs(dx) + abs(dy)) / 2. Equality is a forbidden corner contact.
    for cy in range(min(y, b[1]), max(y, b[1]) + 1):
        for cx in range(min(x, b[0]), max(x, b[0]) + 1):
            if grid[cy][cx]:
                cross = dx * (cy - y) - dy * (cx - x)
                if 2 * abs(cross) <= abs(dx) + abs(dy):
                    return False
    return True


def validate_path(grid, start, goal, path, *, algorithm: str) -> ValidationResult:
    """Check the input, endpoints, and each segment. Only valid paths get a length."""
    def invalid(message):
        return ValidationResult("invalid", None, [message])

    if algorithm not in ("astar", "aco", "theta_star"):
        raise ValueError(f"Unknown algorithm {algorithm!r}")
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list) or not grid[0]:
        return invalid("Expected a nonempty binary grid")
    width = len(grid[0])
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            return invalid("Grid must have equal-width rows")
        if any(type(v) is not int or v not in (0, 1) for v in row):
            return invalid("Grid must contain integer 0/1 cells")

    def free_point(p):
        return (isinstance(p, tuple) and len(p) == 2 and all(type(v) is int for v in p)
                and 0 <= p[0] < width and 0 <= p[1] < len(grid) and grid[p[1]][p[0]] == 0)

    if not free_point(start) or not free_point(goal):
        return invalid("Invalid start or goal")
    if path is None:
        return ValidationResult("none", None, [])
    if not isinstance(path, Sequence) or isinstance(path, (str, bytes)) or not path:
        return invalid("Expected a nonempty path")
    if path[0] != start or path[-1] != goal:
        return invalid("Path endpoints differ from start or goal")
    if start == goal and len(path) != 1:
        return invalid("Equal endpoints require a singleton path")
    for i, point in enumerate(path):
        if not free_point(point):
            return invalid(f"Invalid or blocked waypoint {i}: {point!r}")
    for a, b in zip(path, path[1:]):
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if a == b:
            return invalid(f"Repeated consecutive waypoint {a}")
        if algorithm == "theta_star":
            if not _visible(grid, a, b):
                return invalid(f"Obstructed segment {a} to {b}")
        elif max(dx, dy) != 1 or (dx and dy and (grid[a[1]][b[0]] or grid[b[1]][a[0]])):
            return invalid(f"Illegal grid step {a} to {b}")
    return ValidationResult("valid", calculate_path_length(path), [])
