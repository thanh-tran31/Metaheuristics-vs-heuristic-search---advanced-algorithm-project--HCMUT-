"""A* (8-connected, Octile heuristic) pathfinding package."""

from .heuristics import (
    SQRT2,
    chebyshev_distance,
    check_admissibility_and_consistency,
    compute_heuristic_matrix,
    euclidean_distance,
    get_heuristic_fn,
    manhattan_distance,
    octile_distance,
    zero_heuristic,
)
from .solver import (
    AStarResult,
    AStarSolver,
    compute_path_cost,
    get_last_stats,
    is_valid_grid_path,
    iter_neighbors_8,
    solve,
    solve_with_stats,
    validate_inputs,
)

__all__ = [
    "SQRT2",
    "AStarResult",
    "AStarSolver",
    "chebyshev_distance",
    "check_admissibility_and_consistency",
    "compute_heuristic_matrix",
    "compute_path_cost",
    "euclidean_distance",
    "get_heuristic_fn",
    "get_last_stats",
    "is_valid_grid_path",
    "iter_neighbors_8",
    "manhattan_distance",
    "octile_distance",
    "solve",
    "solve_with_stats",
    "validate_inputs",
    "zero_heuristic",
]

