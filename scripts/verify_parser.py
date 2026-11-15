"""Verify MovingAI parser output without invoking a pathfinding algorithm.

Run from the repository root with python -m scripts.verify_parser --help.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.common.parser import MovingAIParser


DATASET_NAMES = {
    "random": "random-32-32-20",
    "maze": "maze-32-32-2",
}
DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "movingai"


def main() -> int:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--map", choices=DATASET_NAMES, required=True)
    arguments.add_argument("--scen", type=int, choices=range(1, 26), required=True)
    arguments.add_argument("--index", type=int, default=0, help="Zero-based scenario index")
    args = arguments.parse_args()
    if args.index < 0:
        arguments.error("--index must be nonnegative")

    dataset = DATASET_NAMES[args.map]
    map_path = DATA_ROOT / args.map / "map" / f"{dataset}.map"
    scen_path = (
        DATA_ROOT / args.map / "scenarios" / f"{dataset}-random-{args.scen}.scen"
    )

    try:
        parser = MovingAIParser(map_path, scen_path)
        grid, start, goal = parser.get_input(args.index)
        info = parser.get_scenario_info(args.index)

        if not isinstance(grid, list) or len(grid) != parser.height:
            raise ValueError("grid must be a list with the declared number of rows")
        for y, row in enumerate(grid):
            if not isinstance(row, list) or len(row) != parser.width:
                raise ValueError(f"grid row {y} must be a list with the declared width")
            if any(type(cell) is not int or cell not in (0, 1) for cell in row):
                raise ValueError(f"grid row {y} must contain only integer 0/1 cells")
        for label, point in (("start", start), ("goal", goal)):
            if (
                not isinstance(point, tuple)
                or len(point) != 2
                or any(type(value) is not int for value in point)
            ):
                raise TypeError(f"{label} must be a tuple of two integers")
            x, y = point
            if not (0 <= x < parser.width and 0 <= y < parser.height):
                raise ValueError(f"{label} {point} is outside the grid")
            if grid[y][x] != 0:
                raise ValueError(f"{label} {point} is on a blocked cell")
    except (OSError, ValueError, IndexError, TypeError) as exc:
        print(f"Validation: FAILED\nError: {exc}", file=sys.stderr)
        return 1

    free_cells = sum(cell == 0 for row in grid for cell in row)
    blocked_cells = sum(cell == 1 for row in grid for cell in row)
    print("=== MovingAI Parser Verification ===\n")
    print(f"Dataset: {args.map}")
    print(f"Map: {map_path.name}")
    print(f"Dimensions: {parser.width} x {parser.height}")
    print(f"Free cells: {free_cells}")
    print(f"Blocked cells: {blocked_cells}\n")
    print(f"Scenario file number: {args.scen}")
    print(f"Scenario file: {scen_path.name}")
    print(f"Scenario count: {parser.scenario_count}\n")
    print(f"Selected scenario: {args.index}")
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Optimal grid length: {info['optimal_length']}\n")
    print("Input contract:")
    print("grid: list[list[int]]")
    print("start: tuple[int, int]")
    print("goal: tuple[int, int]\n")
    print("Validation: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
