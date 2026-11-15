"""Load binary-grid inputs from MovingAI octile maps and version-1 scenarios.

Format reference: https://movingai.com/benchmarks/formats.html
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


class MovingAIParser:
    """Parse one map/scenario pair once, independently of pathfinding code.

    Malformed content raises ValueError with a file and line number. File access
    errors retain their standard OSError types. Coordinates use (x, y), with
    cells accessed as grid[y][x]. Swamp and water terrain are unsupported.
    """

    def __init__(self, map_path: str | Path, scen_path: str | Path) -> None:
        self._map_path = Path(map_path)
        self._scen_path = Path(scen_path)
        self._grid, self._width, self._height = self._load_map(self._map_path)
        self._scenarios = self._load_scenarios(self._scen_path)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def scenario_count(self) -> int:
        return len(self._scenarios)

    def get_input(
        self, index: int
    ) -> tuple[list[list[int]], tuple[int, int], tuple[int, int]]:
        """Return a fresh grid copy, start, and goal for a zero-based index."""
        scenario = self.get_scenario_info(index)
        return [row.copy() for row in self._grid], scenario["start"], scenario["goal"]

    def get_scenario_info(self, index: int) -> dict[str, Any]:
        """Return independent metadata, including optimal_length.

        Keys are bucket, map_name, width, height, start, goal, optimal_length.
        Noninteger indices raise TypeError; negative or missing indices raise
        IndexError. Metadata values are immutable scalars or coordinate tuples.
        """
        if type(index) is not int:
            raise TypeError("scenario index must be an integer")
        if not 0 <= index < self.scenario_count:
            raise IndexError(
                f"{self._scen_path}: scenario index {index} is out of range "
                f"for {self.scenario_count} records (zero-based)"
            )
        return self._scenarios[index].copy()

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        try:
            return path.read_text(encoding="utf-8-sig").splitlines()
        except UnicodeDecodeError as exc:
            line = exc.object[: exc.start].count(b"\n") + 1
            raise ValueError(f"{path}:{line}: invalid UTF-8 text") from exc

    @staticmethod
    def _load_map(path: Path) -> tuple[list[list[int]], int, int]:
        lines = MovingAIParser._read_lines(path)
        if len(lines) < 4:
            raise ValueError(f"{path}:{len(lines) + 1}: incomplete map header")
        if lines[0].split() != ["type", "octile"]:
            raise ValueError(f"{path}:1: expected 'type octile'")

        dimensions = {}
        for line_number, name in ((2, "height"), (3, "width")):
            fields = lines[line_number - 1].split()
            if len(fields) != 2 or fields[0] != name:
                raise ValueError(f"{path}:{line_number}: expected '{name} <integer>'")
            try:
                value = int(fields[1])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: {name} must be a positive integer"
                ) from exc
            if value <= 0:
                raise ValueError(f"{path}:{line_number}: {name} must be positive")
            dimensions[name] = value

        if lines[3].strip() != "map":
            raise ValueError(f"{path}:4: expected 'map'")
        width, height = dimensions["width"], dimensions["height"]
        rows = lines[4:]
        if len(rows) != height:
            line_number = 5 + min(len(rows), height)
            raise ValueError(
                f"{path}:{line_number}: expected {height} grid rows, got {len(rows)}"
            )

        terrain = {".": 0, "G": 0, "@": 1, "O": 1, "T": 1}
        grid = []
        for line_number, row in enumerate(rows, 5):
            if len(row) != width:
                raise ValueError(
                    f"{path}:{line_number}: expected {width} cells, got {len(row)}"
                )
            for x, symbol in enumerate(row):
                if symbol not in terrain:
                    raise ValueError(
                        f"{path}:{line_number}: unsupported terrain {symbol!r} "
                        f"at (x={x}, y={line_number - 5})"
                    )
            grid.append([terrain[symbol] for symbol in row])
        return grid, width, height

    def _load_scenarios(self, path: Path) -> list[dict[str, Any]]:
        lines = self._read_lines(path)
        if not lines or lines[0].split() not in (["version", "1"], ["version", "1.0"]):
            raise ValueError(f"{path}:1: expected 'version 1' or 'version 1.0'")

        scenarios = []
        for line_number, line in enumerate(lines[1:], 2):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 9:
                raise ValueError(
                    f"{path}:{line_number}: expected 9 scenario fields, got {len(fields)}"
                )
            try:
                bucket = int(fields[0])
                width, height, sx, sy, gx, gy = map(int, fields[2:8])
                optimal_length = float(fields[8])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: bucket, dimensions, and coordinates must "
                    "be integers; optimal_length must be numeric"
                ) from exc

            map_name = fields[1]
            filename = map_name.replace("\\", "/").rsplit("/", 1)[-1]
            if filename != self._map_path.name:
                raise ValueError(
                    f"{path}:{line_number}: referenced map {map_name!r} does not "
                    f"match loaded map {self._map_path.name!r}"
                )
            if (width, height) != (self.width, self.height):
                raise ValueError(
                    f"{path}:{line_number}: scenario dimensions {width} x {height} "
                    f"do not match map dimensions {self.width} x {self.height}"
                )
            if bucket < 0:
                raise ValueError(f"{path}:{line_number}: bucket must be nonnegative")
            if not math.isfinite(optimal_length) or optimal_length < 0:
                raise ValueError(
                    f"{path}:{line_number}: optimal_length must be finite and nonnegative"
                )
            start, goal = (sx, sy), (gx, gy)
            for label, (x, y) in (("start", start), ("goal", goal)):
                if not (0 <= x < self.width and 0 <= y < self.height):
                    raise ValueError(
                        f"{path}:{line_number}: {label} {(x, y)} is outside the map"
                    )
                if self._grid[y][x] != 0:
                    raise ValueError(
                        f"{path}:{line_number}: {label} {(x, y)} is on a blocked cell"
                    )
            scenarios.append(
                {
                    "bucket": bucket,
                    "map_name": map_name,
                    "width": width,
                    "height": height,
                    "start": start,
                    "goal": goal,
                    "optimal_length": optimal_length,
                }
            )
        return scenarios
