"""Ant colony optimization with per-instance configuration."""
from collections import Counter
from math import isfinite, sqrt
from random import Random

DIAG = sqrt(2)


def validate_input(grid, start, goal):
    if not isinstance(grid, list) or not grid or not isinstance(grid[0], list) or not grid[0]:
        raise ValueError("grid must be a non-empty list of non-empty lists")
    width = len(grid[0])
    for row in grid:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("grid must be rectangular")
        if any(type(cell) is not int or cell not in (0, 1) for cell in row):
            raise ValueError("grid cells must be integers equal to 0 or 1")
    for name, point in (("start", start), ("goal", goal)):
        if not isinstance(point, tuple) or len(point) != 2 or any(type(c) is not int for c in point):
            raise ValueError(name + " must be an (x, y) tuple of two integers")
        x, y = point
        if not (0 <= x < width and 0 <= y < len(grid)) or grid[y][x] != 0:
            raise ValueError(name + " must be inside the grid on a free cell")


def octile(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (DIAG - 1) * min(dx, dy)


def heuristic(v, goal):
    return 1.0 / (1.0 + octile(v, goal))


def neighbors(grid, u):
    x, y = u
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if not (0 <= ny < len(grid) and 0 <= nx < len(grid[0])):
                continue
            if grid[ny][nx] == 1:
                continue
            if dx != 0 and dy != 0:
                if grid[y][nx] == 1 or grid[ny][x] == 1:
                    continue
            yield (nx, ny)


def edge(a, b):
    return (a, b) if a < b else (b, a)


def path_cost(path):
    total = 0.0
    for a, b in zip(path, path[1:]):
        total += octile(a, b)
    return total


class ACOSolver:
    """Configurable ACO search on an 8-connected grid without corner-cutting.

    Each solve call starts with fresh pheromone and a private seeded RNG.
    Reusing an instance therefore produces the same path for the same input.
    """

    def __init__(self, *, seed=0, ants=12, rounds=15, budget=512,
                 alpha=1, beta=2, rho=0.2, q=10.0):
        for name, value in (("ants", ants), ("rounds", rounds), ("budget", budget)):
            if type(value) is not int or value < 0:
                raise ValueError(name + " must be an integer >= 0")
        for name, value in (("alpha", alpha), ("beta", beta), ("rho", rho), ("q", q)):
            if type(value) not in (int, float) or not isfinite(value) or value < 0:
                raise ValueError(name + " must be finite and >= 0")
        if rho > 1:
            raise ValueError("rho must be in [0, 1]")
        self.seed = seed
        self.ants = ants
        self.rounds = rounds
        self.budget = budget
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q

    def construct_ant(self, grid, start, goal, bonus, rng):
        """Build one ant's path; solve handles start == goal separately."""
        path = [start]
        seen = {start}
        for _ in range(self.budget):
            u = path[-1]
            options = [v for v in neighbors(grid, u) if v not in seen]
            if not options:
                path.pop()
                if not path:
                    return None
                continue
            weights = []
            for v in options:
                tau = 1.0 + bonus[edge(u, v)]
                eta = heuristic(v, goal)
                weights.append(tau ** self.alpha * eta ** self.beta)
            v = rng.choices(options, weights=weights, k=1)[0]
            path.append(v)
            seen.add(v)
            if v == goal:
                return path
        return None

    def evaporate_and_deposit(self, bonus, deposit):
        """Evaporate only the bonus, then add this round's deposits in place."""
        for e in bonus:
            bonus[e] *= 1.0 - self.rho
        bonus.update(deposit)

    def solve(self, grid, start, goal):
        """Return the best path found within the configured budget, or None."""
        validate_input(grid, start, goal)
        if start == goal:
            return [start]
        rng = Random(self.seed)
        bonus = Counter()
        best = None
        best_cost = float("inf")
        for _ in range(self.rounds):
            deposit = Counter()
            for _ in range(self.ants):
                path = self.construct_ant(grid, start, goal, bonus, rng)
                if path is None:
                    continue
                cost = path_cost(path)
                amount = self.q / cost
                for a, b in zip(path, path[1:]):
                    deposit[edge(a, b)] += amount
                if cost < best_cost:
                    best = path
                    best_cost = cost
            self.evaporate_and_deposit(bonus, deposit)
        return best


def solve(grid, start, goal):
    return ACOSolver().solve(grid, start, goal)
