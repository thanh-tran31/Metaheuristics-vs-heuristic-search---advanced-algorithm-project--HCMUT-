import sys
from math import dist
from pathlib import Path
from src.theta_star.theta_star import solve

grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

start = (0, 3)
goal = (7, 1)

path = solve(grid, start, goal)
print("Path:", path)
if path:
    print("Theta* length:", sum(dist(a, b) for a, b in zip(path, path[1:])))
    print("Direct length:", dist(start, goal))