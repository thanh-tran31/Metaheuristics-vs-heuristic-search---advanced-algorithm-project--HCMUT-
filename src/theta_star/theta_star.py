import heapq
import math
from typing import List, Optional, Sequence, Tuple

Node = Tuple[int, int]

_DIRECTIONS = [
    (0, -1), (0, 1), (-1, 0), (1, 0),
    (-1, -1), (1, -1), (-1, 1), (1, 1),
]


class PriorityQueue:

    def __init__(self):
        self.elements = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = 0

    def __contains__(self, node: Node) -> bool:
        return node in self.entry_finder

    def Insert(self, node: Node, priority: float):
        if node in self.entry_finder:
            self.remove(node)
        self.counter += 1
        entry = [priority, self.counter, node]
        self.entry_finder[node] = entry
        heapq.heappush(self.elements, entry)

    def remove(self, node: Node):
        entry = self.entry_finder.pop(node)
        entry[-1] = self.REMOVED

    def pop(self) -> Node:
        while self.elements:
            _, _, node = heapq.heappop(self.elements)
            if node is not self.REMOVED:
                del self.entry_finder[node]
                return node
        raise KeyError('pop from an empty priority queue')

    def is_empty(self) -> bool:
        return len(self.entry_finder) == 0


def c(a: Node, b: Node) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def h(a: Node, goal: Node) -> float:
    return math.hypot(a[0] - goal[0], a[1] - goal[1])


def _validate(grid: Sequence[Sequence[int]], start: Node, goal: Node) -> None:
    if not grid or not grid[0]:
        raise ValueError('grid rỗng')
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError('grid không phải hình chữ nhật')
    height = len(grid)
    for name, p in (('start', start), ('goal', goal)):
        if len(p) != 2:
            raise ValueError(f'{name} phải là (x, y)')
        x, y = p
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f'{name} {p} nằm ngoài bản đồ')
        if grid[y][x] != 0:
            raise ValueError(f'{name} {p} là vật cản')


def get_neighbors(grid, current: Node) -> List[Node]:
    x, y = current
    height, width = len(grid), len(grid[0])
    result = []
    for dx, dy in _DIRECTIONS:
        nx, ny = x + dx, y + dy
        if not (0 <= nx < width and 0 <= ny < height) or grid[ny][nx] != 0:
            continue
        if dx and dy and (grid[y][nx] != 0 or grid[ny][x] != 0):
            continue  # no corner-cutting
        result.append((nx, ny))
    return result


def has_line_of_sight(grid, a: Node, b: Node) -> bool:
    """LOS giữa tâm ô a và tâm ô b, dùng duyệt lưới bằng số nguyên (không sai số float)."""
    x, y = a
    nx = abs(b[0] - a[0])
    ny = abs(b[1] - a[1])
    sx = 1 if b[0] > a[0] else -1
    sy = 1 if b[1] > a[1] else -1
    ix = iy = 0
    while ix < nx or iy < ny:
        decision = (1 + 2 * ix) * ny - (1 + 2 * iy) * nx
        if decision == 0:
            # đoạn thẳng đi đúng qua đỉnh ô: cả 2 ô kề bên cũng phải trống
            if grid[y][x + sx] != 0 or grid[y + sy][x] != 0:
                return False
            x += sx
            y += sy
            ix += 1
            iy += 1
        elif decision < 0:
            x += sx
            ix += 1
        else:
            y += sy
            iy += 1
        if grid[y][x] != 0:
            return False
    return True


def _update_vertex(grid, parent, g_score, open_set, current, neighbor, goal):
    inf = float('inf')
    p = parent[current]
    if has_line_of_sight(grid, p, neighbor):      # Path 2
        base = p
    else:                                         # Path 1
        base = current
    new_g = g_score[base] + c(base, neighbor)
    if new_g < g_score.get(neighbor, inf):
        g_score[neighbor] = new_g
        parent[neighbor] = base
        if neighbor in open_set:
            open_set.remove(neighbor)
        open_set.Insert(neighbor, new_g + h(neighbor, goal))


def reconstruct_path(parent: dict, current: Node) -> List[Node]:
    path = [current]
    while parent[current] != current:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path


def solve(grid, start, goal) -> Optional[List[Node]]:
    _validate(grid, start, goal)
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    if start == goal:
        return [start]

    open_set = PriorityQueue()
    g_score = {start: 0.0}
    parent = {start: start}
    closed = set()
    open_set.Insert(start, h(start, goal))

    while not open_set.is_empty():
        current = open_set.pop()
        if current == goal:
            return reconstruct_path(parent, current)
        closed.add(current)
        for neighbor in get_neighbors(grid, current):
            if neighbor not in closed:
                _update_vertex(grid, parent, g_score, open_set,
                               current, neighbor, goal)
    return None
