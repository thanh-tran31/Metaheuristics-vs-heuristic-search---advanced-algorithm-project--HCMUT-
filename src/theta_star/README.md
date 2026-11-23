# Theta*

- `grid[y][x]`: `0` = đi được, `1` = vật cản
- `start`, `goal`: `(x, y)` = `(cột, hàng)`, góc trên trái là `(0, 0)`
- Waypoint là tâm ô; 8 hướng, không cắt góc vật cản
- Không có đường: `None`. `start == goal`: `[start]`
- Input sai (ngoài bản đồ, start/goal là vật cản, grid không chữ nhật): `ValueError`
- `grid` không bị sửa đổi

## Ví dụ

```python
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
print(solve(grid, (0, 0), (4, 4)))
```

## Hàm phụ (`theta_star.py`)

| Hàm | Việc |
|---|---|
| `get_neighbors(grid, node)` | Các ô lân cận đi được |
| `has_line_of_sight(grid, a, b)` | Đoạn a→b có chạm vật cản không |
| `c(a, b)`, `h(a, goal)` | Chi phí, heuristic (Euclid) |
| `reconstruct_path(parent, node)` | Truy ngược đường đi |
| `PriorityQueue` | Hàng đợi ưu tiên |

## Test

```
python -m unittest discover -s tests -v
```