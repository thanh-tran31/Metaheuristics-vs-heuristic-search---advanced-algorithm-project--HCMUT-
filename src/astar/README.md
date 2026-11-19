# Module A* (`src/astar`)

Cài đặt thuật toán tìm kiếm **A\* 8 hướng (8-connected)** với khoảng cách **Octile** và quy tắc chống cắt góc vật cản (**no corner-cutting**), tuân thủ đầy đủ bản quy ước chung của nhóm (`IO_of_Algo_and_Repo_Format.md`).

## 1. Cách sử dụng theo Project Contract

```python
from src.astar import solve

grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
start = (0, 0)
goal = (4, 4)

path = solve(grid, start, goal)
# -> [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 3), (4, 4)] (hoặc đường tối ưu tương đương)
```

## 2. Thu thập chỉ số chi tiết (Phục vụ Benchmark & Ablation Study)

```python
from src.astar import solve_with_stats, get_last_stats

res = solve_with_stats(
    grid,
    start,
    goal,
    heuristic="octile",     # "octile" | "euclidean" | "chebyshev" | "manhattan" | "dijkstra"
    weight=1.0,             # Weighted A* (w >= 1.0)
    tie_breaking="high_g",  # "high_g" | "low_g" | "fifo"
)

print(res.path)             # list[tuple[int, int]] | None
print(res.cost)             # Chi phí đường đi tối ưu
print(res.expanded_nodes)   # Số node lấy ra khỏi OPEN và mở rộng (<= 1 lần/node khi consistent)
print(res.evaluated_nodes)  # Số lần đánh giá trạng thái lân cận
print(res.generated_nodes)  # Số lần đẩy trạng thái vào Min-Heap
print(res.runtime_ms)       # Thời gian chạy (ms)
```

Nếu script benchmark chỉ gọi `solve(grid, start, goal)`, vẫn có thể lấy thống kê của lần chạy gần nhất qua `get_last_stats()`.

## 3. Ma trận Heuristic (Phục vụ Hybrid ACO-A\* ở Tuần 48)

```python
from src.astar import compute_heuristic_matrix

# Ma trận khoảng cách Octile từ mọi ô tới goal: H[y][x]
octile_mat = compute_heuristic_matrix(grid, goal, mode="octile")

# Ma trận chi phí đường đi ngắn nhất chính xác (Reverse Dijkstra có tránh vật cản) tới goal
exact_mat = compute_heuristic_matrix(grid, goal, mode="exact")
```

## 4. Chạy Unit Tests

```bash
python3 -m unittest discover -s tests -v
```

