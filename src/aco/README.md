# Module ACO (`src/aco`)

Cài đặt thuật toán **Ant Colony Optimization (ACO)** tìm đường trên lưới **8 hướng (8-connected)** với heuristic **Octile** và quy tắc chống cắt góc vật cản (**no corner-cutting**), theo bản quy ước chung của nhóm (`IO_of_Algo_and_Repo_Format.md`).

## 1. Cách sử dụng theo Project Contract

```python
from src.aco import solve

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
print(path)  # list[tuple[int, int]] | None
```

- Tọa độ `(x, y)` = `(cột, hàng)`, truy cập bằng `grid[y][x]`.
- `0` = ô đi được, `1` = vật cản; `grid` không bị sửa đổi.
- Bước ngang/dọc có chi phí `1`, bước chéo có chi phí `sqrt(2)`.
- `start == goal`: trả `[start]`.
- Input hoặc cấu hình không hợp lệ: báo `ValueError`.
- `None`: không tìm được đường trong ngân sách hiện tại; chưa khẳng định bản đồ vô nghiệm.

## 2. Cấu hình thuật toán (Phục vụ Benchmark & Ablation Study)

```python
from src.aco import ACOSolver

solver = ACOSolver(
    seed=7,       # Hạt giống để tái lập kết quả
    ants=12,      # Số kiến mỗi vòng
    rounds=15,    # Số vòng cập nhật pheromone
    budget=512,   # Số lượt xử lý tối đa mỗi kiến trong một vòng
    alpha=1,      # Mức ảnh hưởng pheromone
    beta=2,       # Mức ảnh hưởng heuristic
    rho=0.2,      # Tỉ lệ bay hơi bonus pheromone
    q=10.0,       # Lượng rải cơ sở: mỗi cạnh nhận q / path_cost
)

path = solver.solve(grid, start, goal)
```

`budget` tính cả bước tiến và quay lui. `ants`, `rounds`, `budget` là số nguyên không âm; `alpha`, `beta`, `q` hữu hạn và không âm; `rho` thuộc `[0, 1]`.

Mỗi lần gọi `solver.solve()` khởi tạo lại pheromone và bộ sinh số ngẫu nhiên theo `seed`. Cùng input và cấu hình cho cùng kết quả trong cùng môi trường chạy; các đối tượng giữ cấu hình riêng.

Khi benchmark ACO, chạy tối thiểu 30 lần cho mỗi instance với các seed được ghi lại:

```python
for seed in range(30):
    path = ACOSolver(seed=seed).solve(grid, start, goal)
    # Kiểm tra đường hợp lệ, tính lại cost và đo runtime trong benchmark chung.
```

ACO không bảo đảm đường đi tối ưu. Phiên bản hiện tại trả đường đi, chưa cung cấp số lần đánh giá trạng thái hoặc lịch sử hội tụ.

## 3. Các hàm và phương thức chính (`solver.py`)

| Hàm / Phương thức | Chức năng |
|---|---|
| `validate_input(grid, start, goal)` | Kiểm tra grid và tọa độ đầu vào |
| `octile(a, b)` | Khoảng cách Octile, bỏ qua vật cản |
| `heuristic(v, goal)` | Độ hấp dẫn `1 / (1 + octile(v, goal))` |
| `neighbors(grid, u)` | Sinh ô lân cận hợp lệ, không cắt góc |
| `edge(a, b)` | Tạo khóa chung cho hai chiều của cạnh |
| `path_cost(path)` | Tính chi phí đường đi đã hợp lệ |
| `ACOSolver.construct_ant(...)` | Một kiến chọn bước theo trọng số và quay lui |
| `ACOSolver.evaporate_and_deposit(...)` | Bay hơi bonus và cộng lượng pheromone rải trong vòng |
| `ACOSolver.solve(grid, start, goal)` | Tìm đường với cấu hình của đối tượng |
| `solve(grid, start, goal)` | Giao diện chung, dùng `ACOSolver` mặc định |

## 4. Chạy Unit Tests

Chạy từ thư mục gốc dự án:

```bash
python -m unittest discover -s tests -v
```
