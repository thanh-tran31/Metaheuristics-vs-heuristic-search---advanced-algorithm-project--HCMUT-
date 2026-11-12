# Project Contract — A*, Theta* & ACO

> **Mục đích:** Thống nhất **input, output và cấu trúc repository** để 3 thành viên có thể viết thuật toán độc lập, còn parser, tests và benchmark được dùng chung.

## 1. Input chung

Cả 3 thuật toán cùng nhận **3 tham số**:

```python
def solve(grid, start, goal):
    ...
```

| Tham số | Kiểu | Ý nghĩa |
|---|---|---|
| `grid` | `list[list[int]]` | Bản đồ 2D: `0` = ô đi được, `1` = vật cản |
| `start` | `tuple[int, int]` | Điểm xuất phát `(x, y)` |
| `goal` | `tuple[int, int]` | Điểm đích `(x, y)` |

Ví dụ:

```python
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]
start = (0, 0)
goal  = (4, 4)

path = solve(grid, start, goal)
```

**Quy ước bắt buộc:**

- `(x, y)` = `(cột, hàng)`; góc trên trái là `(0, 0)`; truy cập ô bằng `grid[y][x]`.
- Bản đồ là hình chữ nhật; `start` và `goal` phải nằm trong bản đồ và không phải vật cản.
- Một bước ngang/dọc có cost **1**; bước chéo có cost **√2**.
- A* và ACO di chuyển trên **8 ô lân cận**, **không cắt góc vật cản** (*no corner-cutting*).
- Theta* cũng mở rộng 8 ô lân cận nhưng có thể nối các waypoint xa nhau nếu có **line-of-sight (LOS)** hợp lệ. Nhóm dùng **tâm ô** làm tọa độ waypoint; đoạn LOS không được xuyên qua vật cản hoặc lách qua góc bị cấm theo quy tắc no corner-cutting đã thống nhất.
- Thuật toán **không sửa đổi `grid` đầu vào**.
- File MovingAI `.map` / `.scen` do **parser chung** đọc và chuyển thành 3 tham số trên. Giá trị `optimal_length` trong `.scen` **chỉ dùng để đánh giá**, không truyền vào `solve()`.

## 2. Output chung

```python
# Tìm thấy đường đi
path = [(0, 0), (1, 0), (2, 1), ..., (4, 4)]

# Không tìm được đường trong lần chạy / ngân sách hiện tại
path = None
```

- Output là `list[tuple[int, int]] | None`.
- Nếu có đường: phần tử đầu là `start`, phần tử cuối là `goal`; mỗi waypoint là một tọa độ hợp lệ.
- **A* / ACO:** Hai waypoint liên tiếp phải là các ô lân cận hợp lệ (8 hướng, no corner-cutting).
- **Theta*:** Hai waypoint liên tiếp có thể cách xa nhau, nhưng phải có LOS hợp lệ.
- Khi `start == goal`, trả `[(x, y)]` (chi phí bằng 0).
- `None` nghĩa là *không tìm được nghiệm trong lần chạy này*; **đặc biệt với ACO, không có nghĩa chắc chắn bản đồ vô nghiệm**.
- **Benchmark chung tự tính lại** độ dài, kiểm tra tính hợp lệ và đo runtime; không phụ thuộc số liệu do thuật toán tự báo cáo.

**Tham số riêng** như `alpha`, `beta`, `rho`, tie-breaking... do người phụ trách thuật toán quản lý trong folder của mình; **không thuộc input chung**. Riêng ACO cần hỗ trợ **seed có thể chỉ định** (qua tham số tùy chọn hoặc cách cấu hình thống nhất với người làm benchmark) để chạy lại kết quả và kiểm định ≥30 lần/instance.

## 3. Cấu trúc repository

```text
project/
├── data/
│   ├── maps/                 # MovingAI .map (ít nhất 2 họ bản đồ)
│   ├── scenarios/            # MovingAI .scen
│   └── fixtures/             # Grid nhỏ, đáp án biết trước cho unit test
├── src/
│   ├── astar/                # Thành viên A*: code, helper riêng
│   ├── theta_star/           # Thành viên Theta*: code, helper riêng
│   ├── aco/                  # Thành viên ACO: code, helper riêng
│   └── common/
│       └── parser.py         # Đọc .map/.scen, xuất grid/start/goal chung
├── tests/                    # Parser, đường hợp lệ, unit/integration tests
├── experiments/              # Benchmark runner, metrics, kết quả
├── docs/                     # Tài liệu và báo cáo
├── notebooks/                # Phân tích/trực quan hóa tùy chọn
└── README.md
```

**Phân công giao tiếp:** Mỗi thuật toán cung cấp một hàm `solve(grid, start, goal)` có thể import từ folder của mình. Tên file cụ thể và logic bên trong do từng người quyết định; người viết benchmark chỉ cần biết import path đã thống nhất.

## 4. Nguyên tắc kiểm định chung

| Đối tượng | Cần kiểm tra |
|---|---|
| Parser | Đọc đúng kích thước, ô chặn, start/goal; báo lỗi input sai |
| Output của cả 3 | Đúng điểm đầu/cuối; không xuyên vật cản; cost do validator tính lại |
| A* | Cost bằng nghiệm tối ưu grid (Dijkstra hoặc MovingAI `optimal_length` nếu cùng quy tắc) |
| Theta* | Mọi đoạn đi hợp lệ theo LOS; **không yêu cầu** luôn tối ưu any-angle |
| ACO | Đường hợp lệ nếu thành công; `None` được tính là thất bại trong ngân sách; chạy lại bằng seed |
| Benchmark | Cùng map/scenario; tách dữ liệu chỉnh tham số và dữ liệu báo cáo; ACO ≥30 lần/instance |

> **Chốt trước khi code:** Toàn nhóm phải xác nhận quy ước `(x, y)`, `0/1`, **tâm ô**, no corner-cutting và chữ ký hàm `solve()`. Đây là các quy ước của **nhóm**, không tự động đồng nhất mọi chi tiết hình học với paper Theta* gốc.
