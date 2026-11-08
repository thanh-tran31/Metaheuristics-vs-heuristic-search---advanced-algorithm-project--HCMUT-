# Source Code Core

Mã nguồn chính của dự án.

## Cấu trúc module
- `core/`: Bộ phân tích bản đồ MovingAI (`.map`, `.scen`), định nghĩa Node và Graph.
- `heuristics/`: Thuật toán A* (8-connected, khoảng cách Octile), Theta* (Any-Angle).
- `metaheuristics/`: Thuật toán ACO / GA và thuật toán lai (Hybrid ACO-A*).
- `utils/`: Hàm kiểm tra tầm nhìn (Line-of-Sight Bresenham), bộ thu thập chỉ số (metrics collector).
