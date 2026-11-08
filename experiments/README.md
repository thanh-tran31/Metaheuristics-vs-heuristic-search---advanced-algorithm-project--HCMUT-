# Automated Experiment Scripts

Thư mục chứa các kịch bản chạy thử nghiệm quy mô lớn và lưu trữ kết quả.

## Nhiệm vụ
- Chạy batch tự động đối chuẩn giữa Heuristic Search (A*, Theta*) và Metaheuristics (ACO/GA).
- Lặp lại tối thiểu 30 lần cho các thuật toán ngẫu nhiên (Metaheuristic) với hạt giống (seed) cố định.
- Xuất dữ liệu log/metrics ra định dạng `.csv` (Runtime, Path Length, Evaluated Nodes, Optimality Gap).
