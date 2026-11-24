# Phản ví dụ của Basic Theta* — Trường hợp Hình 9(a)

## 1. Bản chất vấn đề
**Basic Theta\*** không đảm bảo luôn tìm được đường đi ngắn nhất thực sự (*true shortest path*) vì cha của một đỉnh chỉ có thể là **đỉnh kề có line-of-sight** tới đỉnh đó, hoặc là **cha của một đỉnh kề** đó[cite: 17].

Trong **Hình 9(a)** của bài báo *Theta*: Any-Angle Path Planning on Grids*[cite: 2, 16, 17]:
- Đoạn thẳng nối trực tiếp giữa đỉnh nguồn và đỉnh đích nằm hoàn toàn trong vùng không bị chắn (*line-of-sight* tồn tại)[cite: 17].
- Tuy nhiên, do thứ tự mở rộng đỉnh và ràng buộc chỉ lưu một biến `parent` cho mỗi đỉnh, đỉnh nguồn không trở thành `parent` trực tiếp của đỉnh đích (do không nằm trong tập đỉnh kề có line-of-sight hoặc `parent` của đỉnh kề có line-of-sight tới đỉnh đích)[cite: 17].
- Hệ quả là đường đi trích xuất từ `parent` bị bẻ góc / đi qua một đỉnh trung gian khác thay vì đi thẳng một mạch[cite: 16, 17].

## 2. Cơ chế xảy ra hiện tượng đi vòng qua node khác
1. Trong quá trình tìm kiếm, các đỉnh xung quanh bị gán `parent` là các đỉnh trung gian sớm hơn (ví dụ đỉnh trên đường biên phía trên/dưới vật cản)[cite: 16, 17].
2. Khi xét cập nhật cho đỉnh đích hoặc các đỉnh sát đích, Theta* chỉ kiểm tra `LineOfSight(parent(s), s')` thay vì kiểm tra mọi cặp tổ tiên/đỉnh có thể nối thẳng[cite: 12, 17].
3. Vì `parent(s)` hiện tại không có line-of-sight tới `s'` (hoặc chi phí qua `parent(s)` không được áp dụng tại thời điểm đó), thuật toán giữ `parent` qua node trung gian, dẫn đến đường đi dài hơn hoặc bằng đường nối thẳng trực tiếp[cite: 12, 17].

## 3. Liên hệ với bộ test
- Matrix `0/1` (`0` = ô trống, `1` = vật cản, truy cập `grid[y][x]`) được đặt trực tiếp trong `tests/test_counter_theta_star.py`.
- File test gọi `solve(grid, start, goal)` từ `src.theta_star.theta_star`[cite: 1], in ra danh sách waypoint, độ dài đường đi của Theta* và so sánh với khoảng cách đường thẳng trực tiếp giữa `start` và `goal`.