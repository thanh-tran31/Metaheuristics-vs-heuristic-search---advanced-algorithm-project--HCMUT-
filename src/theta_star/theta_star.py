from typing import List, Tuple, Optional

def has_line_of_sight(grid: List[List[int]], node1: Tuple[int, int], node2: Tuple[int, int]) -> bool:
    """
    Kiểm tra Line-of-Sight hợp lệ giữa 2 điểm.
    Cần đảm bảo không cắt góc vật cản (no corner-cutting) theo quy ước nhóm.
    """
    # TODO: Cài đặt logic Bresenham hoặc Ray-casting tại đây
    pass

def get_neighbors(grid: List[List[int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Lấy danh sách 8 ô lân cận hợp lệ.
    """
    # TODO: Trả về các ô lân cận hợp lệ
    pass

def solve(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """
    Hàm giải chính cho Theta*. Trả về list các tọa độ đường đi hoặc None.
    """
    if start == goal:
        return [start]
        
    max_y = len(grid)
    max_x = len(grid[0]) if max_y > 0 else 0
    
    # Kiểm tra tính hợp lệ của điểm bắt đầu và kết thúc
    if not (0 <= start[0] < max_x and 0 <= start[1] < max_y) or grid[start[1]][start[0]] == 1:
        return None
    if not (0 <= goal[0] < max_x and 0 <= goal[1] < max_y) or grid[goal[1]][goal[0]] == 1:
        return None

    # TODO: Khởi tạo open_list, closed_set, g_score, parent
    
    # TODO: Vòng lặp chính của Theta*
    # while open_list:
    #     ... 
    
    return None

def reconstruct_path(parent: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Truy xuất ngược đường đi từ dictionary parent.
    """
    path = [current]
    while current in parent:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path