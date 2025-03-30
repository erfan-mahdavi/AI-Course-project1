from collections import deque

class BFSSolver:
    
    # we assume aryan has no knowledge about map being squared or exit point being at (n-1, n-1)
    
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def find_shortest_path(self):
        start = (0, 0)
        goal = (self.grid_size - 1, self.grid_size - 1)
        queue = deque()
        visited = set()
        visited.add(start)
        queue.append((start, []))
        directions = [
            ('down', 1, 0),
            ('right', 0, 1)
        ]
        while queue:
            current_pos, path = queue.popleft()
            # print(current_pos,path)
            if current_pos == goal:
                return path
            for dir_name, di, dj in directions:
                new_i = current_pos[0] + di
                new_j = current_pos[1] + dj
                new_pos = (new_i, new_j)
                if 0 <= new_i < self.grid_size and 0 <= new_j < self.grid_size and (new_pos not in visited):
                    queue.append((new_pos, path + [dir_name]))
                    visited.add(new_pos)
        return None
