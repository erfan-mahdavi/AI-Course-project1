from collections import deque

class BFSSolver:
    """
    we use breadth-first search to find the shortest path from (0,0) to (n-1, n-1) ignore all coin gains or thief encounters
    we assume aryan has no knowledge about map being squared or exit point being at (n-1, n-1)
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def find_shortest_path(self):
        start = (0, 0)
        end = (self.grid_size - 1, self.grid_size - 1)
        queue = deque()
        queue.append((start, []))
        visited = set([start])  # there is no purpose in tree search so we do graph search by using visited set
        directions = [
            ('up', -1, 0),
            ('down', 1, 0),
            ('left', 0, -1),
            ('right', 0, 1)
        ]
        while queue:
            current_pos, path = queue.popleft()
            print(current_pos,path)
            if current_pos == end:
                return path
            for dir_name, di, dj in directions:
                new_i = current_pos[0] + di
                new_j = current_pos[1] + dj
                new_pos = (new_i, new_j)
                if 0 <= new_i < self.grid_size and 0 <= new_j < self.grid_size:
                    if new_pos not in visited:
                        visited.add(new_pos)
                        queue.append((new_pos, path + [dir_name]))
        return None
