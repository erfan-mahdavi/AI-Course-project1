import time
from collections import deque

class IDSSolver:
   
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def depth_limited_search(self, depth_limit, goal):
        start = (0, 0)
        max_stack_size = 0
        stack = [(start, [], 0)]  # stack for (position, path, depth)
        visited = set()  # some states in dfs with a depth limit are redundant,since every path to a state is optimal we don'r explore it 

        while stack:
            max_stack_size = max(max_stack_size, len(stack))
            current_pos, path, depth = stack.pop()
            if current_pos in visited:
                continue
            visited.add(current_pos)

            if current_pos == goal:
                return path, max_stack_size, True  

            if depth < depth_limit:
                for move_name, (di, dj) in [('down', (1, 0)), ('right', (0, 1))]:
                    new_pos = (current_pos[0] + di, current_pos[1] + dj)
                    if (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size) :
                        stack.append((new_pos, path + [move_name], depth + 1))

        return None, max_stack_size, False  

    def find_path(self):
        goal = (self.grid_size - 1, self.grid_size - 1)
        if goal[0] < 0 or goal[1] < 0 or goal[0] >= self.grid_size or goal[1] >= self.grid_size:
            return "there is no exit in the map", 0
        
        max_depth = 2 * (self.grid_size - 1)
        for depth_limit in range(max_depth + 1):
            #depth_limit = max_depth
            result, max_stack, found = self.depth_limited_search(depth_limit, goal)
            print(f"IDS with depth limit {depth_limit}: Max stack size = {max_stack}")

            if result:
                return result, max_stack  # return the path when found

        return "there is no exit in the map", 0  # no path found after all depths


#----------------------------------------------------------------------------------------

class BFSSolver:
    def __init__(self, grid_size):
        self.grid_size = grid_size

    def find_shortest_path(self, goal=None):
        # goal is (n-1, n-1)
        if goal is None:
            goal = (self.grid_size - 1, self.grid_size - 1)
        # goal is within the grid
        if not (0 <= goal[0] < self.grid_size and 0 <= goal[1] < self.grid_size):
            return "there is no exit in the map", 0

        start = (0, 0)
        queue = deque([(start, [])])
        visited = set([start])  #mark the start as visited 
        max_queue_size = 1

        directions = [
            ('down', 1, 0),
            ('right', 0, 1)
        ]

        while queue:
            max_queue_size = max(max_queue_size, len(queue))
            current_pos, path = queue.popleft()

            if current_pos == goal:
                return path, max_queue_size + len(visited)  # Combined space usage metric

            for dir_name, di, dj in directions:
                new_pos = (current_pos[0] + di, current_pos[1] + dj)
                if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                    if new_pos not in visited:
                        visited.add(new_pos)
                        queue.append((new_pos, path + [dir_name]))

        return "there is no exit in the map", max_queue_size + len(visited)


def main():
    try:
        n = int(input("Enter grid size "))
    except ValueError:
        print("wrong input")
        return

    print("\nRunning BFS:")
    bfs_solver = BFSSolver(n)
    start_time = time.perf_counter()
    bfs_result, bfs_space = bfs_solver.find_shortest_path()
    bfs_time = time.perf_counter() - start_time
    print("BFS result:", bfs_result)
    print("BFS time taken: {:.6f} seconds".format(bfs_time))
    print(f"BFS approximate space complexity: {bfs_space} (queue + visited)")

    print("\nRunning IDS:")
    ids_solver = IDSSolver(n)
    start_time = time.perf_counter()
    ids_result, ids_space = ids_solver.find_path()
    ids_time = time.perf_counter() - start_time
    print("IDS result:", ids_result)
    print("IDS time taken: {:.6f} seconds".format(ids_time))
    print(f"IDS approximate space complexity: {ids_space} (max stack + visited in the final depth)")

if __name__ == "__main__":
    main()
