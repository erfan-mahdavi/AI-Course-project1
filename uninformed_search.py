from collections import deque

class BFSSolver:
    """
    A solver class that uses Breadth-First Search (BFS) algorithm to find the shortest path
    from the top-left corner (0,0) to the bottom-right corner (grid_size-1, grid_size-1)
    in a square grid. This implementation only allows downward and rightward movements.
    """
    
    # we assume aryan has no knowledge about map being squared or exit point being at (n-1, n-1)
    # (This comment suggests the solver does not assume the grid is square or that
    # the exit point is at the bottom-right corner, though the implementation does make these assumptions)
   
    def __init__(self, grid_size):
        """
        Initialize the BFS solver with the size of the grid.
        
        Args:
            grid_size: The size of the square grid (grid_size x grid_size)
        """
        self.grid_size = grid_size
        
    def find_shortest_path(self):
        """
        Finds the shortest path from the start (0,0) to the goal (grid_size-1, grid_size-1)
        using the Breadth-First Search algorithm.
        
        Returns:
            A tuple containing:
            - A list of directions to follow ('down' or 'right') to reach the goal,
              or None if no path exists
            - The number of states explored during the search
        """
        # Counter to track the number of states explored
        state_counter = 0
        
        # Define start and goal positions
        start = (0, 0)
        goal = (self.grid_size - 1, self.grid_size - 1)
        
        # Initialize the queue with the start position and an empty path
        queue = deque()
        
        # Keep track of visited positions to avoid cycles
        visited = set()
        visited.add(start)
        
        # Add the start position to the queue along with an empty path
        queue.append((start, []))
        
        # Define possible movement directions (only down and right)
        directions = [
            ('down', 1, 0),  # Move one step down
            ('right', 0, 1),  # Move one step right
        ]
        
        # BFS loop
        while queue:
            # Increment the state counter for each dequeued state
            state_counter += 1
            
            # Get the next position and path from the queue
            current_pos, path = queue.popleft()
            
            # If we've reached the goal, return the path and state count
            if current_pos == goal:
                return path, state_counter
                
            # Try each possible direction
            for dir_name, di, dj in directions:
                # Calculate the new position
                new_i = current_pos[0] + di
                new_j = current_pos[1] + dj
                new_pos = (new_i, new_j)
                
                # Check if the new position is valid and not visited
                if (0 <= new_i < self.grid_size and 
                    0 <= new_j < self.grid_size and 
                    (new_pos not in visited)):
                    # Add the new position to the queue with the updated path
                    queue.append((new_pos, path + [dir_name]))
                    # Mark the position as visited
                    visited.add(new_pos)
                    
        # If no path is found, return None and 0
        return None, 0