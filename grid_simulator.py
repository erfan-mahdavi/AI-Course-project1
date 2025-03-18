
# for bfs search we Simulates full path given a list of moves. later we will add step by step grid simulation for A*.
class GridSimulator:
    
    def __init__(self, grid):
        self.original_grid = [row.copy() for row in grid]
        self.n = len(grid)

    def simulate_path(self, path):
        current_grid = [row.copy() for row in self.original_grid]
        coins_collected = 0
        coins_stolen = 0
        current_pos = (0, 0)
        has_thief = False

        # process starting cell at (0,0).
        i, j = current_pos
        cell = current_grid[i][j]
        if cell == '!':
            has_thief = True
        elif isinstance(cell, int):
            if cell > 0:
                coins_collected += cell
                current_grid[i][j] = -1  # mark treasure as collected.
            else:
                coins_collected += cell

        for direction in path:
            i, j = current_pos
            if direction == 'up':
                new_i, new_j = i - 1, j
            elif direction == 'down':
                new_i, new_j = i + 1, j
            elif direction == 'left':
                new_i, new_j = i, j - 1
            elif direction == 'right':
                new_i, new_j = i, j + 1
            else:
                raise ValueError("Invalid direction")
            if not (0 <= new_i < self.n and 0 <= new_j < self.n):
                raise ValueError("Move out of bounds")
            current_pos = (new_i, new_j)
            cell = current_grid[new_i][new_j]
            # if a thief is active process the thief effect seperately
            if has_thief:
                if cell == '!':
                    # Two thieves meet: no coin transfer.
                    pass
                elif isinstance(cell, int):
                    if cell > 0:
                        coins_stolen += cell
                        current_grid[new_i][new_j] = -1
                    else:
                        coins_stolen += abs(cell)
                has_thief = False  # Thief effect used
                continue
            else:
                # normal cell processing
                if cell == '!':
                    has_thief = True
                elif isinstance(cell, int):
                    if cell > 0:
                        coins_collected += cell
                        current_grid[new_i][new_j] = -1
                    else:
                        coins_collected += cell

        return {
            'coins_collected': coins_collected,
            'coins_stolen': coins_stolen,
            'path': path
        }

