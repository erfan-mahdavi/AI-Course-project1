def convert_grid_to_tuple(grid):
    # convert mutable list to immutable tuple
    return tuple(tuple(row) for row in grid)

class GridSimulator:
    
    #for bfs search we Simulates full path given a list of moves.
    
    def __init__(self, grid):
        self.original_grid = [row.copy() for row in grid]
        self.n = len(grid)

    def simulate_path(self, path):
        current_grid = [row.copy() for row in self.original_grid]
        coins_collected = 0
        coins_stolen = 0
        current_pos = (0, 0)
        has_thief = False

        # Process starting cell at (0,0)
        i, j = current_pos
        cell = current_grid[i][j]
        if cell == '!':
            has_thief = True
        elif isinstance(cell, int):
            if cell > 0:
                coins_collected += cell
                current_grid[i][j] = -1  # Mark treasure as collected.
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
            # if a thief is active process the thief effect exclusively
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
                has_thief = False  # Thief effect used.
                continue
            else:
                # Normal cell processing.
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

class SimulatorState:
    """
    A state for the A* search for maximum profit.
    It records the current position, the coins collected/stolen,
    the grid state (as an immutable tuple of tuples), and whether a thief is active.
    """
    def __init__(self, pos, coins_collected, coins_stolen, grid, has_thief):
        self.pos = pos              # (i, j)
        self.coins_collected = coins_collected
        self.coins_stolen = coins_stolen
        self.grid = grid            # Immutable grid state (tuple of tuples)
        self.has_thief = has_thief

    def net_coins(self):
        return self.coins_collected - self.coins_stolen

    def __hash__(self):
        # We use only pos, grid, and has_thief to keep the state space tractable.
        return hash((self.pos, self.grid, self.has_thief))

    def __eq__(self, other):
        return (self.pos == other.pos and self.grid == other.grid and
                self.has_thief == other.has_thief)

def initial_state(grid):
    """
    Create the initial SimulatorState from the grid.
    Processes the starting cell (0,0) accordingly.
    """
    n = len(grid)
    grid_copy = [row.copy() for row in grid]
    coins_collected = 0
    coins_stolen = 0
    has_thief = False
    i, j = 0, 0
    cell = grid_copy[i][j]
    if cell == '!':
        has_thief = True
    elif isinstance(cell, int):
        if cell > 0:
            coins_collected += cell
            grid_copy[i][j] = -1
        else:
            coins_collected += cell
    return SimulatorState((0, 0), coins_collected, coins_stolen, convert_grid_to_tuple(grid_copy), has_thief)

def step(state, direction, n):
    """
    Simulate one move from the given state in the given direction.
    Returns the new SimulatorState (or None if the move is out of bounds).
    """
    grid_list = [list(row) for row in state.grid]
    i, j = state.pos
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
    if not (0 <= new_i < n and 0 <= new_j < n):
        return None

    new_coins_collected = state.coins_collected
    new_coins_stolen = state.coins_stolen
    new_has_thief = state.has_thief
    cell = grid_list[new_i][new_j]
    if new_has_thief:
        #process cell using thief effect
        if cell == '!':
            #two thieves fight so no coin transferis done
            pass
        elif isinstance(cell, int):
            if cell > 0:
                new_coins_stolen += cell
                grid_list[new_i][new_j] = -1
            else:
                new_coins_stolen += abs(cell)
        new_has_thief = False
    else:
        # Normal cell processing.
        if cell == '!':
            new_has_thief = True
        elif isinstance(cell, int):
            if cell > 0:
                new_coins_collected += cell
                grid_list[new_i][new_j] = -1
            else:
                new_coins_collected += cell
    new_state = SimulatorState((new_i, new_j), new_coins_collected, new_coins_stolen,
                               convert_grid_to_tuple(grid_list), new_has_thief)
    return new_state

def heuristic(state, n):
    
    #admissible heuristic for maximum profit because we optimistically assume that in the best case you could collect all remaining treasures.
    total_remaining = 0
    for row in state.grid:
        for cell in row:
            if isinstance(cell, int) and cell > 0:
                total_remaining += cell
    # Since our cost is defined as -net_coins and in heuristic function we should get an optimistic guess for our heuristic to be
    # admissible the best additional coin gain is total_remaining which is the sum of all treasure
    # so we return h(state) = - (total_remaining).    
    # this should change later to account for the manhatan cost of crossing which in best scenario is -1 for each one non treasure step
    return -total_remaining
