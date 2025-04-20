from grid_simulator_informed import State, Priority, PriorityQueue

class Astar:
    def __init__(self, n, grid, phase):
        # Initialize the A* algorithm
        self.n = n                # Grid size (n x n)
        self.grid = grid          # Grid representation with coins, empty cells, or thieves ('!')
        self.phase = phase        # Phase determines whether to maximize profit or minimize loss

        self.calculate_max_profit()


    def calculate_max_profit(self):

        # Collect positions of coins (positive integers) from current position to bottom-right
        coin_values = {}
        for r in range(0, self.n):
            for c in range(0, self.n):
                cell = self.grid[r][c]
                if isinstance(cell, int) and cell > 0:
                    coin_values[(r, c)] = cell
        
        # Use dynamic programming to calculate max coin sum from (row, col) to (n-1, n-1)
        self.dp_max_profit = [[0 for _ in range(self.n)] for _ in range(self.n)]
        
        # Iterate from the bottom-right corner of the grid to the top-left,
        # filling the `dp` table with the maximum coin value collectable from each cell to the bottom-right.
        # Start from (n-1, n-1) and move backward to (row, col), covering a subgrid.
        for r in range(self.n-1, -1, -1):
            for c in range(self.n-1, -1, -1):
                # Get the coin value at the current position, defaulting to 0 if there's no coin.
                pos_value = coin_values.get((r, c), 0)

                if r == self.n-1 and c == self.n-1:
                    # Base case: bottom-right corner of the grid, where the path ends.
                    self.dp_max_profit[r][c] = pos_value
                elif r == self.n-1:
                    # Last row (can only move right from here).
                    self.dp_max_profit[r][c] = pos_value + self.dp_max_profit[r][c+1]
                elif c == self.n-1:
                    # Last column (can only move down from here).
                    self.dp_max_profit[r][c] = pos_value + self.dp_max_profit[r+1][c]
                else:
                    # For all other cells, take the maximum of right and down moves.
                    self.dp_max_profit[r][c] = pos_value + max(self.dp_max_profit[r+1][c], self.dp_max_profit[r][c+1])

    def manhattan_distance(self, r1, c1, r2, c2):
        # Calculate Manhattan distance between two cells in the grid
        return abs(r2 - r1) + abs(c2 - c1)

    def heuristic_1(self, state):
        """
        Heuristic for maximizing collected profit (used in phase 2).
        Uses dynamic programming to compute the max coin value collectible from current cell to goal.
        If the player already encountered a thief, the heuristic returns 0.
        """
        row, col = state.state[0], state.state[1]
        has_thief = state.state[6]

        if has_thief:
            return 0  # No value if a thief has been encountered

        return self.dp_max_profit[row][col]  # Return estimated max profit from current position to goal

    def cost_1(self, state):
        # Cost function for phase 2 (maximize collected coins)
        return state.state[5]  # collected_coin

    def heuristic_2(self, state):
        """
        heuristic for minimizing loss (used in phase 3).
        Predicts the potential loss from encountering thieves on the way to the goal.
        Uses dynamic programming with improved efficiency.
        """
        # Extract current position and thief status from state
        row, col = state.state[0], state.state[1]
        has_thief = state.state[6]  # Boolean indicating if a thief has been encountered
        
        # Early return if already at the goal (bottom-right corner)
        if row == self.n - 1 and col == self.n - 1:
            return 0

        # Pre-compute and cache grid information ---
        # Cache thief locations and positive coin values to avoid repeated grid scans
        # This is done only once and reused across multiple calls to the heuristic
        if not hasattr(self, '_thief_locations') or not hasattr(self, '_coin_positions'):
            self._thief_locations = []  # List to store coordinates of all thieves
            self._coin_positions = {}   # Dictionary mapping (r,c) coordinates to coin values
            
            # Single grid scan to find all thieves and coins
            for r in range(self.n):
                for c in range(self.n):
                    cell = self.grid[r][c]
                    if cell == '!':  # Thief cell
                        self._thief_locations.append((r, c))
                    elif isinstance(cell, int) and cell > 0:  # Positive coin value
                        self._coin_positions[(r, c)] = cell
        
        # --- Handle case where no thief has been encountered yet ---
        if not has_thief:
            # Efficient future thief check ---
            # Filter thief locations to only include those ahead of current position
            # A thief is "ahead" if both its row and column indices are >= current position
            future_thieves = [
                (tr, tc) for tr, tc in self._thief_locations 
                if tr >= row and tc >= col
            ]
            
            # If no thieves ahead, we won't lose any coins
            if not future_thieves:
                return 0
        
        # Sparse dynamic programming table ---
        # Use a dictionary instead of a full 2D array to save space
        # Keys are (row, col) tuples, values are minimum potential losses
        dp = {}
        
        # Dynamic programming approach: compute minimum potential loss
        # Start from bottom-right (goal) and work backwards to current position
        for r in range(self.n - 1, row - 1, -1):
            for c in range(self.n - 1, col - 1, -1):
                # If thief already encountered, we might lose coins at this position
                # Otherwise, no immediate loss at this position
                potential_loss = self._coin_positions.get((r, c), 0) if has_thief else 0
                
                # Base cases and recursive cases for DP
                if r == self.n - 1 and c == self.n - 1:
                    # Base case: At goal position
                    dp[(r, c)] = potential_loss
                elif r == self.n - 1:
                    # Edge case: Bottom row - can only move right
                    dp[(r, c)] = potential_loss + dp[(r, c+1)]
                elif c == self.n - 1:
                    # Edge case: Rightmost column - can only move down
                    dp[(r, c)] = potential_loss + dp[(r+1, c)]
                else:
                    # General case: Choose path with minimum potential loss
                    # Can either move down or right
                    dp[(r, c)] = potential_loss + min(dp[(r+1, c)], dp[(r, c+1)])
        
        # If thief already encountered, return computed minimum potential loss
        if has_thief:
            return dp[(row, col)]
        
        # future loss estimation
        # Filter and sort future thieves by Manhattan distance to current position
        # This prioritizes closest thieves which are more likely to be encountered first
        future_thieves = sorted(
            [(tr, tc) for tr, tc in self._thief_locations if tr >= row and tc >= col],
            key=lambda pos: abs(pos[0] - row) + abs(pos[1] - col)
        )
        
        # Double-check that there are future thieves (should already be covered above)
        if not future_thieves:
            return 0
        
        # Simplified future loss estimation ---
        # Instead of complex calculations for each thief, focus on closest thief
        # Assumption: The closest thief is most likely to be encountered first
        thief_r, thief_c = future_thieves[0]  # Get coordinates of closest thief
        future_loss = 0
        
        # Sum up all coins that could potentially be lost after encountering this thief
        # These are coins that are at or after the thief's position
        for (r, c), value in self._coin_positions.items():
            if r >= thief_r and c >= thief_c:
                future_loss += value
        
        # Return the minimum of:
        # 1. Estimated future loss if we encounter the closest thief
        # 2. The DP-computed loss which assumes thief already encountered
        # This gives us a more balanced estimate based on two approaches
        return min(future_loss, dp[(row, col)])

    def cost_2(self, state):
        # Cost function for phase 3 (minimize stolen coins)
        return state.state[4]  # stolen_coin

    def A_star(self):
        """
        Core A* search algorithm that uses different heuristics depending on the phase:
        - Phase 2: Maximize collected coins (profit)
        - Phase 3: Minimize stolen coins (loss)
        """
        state_counter = 0  # Counter to keep track of how many states were expanded

        # Select heuristic and cost functions based on the phase
        if self.phase == 2:
            h = self.heuristic_1
            g = self.cost_1
            sign = -1  # Maximization problem (minimize negative profit)
        elif self.phase == 3:
            h = self.heuristic_2
            g = self.cost_2
            sign = 1   # Minimization problem

        # Initialize the start state
        initial_state = State(0, 0, self.grid[0][0], None, None)
        goal = (self.n - 1, self.n - 1)

        # Allowed moves: right and down
        action_list = {
            'down': (1, 0),
            'right': (0, 1),
        }

        # Priority queue (fringe) ordered by f(n) = g(n) + h(n)
        fringe = PriorityQueue()
        initial_f = (h(initial_state) + g(initial_state)) * sign
        fringe.put(Priority(initial_f, initial_state))

        # A* main loop
        while not fringe.empty():
            current_state = fringe.get().state
            current_i = current_state.state[0]
            current_j = current_state.state[1]

            # If goal is reached, return solution state and number of expanded states
            if tuple(current_state.state[0:2]) == goal:
                return current_state, state_counter
            
            # Explore all possible actions (down, right)
            for action in action_list:
                i = action_list[action][0]
                j = action_list[action][1]
                new_i = current_i + i
                new_j = current_j + j

                # Check boundaries
                if new_i >= self.n or new_j >= self.n:
                    continue

                # Create new state based on current action
                new_state = State(new_i, new_j, self.grid[new_i][new_j], current_state, action)

                # Calculate f(n) for new state
                f_n = (h(new_state) + g(new_state)) * sign

                # Count and add to fringe
                state_counter += 1
                fringe.put(Priority(f_n, new_state))

        # If no solution found
        return None, state_counter



# other options for heuristic functions

'''
class Astar:
    """
    A* search implementation for grid-based pathfinding with thief mechanics.
    
    This class implements an A* search algorithm for navigating a grid from top-left
    to bottom-right, while handling special "thief" mechanics that can steal coins or
    cause penalties. The algorithm uses precomputed dynamic programming tables as
    admissible heuristics.
    """
    def __init__(self, n, grid, phase):
        """
        Initialize the A* search algorithm with grid parameters.
        
        Args:
            n (int): Size of the n×n grid
            grid (list): 2D grid representation with integers (coins/penalties) and '!' (thief) cells
            phase (int): Determines which heuristic/cost function to use (2 or 3)
        """
        self.n = n
        self.grid = grid
        self.phase = phase

        # Precompute heuristics using dynamic programming for efficiency
        self.profit_dp = self._compute_max_profit_dp()  # Max profit from any position to goal
        self.min_loss_dp = self._compute_min_loss_dp()  # Min loss from any position to goal

    
    def _compute_max_profit_dp(self):
        """
        Bottom-up DP tracking two states: no thief / with thief.
        
        Computes the maximum net profit achievable from any cell to the goal,
        considering the thief mechanics.
        
        Returns two n×n matrices:
          dp_no[r][c]: max net coins from (r,c) to goal entering WITHOUT a thief.
          dp_th[r][c]: max net coins entering WITH a thief.
        
        Rules mirror simulate_path behavior exactly.
        """
        n = self.n
        grid = self.grid
        # Initialize DP arrays with negative infinity (we'll be taking maximums)
        dp_no = [[float('-inf')] * n for _ in range(n)]
        dp_th = [[float('-inf')] * n for _ in range(n)]

        # Helper to get dp value at neighbor based on next_thief state
        def get_dp(nr, nc, has_thief):
            """Return the appropriate DP value based on thief state"""
            return dp_th[nr][nc] if has_thief else dp_no[nr][nc]

        # Iterate from bottom-right to top-left (since we're working backwards from goal)
        for r in range(n - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                cell = grid[r][c]
                # Consider both entering states (with and without thief)
                for has_thief in (False, True):
                    # Process current cell effect based on thief state
                    if has_thief:
                        # Thief on board - special handling
                        if cell == '!':
                            # Thief disappears when hitting another thief cell
                            gain = 0
                            next_thief = False
                        else:
                            # Integer cell: positive stolen => no gain; negative costs twice
                            if cell > 0:
                                gain = 0  # Thief steals positive coins, so no gain
                            else:
                                gain = -2 * abs(cell)  # Penalties are doubled with thief
                            next_thief = False  # Thief disappears after this cell
                    else:
                        # No thief initially
                        if cell == '!':
                            gain = 0  # Thief cell itself has no value
                            next_thief = True  # But we pick up a thief
                        else:
                            gain = cell  # Normal gain/loss from cell
                            next_thief = False  # No thief state change
    
                    # Calculate best future value from possible transitions
                    if r == n - 1 and c == n - 1:
                        # At exit, no further moves possible
                        best_future = 0
                    else:
                        best_future = float('-inf')
                        # Try moving down or right
                        for dr, dc in ((1, 0), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < n and 0 <= nc < n:
                                val = get_dp(nr, nc, next_thief)
                                if val > best_future:
                                    best_future = val
    
                    # Total value = immediate gain + best future value
                    total = gain + best_future
                    
                    # Store in appropriate table based on thief state
                    if has_thief:
                        dp_th[r][c] = total
                    else:
                        dp_no[r][c] = total

        # Return both tables in a dictionary keyed by thief state
        return {False: dp_no, True: dp_th}
    
    def _compute_min_loss_dp(self):
        """
        Bottom-up DP tracking two states: no thief / with thief.
        
        Computes the minimum total coins that could be stolen from any cell
        to the goal, considering the thief mechanics.
        
        Returns two n×n matrices:
          dp_no[r][c]: minimal total coins _stolen_ from (r,c) to goal entering WITHOUT a thief.
          dp_th[r][c]: minimal total coins stolen entering WITH a thief.
          
        Rules mirror simulate_path's stealing behavior exactly.
        """
        n = self.n
        grid = self.grid

        # Initialize DP arrays to +∞ (we'll be taking minimums)
        dp_no = [[float('inf')] * n for _ in range(n)]
        dp_th = [[float('inf')] * n for _ in range(n)]

        # Helper to pick the right table based on thief state
        def get_dp(nr, nc, has_thief):
            """Return the appropriate DP value based on thief state"""
            return dp_th[nr][nc] if has_thief else dp_no[nr][nc]

        # Iterate from bottom-right to top-left (working backwards from goal)
        for r in range(n - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                cell = grid[r][c]

                # Consider both incoming‐states: has_thief False or True
                for has_thief in (False, True):
                    # Calculate immediate cost (theft) from this cell
                    if has_thief:
                        # Thief onboard behavior
                        if cell == '!':
                            cost = 0  # No theft at thief cell
                            next_thief = False  # Thief disappears
                        else:
                            # Thief steals abs(cell) whether treasure or penalty
                            cost = abs(cell) if isinstance(cell, int) else 0
                            next_thief = False  # Thief disappears after stealing
                    else:
                        # No thief on board initially
                        if cell == '!':
                            cost = 0  # No theft at thief cell
                            next_thief = True  # But we pick up a thief
                        else:
                            cost = 0  # No theft without thief
                            next_thief = False  # State remains unchanged
    
                    # Calculate best (minimum) future cost
                    if r == n - 1 and c == n - 1:
                        # At exit, no more theft beyond this cell
                        best_future = 0
                    else:
                        best_future = float('inf')
                        # Try moving down or right
                        for dr, dc in ((1, 0), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < n and 0 <= nc < n:
                                val = get_dp(nr, nc, next_thief)
                                if val < best_future:
                                    best_future = val
    
                    # Total cost = immediate cost + best future cost
                    total = cost + best_future
    
                    # Store result in appropriate table
                    if has_thief:
                        dp_th[r][c] = total
                    else:
                        dp_no[r][c] = total

        # Return both tables in a dictionary keyed by thief state
        return {False: dp_no, True: dp_th}

    def heuristic_1(self, state):
        """
        Realistic profit heuristic respecting future thief effects.
        
        Uses precomputed DP table to estimate maximum possible profit
        from current state to goal.
        
        Args:
            state (State): Current search state
            
        Returns:
            float: Estimated maximum profit to goal
        """
        r, c = state.state[0], state.state[1]
        thief = state.state[6]  # Whether we currently have a thief
        return self.profit_dp[thief][r][c]
    
    def cost_1(self, state):
        """
        Cost function for phase 2 (maximizing profit).
        Returns accumulated cost so far (which is negative for A* to maximize).
        
        Args:
            state (State): Current search state
            
        Returns:
            float: Accumulated cost so far
        """
        return state.state[5]

    def heuristic_2(self, state):
        """
        Minimum loss heuristic respecting thief mechanics.
        
        Uses precomputed DP table to estimate minimum possible coins stolen
        from current state to goal.
        
        Args:
            state (State): Current search state
            
        Returns:
            float: Estimated minimum possible theft to goal
        """
        r, c = state.state[0], state.state[1]
        thief = state.state[6]  # Whether we currently have a thief
        return self.min_loss_dp[thief][r][c]
    
    def cost_2(self, state):
        """
        Cost function for phase 3 (minimizing theft).
        Returns accumulated theft so far.
        
        Args:
            state (State): Current search state
            
        Returns:
            float: Accumulated coins stolen so far
        """
        return state.state[4]

    def A_star(self):
        """
        Implements the A* search algorithm.
        
        Uses different heuristic and cost functions based on the phase:
        - Phase 2: Maximize profit (use heuristic_1, cost_1)
        - Phase 3: Minimize theft (use heuristic_2, cost_2)
        
        Returns:
            tuple: (final_state, state_counter) - the goal state and number of states explored
        """
        state_counter = 0  # Count states expanded for performance analysis
        
        # Select appropriate heuristic and cost functions based on phase
        if self.phase == 2:
            h, g, sign = self.heuristic_1, self.cost_1, -1  # Negate for maximization
        elif self.phase == 3:
            h, g, sign = self.heuristic_2, self.cost_2, 1  # Positive for minimization

        # Create initial state at top-left corner
        initial = State(0, 0, self.grid[0][0], None, None)
        goal = (self.n - 1, self.n - 1)  # Bottom-right is the goal
        
        # Define possible moves (down or right only)
        moves = {
                    'down': (1, 0),  # (row_change, col_change)
                    'right': (0, 1),
                }

        # Initialize priority queue for A* frontier
        fringe = PriorityQueue()
        fringe.put(Priority((h(initial) + g(initial)) * sign, initial))
        
        # Main A* search loop
        while not fringe.empty():
            curr = fringe.get().state  # Get state with best priority
            r, c = curr.state[0], curr.state[1]
            state_counter += 1  # Count expanded states
            
            # Check if we've reached the goal
            if (r, c) == goal:
                return curr, state_counter
                
            # Try each possible move (down, right)
            for act, (dr, dc) in moves.items():
                nr, nc = r + dr, c + dc  # Calculate new position
                
                # Skip invalid moves that would go off grid
                if nr >= self.n or nc >= self.n:
                    continue
                    
                # Create new state for this move
                new_state = State(nr, nc, self.grid[nr][nc], curr, act)
                
                # Calculate f-value (g + h) with appropriate sign for min/max
                f = (h(new_state) + g(new_state)) * sign
                
                # Add to priority queue
                fringe.put(Priority(f, new_state))
                
        # If queue empties without finding goal (shouldn't happen in a connected grid)
        return None, state_counter
'''


'''
    def heuristic_1(self, state):
       row, col = state.state[0], state.state[1]
       original_row, original_col = row, col

       # Get all coins from the current position to bottom-right.
       remaining_coins = [
           [self.grid[r][c], (r, c)]
           for r in range(row, self.n)
           for c in range(col, self.n)
           if isinstance(self.grid[r][c], int) and self.grid[r][c] > 0
       ]

       # Remove coins based on the down/right decision.
       while row < (self.n - 1) and col < (self.n - 1):
           sum_down = sum(
               coin[0] for coin in remaining_coins 
               if coin[1][0] > row and coin[1][1] == col
           )
           sum_right = sum(
               coin[0] for coin in remaining_coins 
               if coin[1][0] == row and coin[1][1] > col
           )
           if sum_down >= sum_right:
               remaining_coins = [coin for coin in remaining_coins if not (coin[1][0] == row and coin[1][1] > col)]
           else:
               remaining_coins = [coin for coin in remaining_coins if not (coin[1][1] == col and coin[1][0] > row)]
           row += 1
           col += 1

       # The total number of moves we can make:
       k = (self.n - 1 - original_row) + (self.n - 1 - original_col)

       # We now want to choose k coins (or coin positions) such that:
       # 1. For each row x from original_row to self.n-1, at least one coin is selected that is in row x.
       # 2. For each column y from original_col to self.n-1, at least one coin is selected that is in column y.
       # 3. Among selections satisfying (1) and (2), the overall sum is maximized.

       selected = []          # List to hold the selected coins.
       selected_positions = set()  # Set to quickly check if a position was already chosen.

       # for each row, pick the best coin available (if any)
       for r in range(original_row, self.n):
           # Get all coins in row r.
           row_coins = [coin for coin in remaining_coins if coin[1][0] == r]
           if row_coins:
               best_coin = max(row_coins, key=lambda coin: coin[0])
               if best_coin[1] not in selected_positions:
                   selected.append(best_coin)
                   selected_positions.add(best_coin[1])

       # column,least one coin.
       for c in range(original_col, self.n):
           # Check if any already-selected coin covers column c.
           if not any(coin[1][1] == c for coin in selected):
               col_coins = [coin for coin in remaining_coins if coin[1][1] == c]
               if col_coins:
                   best_coin = max(col_coins, key=lambda coin: coin[0])
                   if best_coin[1] not in selected_positions:
                       selected.append(best_coin)
                       selected_positions.add(best_coin[1])

       #how many more coins we can add
       slots_remaining = k - len(selected)
       if slots_remaining > 0:
           # Get a sorted list of all remaining coins (not yet selected) by descending value.
           extra_candidates = [coin for coin in remaining_coins if coin[1] not in selected_positions]
           extra_candidates_sorted = sorted(extra_candidates, key=lambda coin: coin[0], reverse=True)
           # Fill remaining slots with the highest value coins.
           selected.extend(extra_candidates_sorted[:slots_remaining])

       # The heuristic is the sum of all selected coin values.
       sum1 = sum(coin[0] for coin in selected)
       return sum1
'''
'''
    #  min loss
    def heuristic_2(self, state):
        row, col = state.state[0], state.state[1]
        current_thief = state.state[6]
        grid_copy = [r.copy() for r in self.grid]
        def simulate(r, c, has_thief, grid, moves_remaining):
            if moves_remaining == 0:
                return 0
            costs = []
            for move in ['down', 'right']:
                new_grid = [row.copy() for row in grid]
                if move == 'down':
                    nr, nc = r + 1, c
                else:
                    nr, nc = r, c + 1
                if nr < 0 or nr >= self.n or nc < 0 or nc >= self.n:
                    continue
                coins_stolen = 0
                cell = new_grid[nr][nc]
                if has_thief:
                    if cell == '!':
                        pass
                    elif isinstance(cell, int):
                        if cell > 0:
                            coins_stolen += cell
                            new_grid[nr][nc] = -1
                        elif cell < 0:
                            coins_stolen += abs(cell)
                            new_grid[nr][nc] = -1
                    new_has_thief = False
                else:
                    if cell == '!':
                        new_has_thief = True
                    else:
                        new_has_thief = False
                        if isinstance(cell, int):
                            if cell > 0:
                                new_grid[nr][nc] = -1
                future = simulate(nr, nc, new_has_thief, new_grid, moves_remaining - 1)
                costs.append(coins_stolen + future)
            if not costs:
                return 0
            return min(costs)
        return simulate(row, col, current_thief, grid_copy, 3)
    '''