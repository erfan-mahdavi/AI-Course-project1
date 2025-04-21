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
    def __init__(self, n, grid, phase):
        # Initialize grid dimensions, map data, and search phase
        self.n = n                # Size of the grid (n x n)
        self.grid = grid          # 2D list containing ints (cost/treasure) or '!' for thieves
        self.phase = phase        # Phase of A*: 2 for maximizing profit, 3 for minimizing loss

        # Precompute two DP tables to use as admissible heuristics
        self.profit_dp = self._compute_profit_dp()      # Max net coins from any cell to goal
        self.min_loss_dp = self._compute_min_loss_dp()  # Min stolen coins from any cell to goal

        # Debug output to verify DP results
        print("exact dp compuatation for each point entering with or without thief for phase 2")
        print(self.profit_dp)
        print("exact dp compuatation for each point entering with or without thief for phase 3")
        print(self.min_loss_dp)

    def _compute_profit_dp(self):
        """
        Bottom-up DP to compute maximum net coins you can end up with
        from cell (r,c) to exit, in two scenarios: entering with or without a thief.
        Returns a dict: {False: dp_no, True: dp_th}, each an n×n table.
        """
        n = self.n
        grid = self.grid
        # Initialize DP tables with -inf so we can take maximums
        dp_no = [[float('-inf')] * n for _ in range(n)]  # entering without thief
        dp_th = [[float('-inf')] * n for _ in range(n)]  # entering with thief
        # Initialize DP tables with -inf so we can take maximums
        dp_no = [[float('-inf')] * n for _ in range(n)]  # entering without thief
        dp_th = [[float('-inf')] * n for _ in range(n)]  # entering with thief

        # Helper: select correct DP table based on thief state
        # Helper: select correct DP table based on thief state
        def get_dp(nr, nc, has_thief):
            return dp_th[nr][nc] if has_thief else dp_no[nr][nc]

        # Fill tables from bottom-right toward top-left
        # Fill tables from bottom-right toward top-left
        for r in range(n - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                cell = grid[r][c]
                # For each cell, consider both possible "has_thief" entering states
                # For each cell, consider both possible "has_thief" entering states
                for has_thief in (False, True):
                    # Determine immediate gain and next thief state
                    # Determine immediate gain and next thief state
                    if has_thief:
                        # If a thief is on board:
                        # If a thief is on board:
                        if cell == '!':
                            gain = 0             # no direct gain
                            next_thief = False  # two thieves fight
                            gain = 0             # no direct gain
                            next_thief = False  # two thieves fight
                        else:
                            # robber steals treasure; on cost you pay twice
                            # robber steals treasure; on cost you pay twice
                            if cell > 0:
                                gain = 0
                                gain = 0
                            else:
                                gain = -2 * abs(cell)
                            next_thief = False
                                gain = -2 * abs(cell)
                            next_thief = False
                    else:
                        # No thief on board:
                        # No thief on board:
                        if cell == '!':
                            gain = 0
                            next_thief = True   # pick up a thief
                            gain = 0
                            next_thief = True   # pick up a thief
                        else:
                            gain = cell         # normal cell effect (can be neg)
                            next_thief = False

                    # If at exit cell, no future moves
                            gain = cell         # normal cell effect (can be neg)
                            next_thief = False

                    # If at exit cell, no future moves
                    if r == n - 1 and c == n - 1:
                        best_future = 0
                    else:
                        # Otherwise consider right and down neighbors
                        # Otherwise consider right and down neighbors
                        best_future = float('-inf')
                        for dr, dc in ((1, 0), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < n and 0 <= nc < n:
                                val = get_dp(nr, nc, next_thief)
                                if val > best_future:
                                    best_future = val


                    total = gain + best_future
                    # Save into the appropriate DP table
                    # Save into the appropriate DP table
                    if has_thief:
                        dp_th[r][c] = total
                    else:
                        dp_no[r][c] = total

        return {False: dp_no, True: dp_th}


    def _compute_min_loss_dp(self):
        """
        Bottom-up DP to compute minimum total stolen coins from cell (r,c)
        to exit, in two scenarios: entering with or without thief.
        Returns dict {False: dp_no, True: dp_th}.
        Bottom-up DP to compute minimum total stolen coins from cell (r,c)
        to exit, in two scenarios: entering with or without thief.
        Returns dict {False: dp_no, True: dp_th}.
        """
        n    = self.n
        n    = self.n
        grid = self.grid
        # Initialize with +inf so we can take minimums
        # Initialize with +inf so we can take minimums
        dp_no = [[float('inf')] * n for _ in range(n)]
        dp_th = [[float('inf')] * n for _ in range(n)]

        # Helper to choose DP by thief state
        # Helper to choose DP by thief state
        def get_dp(nr, nc, has_thief):
            return dp_th[nr][nc] if has_thief else dp_no[nr][nc]

        # Fill in reverse order
        # Fill in reverse order
        for r in range(n - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                cell = grid[r][c]
                for has_thief in (False, True):
                    # Immediate stolen cost at this cell
                    # Immediate stolen cost at this cell
                    if has_thief:
                        if cell == '!':
                            cost = 0
                            next_thief = False
                            cost = 0
                            next_thief = False
                        else:
                            # thief steals abs(cell) if integer
                            # thief steals abs(cell) if integer
                            cost = abs(cell) if isinstance(cell, int) else 0
                            next_thief = False
                            next_thief = False
                    else:
                        if cell == '!':
                            cost = 0
                            next_thief = True
                            cost = 0
                            next_thief = True
                        else:
                            cost = 0
                            next_thief = False

                    # Compute best future stolen coins
                            cost = 0
                            next_thief = False

                    # Compute best future stolen coins
                    if r == n - 1 and c == n - 1:
                        best_future = 0
                    else:
                        best_future = float('inf')
                        for dr, dc in ((1, 0), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < n and 0 <= nc < n:
                                val = get_dp(nr, nc, next_thief)
                                if val < best_future:
                                    best_future = val


                    total = cost + best_future
                    if has_thief:
                        dp_th[r][c] = total
                    else:
                        dp_no[r][c] = total

        return {False: dp_no, True: dp_th}

    def heuristic_1(self, state):
        # Phase 2 heuristic: exact max net coins from current cell to goal
        # Phase 2 heuristic: exact max net coins from current cell to goal
        r, c = state.state[0], state.state[1]
        thief = state.state[6]  # whether a thief is currently in the car
        thief = state.state[6]  # whether a thief is currently in the car
        return self.profit_dp[thief][r][c]


    def cost_1(self, state):
        # Phase 2 cost g(n): coins collected so far (from parent into this state)
        parent = state.state[7]
        if parent is None:
            return 0
        return parent.state[5]
        # Phase 2 cost g(n): coins collected so far (from parent into this state)
        parent = state.state[7]
        if parent is None:
            return 0
        return parent.state[5]

    def heuristic_2(self, state):
        # Phase 3 heuristic: exact min stolen coins from current cell to goal
        # Phase 3 heuristic: exact min stolen coins from current cell to goal
        r, c = state.state[0], state.state[1]
        thief = state.state[6]
        thief = state.state[6]
        return self.min_loss_dp[thief][r][c]


    def cost_2(self, state):
        # Phase 3 cost g(n): coins stolen so far (from parent into this state)
        parent = state.state[7]
        if parent is None:
            return 0
        return parent.state[4]
        # Phase 3 cost g(n): coins stolen so far (from parent into this state)
        parent = state.state[7]
        if parent is None:
            return 0
        return parent.state[4]

    def A_star(self):
        # Main A* search driver
        state_counter = 0
        # Select heuristic, cost, and sign based on phase
        # Main A* search driver
        state_counter = 0
        # Select heuristic, cost, and sign based on phase
        if self.phase == 2:
            h, g, sign = self.heuristic_1, self.cost_1, -1
            h, g, sign = self.heuristic_1, self.cost_1, -1
        elif self.phase == 3:
            h, g, sign = self.heuristic_2, self.cost_2, 1
            h, g, sign = self.heuristic_2, self.cost_2, 1

        # Initialize start state at (0,0)
        # Initialize start state at (0,0)
        initial = State(0, 0, self.grid[0][0], None, None)
        goal = (self.n - 1, self.n - 1)
        moves = {'down': (1, 0), 'right': (0, 1)}
        goal = (self.n - 1, self.n - 1)
        moves = {'down': (1, 0), 'right': (0, 1)}

        # Fringe is a priority queue ordered by f = g + h (with sign)
        # Fringe is a priority queue ordered by f = g + h (with sign)
        fringe = PriorityQueue()
        fringe.put(Priority((h(initial) + g(initial)) * sign, initial))

        # Explore until fringe is empty or goal reached

        # Explore until fringe is empty or goal reached
        while not fringe.empty():
            curr = fringe.get().state
            curr = fringe.get().state
            r, c = curr.state[0], curr.state[1]
            state_counter += 1
            state_counter += 1
            if (r, c) == goal:
                return curr, state_counter

            # Expand both possible moves: down and right

            # Expand both possible moves: down and right
            for act, (dr, dc) in moves.items():
                nr, nc = r + dr, c + dc
                nr, nc = r + dr, c + dc
                if nr >= self.n or nc >= self.n:
                    continue
                new_s = State(nr, nc, self.grid[nr][nc], curr, act)
                # Compute f-score and add to fringe
                f = (h(new_s) + g(new_s)) * sign
                fringe.put(Priority(f, new_s))

        # Return None if no valid path
                new_s = State(nr, nc, self.grid[nr][nc], curr, act)
                # Compute f-score and add to fringe
                f = (h(new_s) + g(new_s)) * sign
                fringe.put(Priority(f, new_s))

        # Return None if no valid path
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