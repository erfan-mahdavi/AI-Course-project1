from grid_simulator_informed import State, Priority, PriorityQueue

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

        # Helper: select correct DP table based on thief state
        def get_dp(nr, nc, has_thief):
            return dp_th[nr][nc] if has_thief else dp_no[nr][nc]

        # Fill tables from bottom-right toward top-left
        for r in range(n - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                cell = grid[r][c]
                # For each cell, consider both possible "has_thief" entering states
                for has_thief in (False, True):
                    # Determine immediate gain and next thief state
                    if has_thief:
                        # If a thief is on board:
                        if cell == '!':
                            gain = 0             # no direct gain
                            next_thief = False  # two thieves fight
                        else:
                            # robber steals treasure; on cost you pay twice
                            if cell > 0:
                                gain = 0
                            else:
                                gain = -2 * abs(cell)
                            next_thief = False
                    else:
                        # No thief on board:
                        if cell == '!':
                            gain = 0
                            next_thief = True   # pick up a thief
                        else:
                            gain = cell         # normal cell effect (can be neg)
                            next_thief = False

                    # If at exit cell, no future moves
                    if r == n - 1 and c == n - 1:
                        best_future = 0
                    else:
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
        """
        n    = self.n
        grid = self.grid
        # Initialize with +inf so we can take minimums
        dp_no = [[float('inf')] * n for _ in range(n)]
        dp_th = [[float('inf')] * n for _ in range(n)]

        # Helper to choose DP by thief state
        def get_dp(nr, nc, has_thief):
            return dp_th[nr][nc] if has_thief else dp_no[nr][nc]

        # Fill in reverse order
        for r in range(n - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                cell = grid[r][c]
                for has_thief in (False, True):
                    # Immediate stolen cost at this cell
                    if has_thief:
                        if cell == '!':
                            cost = 0
                            next_thief = False
                        else:
                            # thief steals abs(cell) if integer
                            cost = abs(cell) if isinstance(cell, int) else 0
                            next_thief = False
                    else:
                        if cell == '!':
                            cost = 0
                            next_thief = True
                        else:
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
        r, c = state.state[0], state.state[1]
        thief = state.state[6]  # whether a thief is currently in the car
        return self.profit_dp[thief][r][c]

    def cost_1(self, state):
        # Phase 2 cost g(n): coins collected so far (from parent into this state)
        parent = state.state[7]
        if parent is None:
            return 0
        return parent.state[5]

    def heuristic_2(self, state):
        # Phase 3 heuristic: exact min stolen coins from current cell to goal
        r, c = state.state[0], state.state[1]
        thief = state.state[6]
        return self.min_loss_dp[thief][r][c]

    def cost_2(self, state):
        # Phase 3 cost g(n): coins stolen so far (from parent into this state)
        parent = state.state[7]
        if parent is None:
            return 0
        return parent.state[4]

    def A_star(self):
        # Main A* search driver
        state_counter = 0
        # Select heuristic, cost, and sign based on phase
        if self.phase == 2:
            h, g, sign = self.heuristic_1, self.cost_1, -1
        elif self.phase == 3:
            h, g, sign = self.heuristic_2, self.cost_2, 1

        # Initialize start state at (0,0)
        initial = State(0, 0, self.grid[0][0], None, None)
        goal = (self.n - 1, self.n - 1)
        moves = {'down': (1, 0), 'right': (0, 1)}

        # Fringe is a priority queue ordered by f = g + h (with sign)
        fringe = PriorityQueue()
        fringe.put(Priority((h(initial) + g(initial)) * sign, initial))

        # Explore until fringe is empty or goal reached
        while not fringe.empty():
            curr = fringe.get().state
            r, c = curr.state[0], curr.state[1]
            state_counter += 1
            if (r, c) == goal:
                return curr, state_counter

            # Expand both possible moves: down and right
            for act, (dr, dc) in moves.items():
                nr, nc = r + dr, c + dc
                if nr >= self.n or nc >= self.n:
                    continue
                new_s = State(nr, nc, self.grid[nr][nc], curr, act)
                # Compute f-score and add to fringe
                f = (h(new_s) + g(new_s)) * sign
                fringe.put(Priority(f, new_s))

        # Return None if no valid path
        return None, state_counter




# other options for heuristic functions

'''


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