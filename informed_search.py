from grid_simulator_informed import State, Priority, PriorityQueue


class Astar:
    def __init__(self, n, grid, phase):
        self.n = n
        self.grid = grid
        self.phase = phase

        # precompute heuristics
        self.profit_dp = self._compute_profit_dp()
        self.min_loss_dp = self._compute_min_loss_dp()

    
    def _compute_profit_dp(self):
        """
        Bottom-up DP tracking two states: no thief / with thief.
        Returns two n×n matrices:
          dp_no[r][c]: max net coins from (r,c) to goal entering WITHOUT a thief.
          dp_th[r][c]: max net coins entering WITH a thief.
        Rules mirror simulate_path behavior exactly.
         """
        n = self.n
        grid = self.grid
        # initialize DP arrays
        dp_no = [[float('-inf')] * n for _ in range(n)]
        dp_th = [[float('-inf')] * n for _ in range(n)]

        # helper to get dp value at neighbor based on next_thief state
        def get_dp(nr, nc, has_thief):
            return dp_th[nr][nc] if has_thief else dp_no[nr][nc]

        # iterate from bottom-right to top-left
        for r in range(n - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                cell = grid[r][c]
                # we consider both entering states
                for has_thief in (False, True):
                    # process current cell effect from 2 possibilities
                    if has_thief:
                        # thief on board 
                        if cell == '!':
                            gain = 0
                            next_thief = False
                        else:
                            # integer cell: positive stolen => no gain; negative costs twice
                            if cell > 0:
                                gain = 0
                            else:
                                gain = -2 * abs(cell)
                            next_thief = False
                    else:
                        # no thief
                        if cell == '!':
                            gain = 0
                            next_thief = True
                        else:
                            gain = cell
                            next_thief = False
    
                    # transition to neighbors
                    if r == n - 1 and c == n - 1:
                        # at exit, no further moves
                        best_future = 0
                    else:
                        best_future = float('-inf')
                        for dr, dc in ((1, 0), (0, 1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < n and 0 <= nc < n:
                                val = get_dp(nr, nc, next_thief)
                                if val > best_future:
                                    best_future = val
    
                    total = gain + best_future
                    
                    if has_thief:
                        dp_th[r][c] = total
                    else:
                        dp_no[r][c] = total

        return {False: dp_no, True: dp_th}
    
    def _compute_min_loss_dp(self):
        """
        Bottom-up DP tracking two states: no thief / with thief.
        Returns two n×n matrices:
          dp_no[r][c]: minimal total coins _stolen_ from (r,c) to goal entering WITHOUT a thief.
          dp_th[r][c]: minimal total coins stolen entering WITH a thief.
        Rules mirror simulate_path's stealing behavior exactly.
        """
        n    = self.n
        grid = self.grid

        # initialize DP arrays to +∞ (we’ll be taking mins)
        dp_no = [[float('inf')] * n for _ in range(n)]
        dp_th = [[float('inf')] * n for _ in range(n)]

        # helper to pick the right table
        def get_dp(nr, nc, has_thief):
            return dp_th[nr][nc] if has_thief else dp_no[nr][nc]

        # iterate from bottom-right to top-left
        for r in range(n - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                cell = grid[r][c]

                # consider both incoming‐states: has_thief False or True
                for has_thief in (False, True):
                    #  immediate cost from this cell:
                    if has_thief:
                        # thief onboard steals:
                        if cell == '!':
                            cost = 0
                            next_thief = False
                        else:
                            # steals abs(cell) whether treasure or penalty
                            cost = abs(cell) if isinstance(cell, int) else 0
                            next_thief = False
                    else:
                        # no thief: only change state if we hit a '!'
                        if cell == '!':
                            cost = 0
                            next_thief = True
                        else:
                            cost = 0
                            next_thief = False
    
                    #  best future cost (down or right)
                    if r == n - 1 and c == n - 1:
                        # at exit, no more stealing beyond this cell
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
        # realistic profit heuristic respecting future thief effects
        r, c = state.state[0], state.state[1]
        thief = state.state[6]
        return self.profit_dp[thief][r][c]
    
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

    def cost_1(self, state):
        return state.state[5]

    def heuristic_2(self, state):
        r, c = state.state[0], state.state[1]
        thief = state.state[6]
        return self.min_loss_dp[thief][r][c]
    
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

    def cost_2(self, state):
        return state.state[4]

    def A_star(self):
        state_counter = 0
        if self.phase == 2:
            h, g, sign = self.heuristic_1, self.cost_1, -1
        elif self.phase == 3:
            h, g, sign = self.heuristic_2, self.cost_2, 1

        initial = State(0, 0, self.grid[0][0], None, None)
        goal = (self.n - 1, self.n - 1)
        moves = {'down': (1, 0), 'right': (0, 1)}

        fringe = PriorityQueue()
        fringe.put(Priority((h(initial) + g(initial)) * sign, initial))
        while not fringe.empty():
            curr = fringe.get().state
            r, c = curr.state[0], curr.state[1]
            state_counter += 1
            if (r, c) == goal:
                return curr, state_counter
            for act, (dr, dc) in moves.items():
                nr, nc = r + dr, c + dc
                if nr >= self.n or nc >= self.n:
                    continue
                new_s = State(nr, nc, self.grid[nr][nc], curr, act)
                f = (h(new_s) + g(new_s)) * sign
                fringe.put(Priority(f, new_s))
        return None, state_counter