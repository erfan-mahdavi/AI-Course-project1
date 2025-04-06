from grid_simulator_informed import State,Priority,PriorityQueue


class Astar:
    def __init__(self,n,grid,phase):
        self.n = n
        self.grid = grid
        self.phase = phase

    def manhattan_distance(self, r1, c1, r2, c2):
        return abs(r2 - r1) + abs(c2 - c1)

    #  max profit
    def heuristic_1(self,state):
            row, col = state.state[0], state.state[1]
            has_thief = state.state[6]
            
            coin_values = {}
            for r in range(row, self.n):
                for c in range(col, self.n):
                    cell = self.grid[r][c]
                    if isinstance(cell, int) and cell > 0:
                        coin_values[(r, c)] = cell
            
            dp = [[0 for _ in range(self.n)] for _ in range(self.n)]
            
            for r in range(self.n-1, row-1, -1):
                for c in range(self.n-1, col-1, -1):
                    pos_value = coin_values.get((r, c), 0)
                    
                    if r == self.n-1 and c == self.n-1:
                        dp[r][c] = pos_value
                    elif r == self.n-1:
                        dp[r][c] = pos_value + dp[r][c+1]
                    elif c == self.n-1:
                        dp[r][c] = pos_value + dp[r+1][c]
                    else:
                        dp[r][c] = pos_value + max(dp[r+1][c], dp[r][c+1])
            
            if has_thief:
                return 0
            
            return dp[row][col]
    '''
     state format : (
        0         index_i,
        1         index_j,
        2         current_cost,
        3         current_profit,
        4         stolen_coin, 
        5         collected_coin,
        6         has_thief, 
        7         parent_state,
        8         action,
        9         list_of_actions,
     )
    '''

    def cost_1(self,state):
        return state.state[5]

    #  min loss
    def heuristic_2(self, state):
        row, col = state.state[0], state.state[1]
        has_thief = state.state[6]
        
        if row == self.n - 1 and col == self.n - 1:
            return 0
        

        if not has_thief:
            has_future_thief = False
            for r in range(row, self.n):
                for c in range(col, self.n):
                    if r == row and c == col:
                        continue  
                    if self.grid[r][c] == '!':
                        has_future_thief = True
                        break
                if has_future_thief:
                    break
            
            if not has_future_thief:
                return 0  
        
        dp = [[0 for _ in range(self.n + 1)] for _ in range(self.n + 1)]
        
        for r in range(self.n - 1, -1, -1):
            for c in range(self.n - 1, -1, -1):
                if r < row or (r == row and c < col):
                    continue
                    
                cell_value = self.grid[r][c]
                
                potential_loss = 0
                if isinstance(cell_value, int) and cell_value > 0 and has_thief:
                    potential_loss = cell_value
                    
                if r == self.n - 1 and c == self.n - 1:
                    dp[r][c] = potential_loss
                elif r == self.n - 1:
                    dp[r][c] = potential_loss + dp[r][c+1]
                elif c == self.n - 1:
                    dp[r][c] = potential_loss + dp[r+1][c]
                else:
                    dp[r][c] = potential_loss + min(dp[r+1][c], dp[r][c+1])
        

        if has_thief:
            return dp[row][col]

        min_future_loss = float('inf')
        
        thief_locations = []
        for r in range(row, self.n):
            for c in range(col, self.n):
                if self.grid[r][c] == '!':
                    thief_locations.append((r, c))
        
        if not thief_locations:
            return 0 
        
        for thief_r, thief_c in thief_locations:
            dist_to_thief = abs(thief_r - row) + abs(thief_c - col)
            
            if thief_r < row or thief_c < col:
                continue
                

            future_loss = 0
            for r in range(thief_r, self.n):
                for c in range(thief_c, self.n):
                    if r == thief_r and c == thief_c:
                        continue
                    cell = self.grid[r][c]
                    if isinstance(cell, int) and cell > 0:
                        future_loss += cell
                        break  
                
            min_future_loss = min(min_future_loss, future_loss)
        

        return min(min_future_loss, dp[row][col])

    def cost_2(self,state):
        return state.state[4]

    def A_star(self):
        state_counter=0
        if self.phase == 2:
            h = self.heuristic_1
            g = self.cost_1
            sign = -1
        elif self.phase == 3:
            h = self.heuristic_2                                          
            g = self.cost_2
            sign = 1
        
        initial_state = State(0,0,self.grid[0][0],None,None)
        goal = (self.n-1,self.n-1)
        action_list = {
            'down' : (1,0),
            'right' : (0,1),
        }
        fringe = PriorityQueue()  # min-first priority queue
        initial_f = (h(initial_state) + g(initial_state)) * sign
        fringe.put(Priority(initial_f, initial_state))

        while not fringe.empty():
            current_state = fringe.get().state
            current_i = current_state.state[0]
            current_j = current_state.state[1]
            if tuple(current_state.state[0:2]) == goal:
                return current_state,state_counter
            
            for action in action_list:
                i = action_list[action][0]
                j = action_list[action][1]
                new_i = current_i+i
                new_j = current_j+j
                if new_i>=self.n or new_j>=self.n:
                    continue
                new_state = State(new_i,new_j,self.grid[new_i][new_j],current_state,action)
                f_n = (h(new_state) + g(new_state))*sign
                state_counter+=1
                fringe.put(Priority(f_n,new_state))

        return None,state_counter


if __name__=='__main__':
    a = Astar(2,[['!',2],[3,4]],2)
    print(a.A_star())