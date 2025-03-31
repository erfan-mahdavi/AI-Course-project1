from grid_simulator_informed import State,Priority,PriorityQueue


class Astar:
    def __init__(self,n,grid,phase):
        self.n = n
        self.grid = grid
        self.phase = phase

    #  max profit
    
    def heuristic_1(self,state):
        row, col = state.state[0], state.state[1]
        original_row = row
        original_col = col
        remaining_coins = [
            [self.grid[r][c],(r,c)] for r in range(row, self.n) for c in range(col, self.n)
            if isinstance(self.grid[r][c], int) and self.grid[r][c] > 0
        ]
        while (row<(self.n-1)) and  (col<(self.n-1)):
            sum_down = sum([i[0] for i in remaining_coins if ( (i[1][0]>row)and(i[1][1]==col)) ])
            sum_right = sum([i[0] for i in remaining_coins if ( (i[1][0]==row)and(i[1][1]>col)) ])
            if sum_down >= sum_right:
                remaining_coins = [i for i in remaining_coins if not((i[1][0]==row)and (i[1][1]>col))]
            else:
                remaining_coins = [i for i in remaining_coins if not((i[1][1]==col)and (i[1][0]>row))]
            row+=1
            col+=1
                
        sorted_list = sorted(remaining_coins, key=lambda x: x[0], reverse=True)[:(self.n - 1 - original_row + self.n - 1 - original_col)]
        
        sum1 = 0
        for i in sorted_list:
            sum1 += i[0]
        #print(sorted_list)
        return sum1
    '''
    def heuristic_1(self,state):
        row, col = state.state[0], state.state[1]
        remaining_coins = [
            self.grid[r][c] for r in range(row, self.n) for c in range(col, self.n)
            if isinstance(self.grid[r][c], int) and self.grid[r][c] > 0
        ]
        return sum(sorted(remaining_coins, reverse=True)[:(self.n - 1 - row + self.n - 1 - col)])
    '''
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

    def cost_1(self,state):
        return state.state[5]

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
        # fringe.put(Priority(1,initial_state))
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
                # print(f'i:{new_i},  j:{new_j},  grid[i][j]:{self.grid[new_i][new_j]},  f_n : {f_n}, (h,g) : ({h(new_state) },{ g(new_state)})')
                state_counter+=1
                #print(state_counter)
                fringe.put(Priority(f_n,new_state))

        return None,state_counter


if __name__=='__main__':
    a = Astar(2,[['!',2],[3,4]],2)
    print(a.A_star())