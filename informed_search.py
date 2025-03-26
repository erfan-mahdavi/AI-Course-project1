from queue import PriorityQueue

class State:
        # state format : (
        #                     index_i,
        #                     index_j,
        #                     current_cost,
        #                     current_profit,
        #                     stolen_coin, 
        #                     collected_coin,
        #                     has_thief, 
        #                     parent_state,
        #                     action,
        #                )


        def __init__(self,i,j,value,parent_state,action):
            if i==0 and j==0:
                self.state = (
                    0,
                    0,
                    (abs(value) if value<0 else 0) if value!='!' else 0,
                    (value if value>0 else 0) if value!='!' else 0,
                    0,
                    0,
                    True if value=='!' else False,
                    parent_state,
                    action,
                )
            else:
                if value=='!':
                    current_cost, current_profit, has_thief, stolen_coin, collected_coin = self.thief(value,parent_state)
                elif value>=0:
                    current_cost, current_profit, has_thief, stolen_coin, collected_coin = self.positive_value(value,parent_state)
                else:
                    current_cost, current_profit, has_thief, stolen_coin, collected_coin = self.negetive_value(value,parent_state)

                self.state = (
                    i,
                    j,
                    current_cost,
                    current_profit,
                    stolen_coin,
                    collected_coin,
                    has_thief,
                    parent_state,
                    action,
                )
        
        def thief(self,value,parent_state):
            if parent_state.state[6]:
                return parent_state.state[2], parent_state.state[3], parent_state.state[4], parent_state.state[5], False
            else:
                return parent_state.state[2], parent_state.state[3], parent_state.state[4], parent_state.state[5], True

        def positive_value(self,value,parent_state):
            if parent_state.state[6]:
                return parent_state.state[2], parent_state.state[3]+value , parent_state.state[4]+value , parent_state.state[5], False
            else:
                return parent_state.state[2], parent_state.state[3]+value , parent_state.state[4] , parent_state.state[5]+value , False
        
        def negetive_value(self,value,parent_state):
            if parent_state.state[6]:
                return parent_state.state[2]+abs(value) , parent_state.state[3] , parent_state.state[4]+abs(value) , parent_state.state[5]-2*abs(value), False
            else:
                return parent_state.state[2]+abs(value) , parent_state.state[3] , parent_state.state[4] , parent_state.state[5]-abs(value), False



class Astar:
    def __init__(self,n,grid,phase):
        self.n = n
        self.grid = grid
        self.phase = phase

    def heuristic_1(self):
        ...

    def cost_1(self):
        ...

    def heuristic_2(self):
        ...

    def cost_2(self):
        ...

    def A_star(self):
        if self.phase == 2:
            h = self.heuristic_1
            g = self.cost_1
        elif self.phase == 3:
            h = self.heuristic_2
            g = self.cost_2
        
        

        initial_state = (
            0,
            0,
            (abs(self.grid[0][0]) if self.grid[0][0]<0 else 0) if self.grid[0][0]!='!' else 0,
            (self.grid[0][0] if self.grid[0][0]>0 else 0) if self.grid[0][0]!='!' else 0,
            True if self.grid[0][0]=='!' else False,
            None,
            None,
            )
        goal = (self.n-1,self.n-1)



if __name__=='__main__':
    a = Astar(2,[['!',2],[3,4]],2)
    a.A_star()