from queue import PriorityQueue

class Priority:
    def __init__(self, priority, state):
        self.priority = priority
        self.state = state
    
    def __lt__(self, other):
        return self.priority <= other.priority

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
        #                     list_of_actions,
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
                    [],
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
                    parent_state.state[9] + [action],
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

    #  max profit
    def heuristic_1(self,state):
        return 1

    def cost_1(self,state):
        return state.state[5]

    #  min loss
    def heuristic_2(self,state):
        ...

    def cost_2(self,state):
        return state.state[4]

    def A_star(self):
        if self.phase == 2:
            h = self.heuristic_1
            g = self.cost_1
        elif self.phase == 3:
            h = self.heuristic_2
            g = self.cost_2
        
        initial_state = State(0,0,self.grid[0][0],None,None)
        goal = (self.n-1,self.n-1)
        action_list = {
            'down' : (1,0),
            'right' : (0,1),
        }
        fringe = PriorityQueue()
        fringe.put(Priority(1,initial_state))

        while not fringe.empty():
            current_state = fringe.get().state
            current_i = current_state.state[0]
            current_j = current_state.state[1]
            if tuple(current_state.state[0:2]) == goal:
                return current_state
            for action in action_list:
                i = action_list[action][0]
                j = action_list[action][1]
                new_i = current_i+i
                new_j = current_j+j
                if new_i>=self.n or new_j>=self.n:
                    continue
                new_state = State(new_i,new_j,self.grid[new_i][new_j],current_state,action)
                f_n = h(current_state) + g(current_state)
                fringe.put(Priority(f_n,new_state))

        return 'there is no answer'


if __name__=='__main__':
    a = Astar(2,[['!',2],[3,4]],2)
    print(a.A_star())