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
                    value if value!='!' else 0,
                    True if value=='!' else False,
                    parent_state,
                    action,
                    [],
                )
            else:
                if value=='!':
                    current_cost, current_profit, stolen_coin, collected_coin, has_thief = self.thief(value,parent_state)
                elif value>=0:
                    current_cost, current_profit, stolen_coin, collected_coin, has_thief = self.positive_value(value,parent_state)
                else:
                    current_cost, current_profit, stolen_coin, collected_coin, has_thief = self.negetive_value(value,parent_state)

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