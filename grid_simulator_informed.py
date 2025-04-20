from queue import PriorityQueue

class Priority:
    """
    A wrapper class for states with a priority value for use in a PriorityQueue.
    Used to determine which state to explore next in a search algorithm.
    """
    def __init__(self, priority, state):
        self.priority = priority  # The priority value for ordering in the queue
        self.state = state        # The actual state object being prioritized
   
    def __lt__(self, other):
        """
        Defines the less-than comparison operator for Priority objects.
        This allows the PriorityQueue to order items correctly.
        Using <= instead of < ensures stability in the priority queue.
        """
        return self.priority <= other.priority


class State:
    """
    Represents a state in a game/puzzle that involves navigating a grid,
    collecting coins, and potentially dealing with thieves.
    
    State format: (
        index_i,             # Row position in the grid
        index_j,             # Column position in the grid
        current_cost,        # Accumulated cost so far
        current_profit,      # Accumulated profit so far
        stolen_coin,         # Amount of coins stolen
        collected_coin,      # Total coins collected
        has_thief,           # Boolean flag if the current position has a thief
        parent_state,        # Reference to the previous state
        action,              # Action taken to reach this state
        list_of_actions,     # History of all actions taken to reach this state
    )
    """
    def __init__(self, i, j, value, parent_state, action):
        """
        Initialize a new state based on grid position, cell value, parent state and action taken.
        
        Args:
            i, j: Grid coordinates
            value: The value at position (i,j):
                   - Positive: coin to collect
                   - Negative: cost/penalty
                   - '!': thief
            parent_state: The previous state
            action: The action taken to reach this state
        """
        if i == 0 and j == 0:  # Initial starting position
            self.state = (
                0,                                    # Starting row index
                0,                                    # Starting column index
                (abs(value) if value < 0 else 0) if value != '!' else 0,  # Initial cost
                (value if value > 0 else 0) if value != '!' else 0,       # Initial profit
                0,                                    # No stolen coins yet
                value if value != '!' else 0,         # Initial coins collected
                True if value == '!' else False,      # Whether starting position has a thief
                parent_state,                         # No parent for starting state
                action,                               # No action for starting state
                [],                                   # Empty action history
            )
        else:  # Non-starting positions
            # Calculate new state values based on the cell value and parent state
            if value == '!':
                current_cost, current_profit, stolen_coin, collected_coin, has_thief = self.thief(value, parent_state)
            elif value >= 0:
                current_cost, current_profit, stolen_coin, collected_coin, has_thief = self.positive_value(value, parent_state)
            else:
                current_cost, current_profit, stolen_coin, collected_coin, has_thief = self.negetive_value(value, parent_state)
            
            # Create the new state tuple
            self.state = (
                i,                                  # Current row position
                j,                                  # Current column position
                current_cost,                       # Updated cost
                current_profit,                     # Updated profit
                stolen_coin,                        # Updated stolen coins
                collected_coin,                     # Updated collected coins
                has_thief,                          # Updated thief status
                parent_state,                       # Reference to parent state
                action,                             # Action that led to this state
                parent_state.state[9] + [action],   # Updated action history
            )
       
    def thief(self, value, parent_state):
        """
        Handle the logic when the current cell contains a thief.
        If the player already has a thief, the new thief doesn't affect the player.
        Otherwise, the player acquires a thief.
        
        Returns: (current_cost, current_profit, stolen_coin, collected_coin, has_thief)
        """
        if parent_state.state[6]:
            # Already has a thief, nothing changes except we lose the thief
            return parent_state.state[2], parent_state.state[3], parent_state.state[4], parent_state.state[5], False
        else:
            # Gain a thief, but other values remain the same
            return parent_state.state[2], parent_state.state[3], parent_state.state[4], parent_state.state[5], True
    
    def positive_value(self, value, parent_state):
        """
        Handle the logic when the current cell contains a positive value (coin).
        If the player has a thief, the thief steals the coin.
        Otherwise, the player collects the coin.
        
        Returns: (current_cost, current_profit, stolen_coin, collected_coin, has_thief)
        """
        if parent_state.state[6]:
            # Has a thief - thief steals the coin
            return parent_state.state[2], parent_state.state[3] + value, parent_state.state[4] + value, parent_state.state[5], False
        else:
            # No thief - player collects the coin
            return parent_state.state[2], parent_state.state[3] + value, parent_state.state[4], parent_state.state[5] + value, False
       
    def negetive_value(self, value, parent_state):
        """
        Handle the logic when the current cell contains a negative value (penalty/cost).
        If the player has a thief, the player pays the cost and the thief steals additional coins.
        Otherwise, the player just pays the cost.
        
        Returns: (current_cost, current_profit, stolen_coin, collected_coin, has_thief)
        """
        if parent_state.state[6]:
            # Has a thief - pay cost and thief steals more coins (double penalty)
            return parent_state.state[2] + abs(value), parent_state.state[3], parent_state.state[4] + abs(value), parent_state.state[5] - 2 * abs(value), False
        else:
            # No thief - just pay the cost
            return parent_state.state[2] + abs(value), parent_state.state[3], parent_state.state[4], parent_state.state[5] - abs(value), False