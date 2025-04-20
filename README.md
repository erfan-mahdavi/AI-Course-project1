# Grid Traversal Optimization using BFS and A* Search

**Authors:** Erfan Mahdavi, Taraneh Kordi

## Overview

This project explores pathfinding algorithms on a specialized 2D grid environment. The goal is to navigate from a starting point (0,0) to an end point (n-1, n-1) using only 'down' and 'right' moves. The grid contains cells with costs, treasures, and interactive thieves, adding complexity to the traversal.

The project implements and compares three approaches:
1.  **Uninformed Search (BFS):** Finds the shortest path in terms of the number of steps.
2.  **Informed Search (A*) - Max Profit:** Finds the path that maximizes the net coins collected, considering costs, treasures, and thief interactions.
3.  **Informed Search (A*) - Min Loss:** Finds the path that minimizes the total value of coins stolen by thieves.

## Problem Description

The environment is an n x n grid where each cell can be one of the following:

* **Normal Cells (Negative Integers):** Represent a cost. Passing through subtracts the absolute value of the integer from the player's coin total.
* **Treasure Cells (Positive Integers):** Represent a treasure. Passing through adds the integer value to the player's coin total. Collected treasures are effectively removed (handled implicitly in DP/State or explicitly in simulation).
* **Thief Cells ('!'):** Contain thieves with specific behavior:
    * **Entering '!' without a thief:** The player picks up a thief. No immediate coin change occurs. The thief accompanies the player to the *next* cell.
    * **Entering *any* cell with a thief:**
        * **Next cell is '!'**: The accompanying thief and the new thief 'fight'. The accompanying thief leaves, and no coins are exchanged or stolen at this step. The player does not pick up the new thief.
        * **Next cell is Treasure (>0)**: The thief steals the *entire* treasure value from that cell. The player gains 0 from the cell. The thief leaves.
        * **Next cell is Normal (<0)**: The thief steals coins equal to the *absolute value* of the cell's cost *in addition* to the normal cost incurred by the player. Effectively, the player loses *twice* the absolute value of the cell. The thief leaves.
    * A thief always leaves after their effect (or non-effect in the '!' vs '!' case) in the cell *after* the one where they were picked up.

**Movement:** Allowed moves are strictly 'down' and 'right'. The path starts at `(0,0)` and ends at `(n-1, n-1)`.

## Project Goals

1.  **Phase 1: Uninformed Search (BFS)**
    * Find the path from start to goal with the minimum number of steps, ignoring costs, treasures, and thieves during the search itself (though the final path can be simulated later).
2.  **Phase 2: Informed Search (A* for Max Profit)**
    * Find the path that results in the maximum possible `collected_coin` value at the goal, considering all rules.
3.  **Phase 3: Informed Search (A* for Min Loss)**
    * Find the path that minimizes the total `stolen_coin` value accumulated due to thief actions.

## Implementation Details

### Algorithms

1.  **Breadth-First Search (BFS)** (`BFSSolver` class):
    * Implemented in `BFSSolver.py`.
    * Uses a `deque` for the queue and a `set` for visited states `(i, j)`.
    * Explores the grid layer by layer, guaranteeing the shortest path in terms of moves.
    * Returns the sequence of moves ('down', 'right') and the number of states explored.

2.  **A* Search** (`Astar` class):
    * Implemented in `Astar.py`.
    * Uses a `PriorityQueue` (`grid_simulator_informed.PriorityQueue`) to manage the fringe, prioritizing states based on `f = g + h`.
    * The definition of `g` (cost-so-far) and `h` (heuristic estimate) depends on the optimization phase (`phase` parameter).
    * Handles maximization (profit) by multiplying the priority `f` by `-1` before inserting into the min-priority queue.



## Core Components

### 1. State Management (`grid_simulator_informed.py`)

This file defines the core data structures for representing states in the search algorithms:

#### `Priority` Class

Wrapper class for states with priority values for use in priority queues:

```python
class Priority:
    def __init__(self, priority, state):
        self.priority = priority  # Priority value for ordering
        self.state = state        # State object being prioritized
   
    def __lt__(self, other):
        # Less-than comparison for PriorityQueue ordering
        return self.priority <= other.priority
```

#### `State` Class

Represents a state in the grid environment:

```python
class State:
    def __init__(self, i, j, value, parent_state, action):
        # State initialization with position, cell value, parent state and action
        # ...
```

State format is a tuple containing:
- `index_i`, `index_j`: Grid position (row, column)
- `current_cost`: Accumulated cost
- `current_profit`: Accumulated profit
- `stolen_coin`: Amount of coins stolen
- `collected_coin`: Total coins collected
- `has_thief`: Boolean flag for thief status
- `parent_state`: Reference to previous state
- `action`: Action taken to reach this state
- `list_of_actions`: History of actions

Special methods handle different cell types:
- `thief()`: Handles thief cells (toggling thief status)
- `positive_value()`: Handles coin collection (affected by thief status)
- `negetive_value()`: Handles penalties (affected by thief status)

### 2. Grid Simulation (`grid_simulator_uninformed.py`)

Contains the `GridSimulator` class which simulates grid traversal based on a given path:

```python
class GridSimulator:
    def __init__(self, grid):
        self.original_grid = [row.copy() for row in grid]
        self.n = len(grid)

    def simulate_path(self, path):
        # Simulates following a path through the grid
        # Updates tracking variables (coins, thieves) as cells are traversed
        # Returns final statistics
```

The simulator maintains the game state by tracking:
- Current position
- Coins collected
- Coins stolen
- Thief status

### 3. Uninformed Search (`uninformed_search.py`)

Implements Breadth-First Search for finding the shortest path:

```python
class BFSSolver:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        
    def find_shortest_path(self):
        # BFS implementation that finds shortest path from (0,0) to (grid_size-1, grid_size-1)
        # Returns path and number of states explored
```

BFS is used for Phase 1, seeking only to reach the exit with the minimum number of moves.

### 4. Informed Search (`informed_search.py`)

Implements A* search with different heuristics:

```python
class Astar:
    def __init__(self, n, grid, phase):
        self.n = n
        self.grid = grid
        self.phase = phase
        # Precompute heuristics using dynamic programming
        self.profit_dp = self._compute_max_profit_dp()  
        self.min_loss_dp = self._compute_min_loss_dp()  
```

Key components:
- Dynamic programming tables for heuristic computation
- Different heuristic and cost functions for different objectives
- A* implementation handling thief mechanics

Phases:
- **Phase 2**: Maximize profit using `heuristic_1` and `cost_1`
- **Phase 3**: Minimize theft using `heuristic_2` and `cost_2`

### 5. Visualization and UI (`main_file_graphical.py`)

Provides a GUI interface for:
- Creating/loading grids
- Running different algorithms
- Visualizing results

Main components:
- `GridApp`: Initial grid setup and configuration
- `Menu`: Algorithm selection and result visualization
- Utility functions for grid generation and file I/O

## Gameplay and Mechanics

### Grid Structure
- Square grid with size n×n
- Movement restricted to down and right only
- Start position: (0,0)
- Goal position: (n-1, n-1)

### Cell Types
- **Positive integers**: Coins that can be collected
- **Negative integers**: Penalty/cost cells
- **'!'**: Thief cells

### Thief Mechanics
When a player has a thief:
- Positive value cells: Thief steals the coins (no collection)
- Negative value cells: Player pays the cost and thief steals additional coins (double penalty)
- Thief disappears after visiting one cell

When a player doesn't have a thief:
- Positive value cells: Player collects the coins
- Negative value cells: Player just pays the cost
- Encountering a thief cell gives the player a thief

## Algorithms

### Breadth-First Search (BFS)
- Used for Phase 1 (exit-only)
- Guarantees shortest path to exit
- Ignores coin collection/theft mechanics

### A* Search
- Used for Phases 2 and 3
- Informed search with admissible heuristics
- Pre-computed dynamic programming tables for efficiency

The `Astar` class defines cost (`g`) and heuristic (`h`) functions based on the phase:

* **Phase 2 (Max Profit):**
    * `g = cost_1(state)`: Returns `state.state[5]` (current `collected_coin`). This represents the "cost so far" in terms of profit achieved.
    * `h = heuristic_1(state)`: Returns `self.profit_dp[thief][r][c]`, the precomputed maximum *future* profit from the current cell `(r, c)` given the current `thief` status.
    * Priority: `f = (g + h) * -1`. We maximize `g+h` (total possible profit) using a min-priority queue by negating the value.

* **Phase 3 (Min Loss):**
    * `g = cost_2(state)`: Returns `state.state[4]` (current `stolen_coin`). This is the cost (loss) accumulated so far.
    * `h = heuristic_2(state)`: Returns `self.min_loss_dp[thief][r][c]`, the precomputed minimum *future* loss from the current cell `(r, c)` given the current `thief` status.
    * Priority: `f = (g + h) * 1`. We minimize `g+h` (total loss).

#### Phase 2: Maximizing Profit
- Goal: Collect maximum possible coins
- Uses profit-maximizing heuristic
- Considers thief mechanics when computing optimal path

#### Phase 3: Minimizing Theft
- Goal: Minimize coins stolen
- Uses theft-minimizing heuristic
- May take longer paths to avoid thieves or high-value cells when a thief is present

## Implementation Details

### Dynamic Programming Heuristics

Two DP tables are precomputed for efficiency:

1. **Max Profit DP**:
   - Tracks maximum possible profit from any cell to goal
   - Maintains separate tables for "with thief" and "without thief" states
   - Used as heuristic for Phase 2

   * Calculates the maximum possible net `collected_coin` achievable from any cell `(r, c)` to the goal `(n-1, n-1)`.
    * Uses two DP tables: `dp_no` (entering `(r,c)` without a thief) and `dp_th` (entering `(r,c)` with a thief).
    * Iterates bottom-up from the goal.
    * Result stored in `self.profit_dp`.

2. **Min Loss DP**:
   - Tracks minimum possible theft from any cell to goal
   - Maintains separate tables for "with thief" and "without thief" states
   - Used as heuristic for Phase 3

   * Calculates the minimum possible `stolen_coin` accumulated from any cell `(r, c)` to the goal `(n-1, n-1)`.
    * Also uses `dp_no` and `dp_th` tables, initialized to `float('inf')`.
    * Iterates bottom-up from the goal.
    * Result stored in `self.min_loss_dp`.

### User Interface

The application provides:
- Grid creation and editing
- Random grid generation
- File I/O for grid loading/saving
- Visual path representation
- Statistics display for comparison

## Usage

1. Start the application (`main_file_graphical.py`)
2. Create or load a grid
3. Choose the desired approach (Phase 1, 2, or 3)
4. View the path and statistics

## Results Comparison

The system allows comparing the three approaches:
- Path length
- Coins collected
- Coins stolen
- Number of states searched (algorithmic efficiency)

## files name 
* `informed_search.py`: Contains the `Astar` class implementing the A* search logic, DP precomputation, and heuristic/cost functions.
* `uninformed_search.py`: Contains the `BFSSolver` class implementing the BFS algorithm.
* `grid_simulator_informed.py`: Contains helper classes for A* (`State`, `Priority`, `PriorityQueue`).
* `grid_simulator_uninformed.py` : Contains logic to simulate a path found by BFS .
* `main_file_graphical.py` : graphical interface.
* `main_file.py`: normal main, allowing the user to select the phase (terminal interface).
* `grid_graphics.py`: GUI, visulization logic.

## How to Run

1.  Ensure you have Python installed.
2.  Install any necessary libraries (e.g., `tkinter` for GUI if used).
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the main script:
    ```bash
    python main_file.py # or main_file_graphical
    ```
4.  Use it to input/generate a map and select the desired search phase (BFS, Max Profit A*, Min Loss A*).

## Notes

* The DP-based heuristics used in A* are "perfect" because they calculate the true optimal future value, ensuring A* finds the optimal path with potentially fewer state expansions than with less accurate heuristics.
* The alternative commented-out heuristics provide examples of other possible approaches, though they might not be as effective especially the phase 2 heuristic.

---
