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

### State Representation (`State` class)

* Defined in `grid_simulator_informed.py`.
* Crucial for tracking the necessary information during the A* search.
* Each state object stores a tuple:
    ```
    (
        index_i,          # Current row
        index_j,          # Current column
        current_cost,     # Accumulated cost from negative cells (absolute values)
        current_profit,   # Accumulated profit from positive cells
        stolen_coin,      # Accumulated coins stolen by thieves
        collected_coin,   # Net coins (profit - cost - extra stolen cost)
        has_thief,        # Boolean: True if currently accompanied by a thief
        parent_state,     # Reference to the previous State object
        action,           # Action ('down' or 'right') taken to reach this state
        list_of_actions   # List of actions from start to this state
    )
    ```
* The `__init__` method calculates the state values based on the `parent_state` and the `value` of the current cell, correctly applying the game rules (using helper methods `thief`, `positive_value`, `negative_value`).

### Dynamic Programming Heuristics

To guide the A* search efficiently, dynamic programming is used to precompute the *exact* optimal future values from any grid cell. These serve as perfect heuristics.

1.  **`_compute_profit_dp`:**
    * Calculates the maximum possible net `collected_coin` achievable from any cell `(r, c)` to the goal `(n-1, n-1)`.
    * Uses two DP tables: `dp_no` (entering `(r,c)` without a thief) and `dp_th` (entering `(r,c)` with a thief).
    * Iterates bottom-up from the goal.
    * Result stored in `self.profit_dp`.

2.  **`_compute_min_loss_dp`:**
    * Calculates the minimum possible `stolen_coin` accumulated from any cell `(r, c)` to the goal `(n-1, n-1)`.
    * Also uses `dp_no` and `dp_th` tables, initialized to `float('inf')`.
    * Iterates bottom-up from the goal.
    * Result stored in `self.min_loss_dp`.

### A* Heuristic and Cost Functions

The `Astar` class defines cost (`g`) and heuristic (`h`) functions based on the phase:

* **Phase 2 (Max Profit):**
    * `g = cost_1(state)`: Returns `state.state[5]` (current `collected_coin`). This represents the "cost so far" in terms of profit achieved.
    * `h = heuristic_1(state)`: Returns `self.profit_dp[thief][r][c]`, the precomputed maximum *future* profit from the current cell `(r, c)` given the current `thief` status.
    * Priority: `f = (g + h) * -1`. We maximize `g+h` (total possible profit) using a min-priority queue by negating the value.

* **Phase 3 (Min Loss):**
    * `g = cost_2(state)`: Returns `state.state[4]` (current `stolen_coin`). This is the cost (loss) accumulated so far.
    * `h = heuristic_2(state)`: Returns `self.min_loss_dp[thief][r][c]`, the precomputed minimum *future* loss from the current cell `(r, c)` given the current `thief` status.
    * Priority: `f = (g + h) * 1`. We minimize `g+h` (total loss).

*(Note: The commented-out alternative heuristics in `Astar.py` represent attempts at potentially simpler, but likely less accurate or non-admissible, heuristic calculations.)*

### Grid Simulation

* The core simulation logic is embedded within the `State` class's update methods and mirrored precisely in the DP calculations.
* An external `GridSimulator` class (informed/uninformed, as mentioned) likely exists to take a generated path and calculate the final outcome (coins collected, stolen, etc.) for verification or display. *(Code for this simulator was discussed but not fully shown in the last prompt)*.

## Graphical User Interface (GUI)

The project includes a graphical user interface with the following features:
* Inputting the grid map (e.g., via copy-pasting).
* Generating random grid maps.
* Visualizing the calculated optimal path on the grid.
* Displaying key results:
    * Final path (sequence of moves).
    * Final net coins collected.
    * Total coins stolen by thieves.
    * Number of states explored by the search algorithm.

## Code Structure (Inferred)

* `informed_search.py`: Contains the `Astar` class implementing the A* search logic, DP precomputation, and heuristic/cost functions.
* `uninformed_search.py`: Contains the `BFSSolver` class implementing the BFS algorithm.
* `grid_simulator_informed.py`: Contains helper classes for A* (`State`, `Priority`, `PriorityQueue`).
* `grid_simulator_uninformed.py` : Contains logic to simulate a path found by BFS .
* `main_file_graphical.py` : graphical interface.
* `main_file.py`: normal main, allowing the user to select the phase (terminal interface).
* `grid_graphics.py`: GUI, visulization logic.

## How to Run

*(Provide specific instructions here if available)*
Example:
1.  Ensure you have Python installed.
2.  Install any necessary libraries (e.g., `tkinter` for GUI if used).
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the main script:
    ```bash
    python main.py
    ```
4.  Use the GUI to input/generate a map and select the desired search phase (BFS, Max Profit A*, Min Loss A*).

## Notes

* The DP-based heuristics used in A* are "perfect" because they calculate the true optimal future value, ensuring A* finds the optimal path with potentially fewer state expansions than with less accurate heuristics.
* The alternative commented-out heuristics provide examples of other possible approaches, though they might not be as effective especially the phase 2 heuristic.

---
