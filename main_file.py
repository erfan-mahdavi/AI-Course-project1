# main.py
from grid_simulator import GridSimulator
from uninformed_search import BFSSolver

def main():
    try:
        n = int(input())
    except ValueError:
        print("Invalid grid size.")
        return

    grid = []
    for _ in range(n):
        row_input = input().strip().split()
        row = []
        for cell in row_input:
            if cell == '!':
                row.append('!')
            else:
                try:
                    row.append(int(cell))
                except ValueError:
                    print("Invalid cell value:", cell)
                    return
        if len(row) != n:
            print("Row length does not match grid size.")
            return
        grid.append(row)

    print("Select approach:")
    print("1: Exit-only (BFS)")
    choice = input("Enter 1 : ").strip()

    if choice == '1':
        bfs_solver = BFSSolver(n)
        path = bfs_solver.find_shortest_path()
        if not path:
            print("No path found.")
            return
        simulator = GridSimulator(grid)
        result = simulator.simulate_path(path)
        print("\nApproach 1 (Exit-only):")
        print("Path steps:", result['path'])
        print("Coins collected:", result['coins_collected'])
        print("Coins stolen:", result['coins_stolen'])
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
