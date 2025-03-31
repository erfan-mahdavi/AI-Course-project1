# main.py
from grid_simulator_uninformed import GridSimulator
from uninformed_search import BFSSolver
from informed_search import Astar


class Menu:
    def __init__(self,n,grid):
        while True:
            print("Select approach:")
            print("1: phase 1 ") # Exit-only (BFS)
            print('2: phase 2 ') # Max-profit (Astar)
            print('3: phase 3 ') # Min-loss (Astar)
            print('4: exit')
            choice = input("Enter your choice : ").strip()

            if choice == '1':
                self.phase1(n,grid)
            elif choice=='2':
                self.phase2(n,grid)
            elif choice=='3':
                self.phase3(n,grid)
            elif choice=='4':
                exit()
            else:
                print('invalid input\ntry again')

    def phase1(self,n,grid):
        print('===============  phase 1 ====================')
        bfs_solver = BFSSolver(n)
        path = bfs_solver.find_shortest_path()
        if path:
            simulator = GridSimulator(grid)
            result = simulator.simulate_path(path)
            print("\nApproach 1 (Exit-only):")
            print("Path steps:", result['path'])
            print("Coins collected:", result['coins_collected'])
            print("Coins stolen:", result['coins_stolen'])
        else:
            print('No Path Found.')
        print('=================== phase 1 ended ========================')

    def phase2(self,n,grid):
        print('===============  phase 2 ====================')
        astar = Astar(n,grid,2)
        result,state_counter = astar.A_star()
        if result!=None:
            print("\nApproach 2 (Max-profit):")
            print("Path steps:", result.state[9])
            print("Coins collected:", result.state[5])
            print("Coins stolen:", result.state[4])
            print("number of searched states:", state_counter)
        else:
            print('No Path Found.')
        print('=================== phase 2 ended ========================')

    def phase3(self,n,grid):
        print('===============  phase 3 ====================')
        astar = Astar(n,grid,3)
        result = astar.A_star()
        if result!=None:
            print("\nApproach 3 (Min-loss):")
            print("Path steps:", result.state[9])
            print("Coins collected:", result.state[5])
            print("Coins stolen:", result.state[4])
        else:
            print('No Path Found.')
        print('=================== phase 3 ended ========================')

def main():
    while True:
        try:
            n = int(input('enter number of rows : '))
            if n<0:
                raise(ValueError)
            break
        except ValueError:
            print("Invalid grid size.\ntry again")


    grid = []
    while True:
        print('enter grid : ')
        for _ in range(n):
            row_input = input().strip().split()
            row = []
            flag1=0
            for cell in row_input:
                if cell == '!':
                    row.append('!')
                else:
                    try:
                        row.append(int(cell))
                    except ValueError:
                        print("Invalid cell value:", cell)
                        flag1 = 1
                        print('try again')
                        grid.clear()
                        break
            if len(row) != n and flag1==0:
                print("Row length does not match grid size.\ntry again")
                grid.clear()
                break
            if flag1==1:
                break
            grid.append(row)
        else:
            m = Menu(n,grid)


if __name__ == "__main__":
    main()
