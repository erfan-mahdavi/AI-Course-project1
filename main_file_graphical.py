import tkinter as tk
from tkinter import ttk, messagebox
from grid_simulator_uninformed import GridSimulator
from uninformed_search import BFSSolver
from informed_search import Astar
from grid_graphics import GridGraphics

class Menu:
    def __init__(self, n, grid):
        self.n = n
        self.grid = grid
        
        self.root = tk.Tk()
        self.root.title("Grid Pathfinding Visualization")
        self.root.geometry("600x700")
        
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill="x")
        
        ttk.Label(control_frame, text="Select approach:", font=("Arial", 12, "bold")).pack(pady=5)
        
        phase_frame = ttk.Frame(control_frame)
        phase_frame.pack(pady=10)
        
        ttk.Button(phase_frame, text="Phase 1: Exit-only (BFS)", 
                  command=self.phase1).pack(side="left", padx=5)
        ttk.Button(phase_frame, text="Phase 2: Max-profit (A*)", 
                  command=self.phase2).pack(side="left", padx=5)
        ttk.Button(phase_frame, text="Phase 3: Min-loss (A*)", 
                  command=self.phase3).pack(side="left", padx=5)
        
        self.grid_vis = GridGraphics(self.root, self.grid)
        
        self.root.mainloop()

    def phase1(self):
        print('===============  phase 1 ====================')
        bfs_solver = BFSSolver(self.n)
        path = bfs_solver.find_shortest_path()
        if path:
            simulator = GridSimulator(self.grid)
            result = simulator.simulate_path(path)
            print("\nApproach 1 (Exit-only):")
            print("Path steps:", result['path'])
            print("Coins collected:", result['coins_collected'])
            print("Coins stolen:", result['coins_stolen'])
            
            self.grid_vis.visualize_path(path, 1, result)
        else:
            print('No Path Found.')
            messagebox.showerror("Error", "No path found for Phase 1")
        print('=================== phase 1 ended ========================')

    def phase2(self):
        print('===============  phase 2 ====================')
        astar = Astar(self.n, self.grid, 2)
        result = astar.A_star()
        if result != None:
            path = result.state[9]
            simulation_result = {
                'coins_collected': result.state[5],
                'coins_stolen': result.state[4],
                'path': path
            }
            print("\nApproach 2 (Max-profit):")
            print("Path steps:", path)
            print("Coins collected:", result.state[5])
            print("Coins stolen:", result.state[4])
            
            self.grid_vis.visualize_path(path, 2, simulation_result)
        else:
            print('No Path Found.')
            messagebox.showerror("Error", "No path found for Phase 2")
        print('=================== phase 2 ended ========================')

    def phase3(self):
        print('===============  phase 3 ====================')
        astar = Astar(self.n, self.grid, 3)
        result = astar.A_star()
        if result != None:
            path = result.state[9]
            simulation_result = {
                'coins_collected': result.state[5],
                'coins_stolen': result.state[4],
                'path': path
            }
            print("\nApproach 3 (Min-loss):")
            print("Path steps:", path)
            print("Coins collected:", result.state[5])
            print("Coins stolen:", result.state[4])
            
            self.grid_vis.visualize_path(path, 3, simulation_result)
        else:
            print('No Path Found.')
            messagebox.showerror("Error", "No path found for Phase 3")
        print('=================== phase 3 ended ========================')


class GridApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Grid Setup")
        self.root.geometry("500x600")
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        size_frame = ttk.Frame(main_frame)
        size_frame.pack(fill="x", pady=10)
        
        ttk.Label(size_frame, text="Grid Size (n):", font=("Arial", 12)).pack(side="left")
        self.size_var = tk.StringVar()
        size_entry = ttk.Entry(size_frame, textvariable=self.size_var, width=5)
        size_entry.pack(side="left", padx=10)
        ttk.Button(size_frame, text="Create Grid", command=self.create_grid_inputs).pack(side="left", padx=10)
        
        self.grid_frame = ttk.Frame(main_frame)
        self.grid_frame.pack(fill="both", expand=True, pady=10)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Simulation", 
                                      command=self.start_simulation, state="disabled")
        self.start_button.pack(pady=10)
        
        self.grid_entries = []
        
        self.root.mainloop()
    
    def create_grid_inputs(self):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        
        try:
            n = int(self.size_var.get())
            if n <= 0:
                raise ValueError("Grid size must be positive")
            
            canvas_frame = ttk.Frame(self.grid_frame)
            canvas_frame.pack(fill="both", expand=True)
            
            canvas = tk.Canvas(canvas_frame)
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            ttk.Label(scrollable_frame, text="Enter grid values (use '!' for thief cells, integers for other cells):",
                     font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=n+1, pady=10)
            
            self.grid_entries = []
            for i in range(n):
                row_entries = []
                for j in range(n):
                    cell_var = tk.StringVar()
                    entry = ttk.Entry(scrollable_frame, textvariable=cell_var, width=5)
                    entry.grid(row=i+1, column=j, padx=2, pady=2)
                    row_entries.append(cell_var)
                self.grid_entries.append(row_entries)
            
            self.start_button.config(state="normal")
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
    
    def start_simulation(self):
        try:
            n = len(self.grid_entries)
            grid = []
            
            for i in range(n):
                row = []
                for j in range(n):
                    value = self.grid_entries[i][j].get().strip()
                    if value == '!':
                        row.append('!')
                    else:
                        try:
                            row.append(int(value))
                        except ValueError:
                            raise ValueError(f"Invalid entry at position ({i+1}, {j+1}): {value}")
                grid.append(row)
            
            self.root.destroy()
            
            Menu(n, grid)
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))


def main():
    app = GridApp()


if __name__ == "__main__":
    main()