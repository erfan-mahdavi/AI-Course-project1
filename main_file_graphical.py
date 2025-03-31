import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import random
from grid_simulator_uninformed import GridSimulator
from uninformed_search import BFSSolver
from informed_search import Astar
from grid_graphics import GridGraphics

def generate_grid(n):
    grid = []
    for _ in range(n):
        row = []
        for _ in range(n):
            choice = random.choice([1, -1, '!'])
            if choice == 1:
                num = random.randint(1, 500)
                row.append(num)
            elif choice == -1:
                num = random.randint(-500, -1)
                row.append(num)
            else:
                row.append('!')
        grid.append(row)
    return grid

def save_grid_to_file(grid, filename):
    with open(filename, 'w') as f:
        for row in grid:
            f.write(' '.join(map(str, row)) + '\n')

class Menu:
    def __init__(self, n, grid):
        self.n = n
        self.grid = grid
        self.root = tk.Tk()
        self.root.title("Grid Pathfinding Visualization")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill="x")
        ttk.Label(control_frame, text="Select approach:", font=("Arial", 12, "bold")).pack(pady=5)
        phase_frame = ttk.Frame(control_frame)
        phase_frame.pack(pady=10)
        ttk.Button(phase_frame, text="Phase 1: Exit-only (BFS)", command=self.phase1).pack(side="left", padx=5)
        ttk.Button(phase_frame, text="Phase 2: Max-profit (A*)", command=self.phase2).pack(side="left", padx=5)
        ttk.Button(phase_frame, text="Phase 3: Min-loss (A*)", command=self.phase3).pack(side="left", padx=5)
        self.grid_vis = GridGraphics(self.root, self.grid)
        self.root.mainloop()

    def phase1(self):
        bfs_solver = BFSSolver(self.n)
        path,state_counter = bfs_solver.find_shortest_path()
        if path:
            simulator = GridSimulator(self.grid)
            result = simulator.simulate_path(path)
            result["states_searched"]=state_counter
            print("\nApproach 1 (Exit-only):")
            print("Path steps:", result['path'])
            print("Coins collected:", result['coins_collected'])
            print("Coins stolen:", result['coins_stolen'])
            print("States searched:", state_counter)
            self.grid_vis.visualize_path(path, 1, result)
        else:
            messagebox.showerror("Error", "No path found for Phase 1")

    def phase2(self):
        astar = Astar(self.n, self.grid, 2)
        result, state_counter = astar.A_star()
        if result != None:
            path = result.state[9]
            simulation_result = {'coins_collected': result.state[5], 'coins_stolen': result.state[4], 'path': path, 'states_searched': state_counter}
            print("\nApproach 2 (Max-profit):")
            print("Path steps:", path)
            print("Coins collected:", result.state[5])
            print("Coins stolen:", result.state[4])
            print("States searched:", state_counter)
            self.grid_vis.visualize_path(path, 2, simulation_result)
        else:
            messagebox.showerror("Error", "No path found for Phase 2")

    def phase3(self):
        astar = Astar(self.n, self.grid, 3)
        result, state_counter = astar.A_star()
        if result != None:
            path = result.state[9]
            simulation_result = {'coins_collected': result.state[5], 'coins_stolen': result.state[4], 'path': path, 'states_searched': state_counter}
            print("\nApproach 3 (Min-loss):")
            print("Path steps:", path)
            print("Coins collected:", result.state[5])
            print("Coins stolen:", result.state[4])
            print("States searched:", state_counter)
            self.grid_vis.visualize_path(path, 3, simulation_result)
        else:
            messagebox.showerror("Error", "No path found for Phase 3")

class GridApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Grid Setup")
        self.root.geometry("800x700")
        self.root.resizable(True, True)
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill="both", expand=True)
        size_frame = ttk.Frame(main_frame)
        size_frame.pack(fill="x", pady=10)
        ttk.Label(size_frame, text="Grid Size (n):", font=("Arial", 12)).pack(side="left")
        self.size_var = tk.StringVar()
        size_entry = ttk.Entry(size_frame, textvariable=self.size_var, width=5)
        size_entry.pack(side="left", padx=10)
        ttk.Button(size_frame, text="Create Grid", command=self.create_grid_inputs).pack(side="left", padx=10)
        control_buttons_frame = ttk.Frame(main_frame)
        control_buttons_frame.pack(fill="x", pady=10)
        ttk.Button(control_buttons_frame, text="Generate Random Grid", command=self.generate_random_grid).pack(side="left", padx=5)
        ttk.Button(control_buttons_frame, text="Paste Grid", command=self.paste_grid).pack(side="left", padx=5)
        ttk.Button(control_buttons_frame, text="Load Grid", command=self.load_grid_from_file).pack(side="left", padx=5)
        self.grid_frame = ttk.Frame(main_frame)
        self.grid_frame.pack(fill="both", expand=True, pady=10)
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=10)
        self.start_button = ttk.Button(button_frame, text="Start Simulation", command=self.start_simulation, state="disabled")
        self.start_button.pack(pady=10)
        self.grid_entries = []
        self.root.mainloop()

    def create_grid_inputs(self, grid=None):
        for widget in self.grid_frame.winfo_children():
            widget.destroy()
        try:
            if grid is not None:
                n = len(grid)
                for row in grid:
                    if len(row) != n:
                        raise ValueError("Grid is not square.")
            else:
                n = int(self.size_var.get())
                if n <= 0:
                    raise ValueError("Grid size must be positive")
            self.size_var.set(str(n))
            canvas_frame = ttk.Frame(self.grid_frame)
            canvas_frame.pack(fill="both", expand=True)
            canvas = tk.Canvas(canvas_frame)
            scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            ttk.Label(scrollable_frame, text="Enter grid values (use '!' for thief cells, integers for other cells):", font=("Arial", 10, "bold")).grid(row=0, column=0, columnspan=n+1, pady=10)
            self.grid_entries = []
            for i in range(n):
                row_entries = []
                for j in range(n):
                    cell_var = tk.StringVar()
                    if grid is not None:
                        cell_value = grid[i][j]
                        cell_var.set(str(cell_value))
                    entry = ttk.Entry(scrollable_frame, textvariable=cell_var, width=5)
                    entry.grid(row=i+1, column=j, padx=2, pady=2)
                    row_entries.append(cell_var)
                self.grid_entries.append(row_entries)
            self.start_button.config(state="normal")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def generate_random_grid(self):
        try:
            n = int(self.size_var.get())
            if n <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Enter a valid grid size first.")
            return
        grid = generate_grid(n)
        self.create_grid_inputs(grid=grid)

    def paste_grid(self):
        paste_window = tk.Toplevel(self.root)
        paste_window.title("Paste Grid")
        text_area = tk.Text(paste_window, width=30, height=15)
        text_area.pack(padx=10, pady=10)
        def process_pasted_grid():
            text = text_area.get("1.0", tk.END).strip()
            lines = text.split('\n')
            grid = []
            for line in lines:
                elements = line.strip().split()
                row = []
                for elem in elements:
                    if elem == '!':
                        row.append('!')
                    else:
                        try:
                            row.append(int(elem))
                        except:
                            messagebox.showerror("Error", f"Invalid element: {elem}")
                            paste_window.destroy()
                            return
                grid.append(row)
            if not grid:
                messagebox.showerror("Error", "Empty grid.")
                paste_window.destroy()
                return
            n = len(grid)
            for row in grid:
                if len(row) != n:
                    messagebox.showerror("Error", "Grid is not square.")
                    paste_window.destroy()
                    return
            self.create_grid_inputs(grid=grid)
            paste_window.destroy()
        ttk.Button(paste_window, text="Submit", command=process_pasted_grid).pack(pady=5)

    def load_grid_from_file(self):
        filename = filedialog.askopenfilename(title="Select Grid File", filetypes=[("Text Files", "*.txt")])
        if not filename:
            return
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            grid = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                elements = line.split()
                row = []
                for elem in elements:
                    if elem == '!':
                        row.append('!')
                    else:
                        try:
                            row.append(int(elem))
                        except:
                            raise ValueError(f"Invalid element: {elem}")
                grid.append(row)
            if not grid:
                raise ValueError("Empty grid.")
            n = len(grid)
            for row in grid:
                if len(row) != n:
                    raise ValueError("Grid is not square.")
            self.create_grid_inputs(grid=grid)
        except Exception as e:
            messagebox.showerror("Error", str(e))

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
    GridApp()

if __name__ == "__main__":
    main()

