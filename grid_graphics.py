import tkinter as tk
from tkinter import ttk, messagebox
import time

class GridGraphics:
    def __init__(self, master, grid, cell_size=60):
        self.master = master
        self.grid = grid
        self.n = len(grid)
        self.cell_size = cell_size
        
        self.colors = {
            "background": "#f0f0f0",
            "grid_line": "#cccccc",
            "normal_cell": "#ffffff",
            "treasure_cell": "#90ee90",  # Light green
            "thief_cell": "#ffcccb",     # Light red
            "negative_cell": "#d3d3d3",  # Light gray
            "path": "#4682b4",           # Steel blue
            "current_pos": "#ff7f50",    # Coral
            "visited": "#e6e6fa",        # Lavender
            "start": "#32cd32",          # Lime green
            "goal": "#ff4500"            # Orange red
        }
        
        self.frame = ttk.Frame(master)
        self.frame.pack(padx=20, pady=20)
        
        canvas_width = self.n * cell_size + 1
        canvas_height = self.n * cell_size + 1
        self.canvas = tk.Canvas(self.frame, width=canvas_width, height=canvas_height, 
                               bg=self.colors["background"])
        self.canvas.pack(pady=10)
        
        self.info_frame = ttk.Frame(self.frame)
        self.info_frame.pack(fill="x", pady=10)
        
        self.info_labels = {}
        info_texts = ["Phase", "Path", "Coins Collected", "Coins Stolen"]
        
        for i, text in enumerate(info_texts):
            ttk.Label(self.info_frame, text=f"{text}:").grid(row=i, column=0, sticky="w", pady=2)
            self.info_labels[text.lower().replace(" ", "_")] = ttk.Label(self.info_frame, text="")
            self.info_labels[text.lower().replace(" ", "_")].grid(row=i, column=1, sticky="w", padx=10, pady=2)
        
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.pack(fill="x", pady=10)
        
        self.step_button = ttk.Button(self.button_frame, text="Step", state="disabled")
        self.step_button.pack(side="left", padx=5)
        
        self.auto_button = ttk.Button(self.button_frame, text="Auto Run", state="disabled")
        self.auto_button.pack(side="left", padx=5)
        
        self.reset_button = ttk.Button(self.button_frame, text="Reset")
        self.reset_button.pack(side="left", padx=5)
        
        self.draw_grid()
        self.cell_text_ids = {}
        self.fill_grid()
        
        self.current_path = []
        self.current_step = 0
        self.simulation_running = False
        self.animation_speed = 500
    
    def draw_grid(self):
        """Draw the grid lines"""
        for i in range(self.n + 1):
            self.canvas.create_line(
                0, i * self.cell_size, 
                self.n * self.cell_size, i * self.cell_size,
                fill=self.colors["grid_line"], width=1
            )
            self.canvas.create_line(
                i * self.cell_size, 0, 
                i * self.cell_size, self.n * self.cell_size,
                fill=self.colors["grid_line"], width=1
            )
    
    def fill_grid(self):
        """Fill the grid with cell values"""
        for i in range(self.n):
            for j in range(self.n):
                cell_value = self.grid[i][j]
                self.fill_cell(i, j, cell_value)
    
    def fill_cell(self, i, j, value, highlight=None):
        """Fill a single cell with its value and optional highlighting"""
        x1 = j * self.cell_size
        y1 = i * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        
        if highlight:
            cell_color = self.colors[highlight]
        elif value == '!':
            cell_color = self.colors["thief_cell"]
        elif isinstance(value, int):
            if value > 0:
                cell_color = self.colors["treasure_cell"]
            else:
                cell_color = self.colors["negative_cell"]
        else:
            cell_color = self.colors["normal_cell"]
        
        if i == 0 and j == 0:
            cell_color = self.colors["start"]
        elif i == self.n - 1 and j == self.n - 1:
            cell_color = self.colors["goal"]
        
        cell_id = f"cell_{i}_{j}"
        if hasattr(self, 'cell_ids') and cell_id in self.cell_ids:
            self.canvas.itemconfig(self.cell_ids[cell_id], fill=cell_color)
        else:
            if not hasattr(self, 'cell_ids'):
                self.cell_ids = {}
            self.cell_ids[cell_id] = self.canvas.create_rectangle(
                x1, y1, x2, y2, fill=cell_color, outline=self.colors["grid_line"]
            )
        
        text_id = f"text_{i}_{j}"
        text_value = value if value != '!' else "!"
        
        if text_id in self.cell_text_ids:
            self.canvas.itemconfig(self.cell_text_ids[text_id], text=str(text_value))
        else:
            self.cell_text_ids[text_id] = self.canvas.create_text(
                x1 + self.cell_size // 2,
                y1 + self.cell_size // 2,
                text=str(text_value),
                font=("Arial", 12, "bold")
            )
    
    def update_info(self, phase=None, path=None, coins_collected=None, coins_stolen=None):
        """Update the information labels"""
        if phase is not None:
            self.info_labels["phase"].config(text=f"Phase {phase}")
        if path is not None:
            self.info_labels["path"].config(text=", ".join(path))
        if coins_collected is not None:
            self.info_labels["coins_collected"].config(text=str(coins_collected))
        if coins_stolen is not None:
            self.info_labels["coins_stolen"].config(text=str(coins_stolen))
    
    def visualize_path(self, path, phase, simulation_result):
        """Set up path visualization"""
        self.current_path = path
        self.current_step = 0
        self.reset_grid()
        
        self.update_info(
            phase=phase,
            path=path,
            coins_collected=simulation_result["coins_collected"],
            coins_stolen=simulation_result["coins_stolen"]
        )
        
        self.step_button.config(state="normal", command=self.step_forward)
        self.auto_button.config(state="normal", command=self.auto_run)
        self.reset_button.config(command=self.reset_visualization)
    
    def step_forward(self):
        """Advance one step in the path visualization"""
        if self.current_step < len(self.current_path):
            i, j = 0, 0
            for s in range(self.current_step):
                if self.current_path[s] == "down":
                    i += 1
                else:  # right
                    j += 1
            
            self.fill_cell(i, j, self.grid[i][j], highlight="visited")
            
            if self.current_path[self.current_step] == "down":
                i += 1
            else:  # right
                j += 1
            
            self.fill_cell(i, j, self.grid[i][j], highlight="current_pos")
            
            self.current_step += 1
            
            if self.current_step >= len(self.current_path):
                self.step_button.config(state="disabled")
                self.fill_cell(self.n - 1, self.n - 1, self.grid[self.n - 1][self.n - 1], highlight="goal")
    
    def auto_run(self):
        """Automatically run through the path visualization"""
        self.simulation_running = True
        self.auto_button.config(state="disabled")
        self.step_button.config(state="disabled")
        self.auto_step()
    
    def auto_step(self):
        """Helper function for auto run"""
        if self.simulation_running and self.current_step < len(self.current_path):
            self.step_forward()
            self.master.after(self.animation_speed, self.auto_step)
        elif self.simulation_running:
            self.simulation_running = False
            self.auto_button.config(state="normal")
    
    def reset_visualization(self):
        """Reset the visualization to initial state"""
        self.current_step = 0
        self.simulation_running = False
        self.reset_grid()
        self.step_button.config(state="normal")
        self.auto_button.config(state="normal")
    
    def reset_grid(self):
        """Reset the grid to its initial state"""
        for i in range(self.n):
            for j in range(self.n):
                self.fill_cell(i, j, self.grid[i][j])

def show_grid_visualization(grid, phase, path, simulation_result):
    """Helper function to show grid visualization in a new window"""
    root = tk.Tk()
    root.title(f"Phase {phase} Grid Visualization")
    
    grid_vis = GridGraphics(root, grid)
    grid_vis.visualize_path(path, phase, simulation_result)
    
    root.mainloop()