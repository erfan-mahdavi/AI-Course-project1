import tkinter as tk
from tkinter import ttk, messagebox
import time

# In grid_graphics.py - Improved visualization for larger grids
class GridGraphics:
    def __init__(self, master, grid, cell_size=None):
        self.master = master
        self.grid = grid
        self.n = len(grid)
        
        # Automatically adjust cell size based on grid dimensions
        # This ensures better visibility for larger grids
        if cell_size is None:
            # Calculate optimal cell size based on grid size
            default_canvas_size = 600
            self.cell_size = max(20, min(60, default_canvas_size // self.n))
        else:
            self.cell_size = cell_size
            
        # Color scheme with better contrast
        self.colors = {
            "background": "#f0f0f0", 
            "grid_line": "#cccccc", 
            "normal_cell": "#ffffff", 
            "treasure_cell": "#90ee90", 
            "thief_cell": "#ffcccb", 
            "negative_cell": "#d3d3d3", 
            "path": "#4682b4", 
            "current_pos": "#ff7f50", 
            "visited": "#e6e6fa", 
            "start": "#32cd32", 
            "goal": "#ff4500"
        }
        
        # Main frame setup
        self.frame = ttk.Frame(master)
        self.frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Canvas dimensions based on grid size
        canvas_width = self.n * self.cell_size + 1
        canvas_height = self.n * self.cell_size + 1
        
        # Set maximum canvas size with scrollbars for large grids
        max_canvas_width = min(canvas_width, 800)
        max_canvas_height = min(canvas_height, 600)
        
        # Canvas with scrollbars
        self.canvas = tk.Canvas(
            self.frame, 
            bg=self.colors["background"], 
            width=max_canvas_width, 
            height=max_canvas_height
        )
        
        self.h_scroll = ttk.Scrollbar(self.frame, orient="horizontal", command=self.canvas.xview)
        self.v_scroll = ttk.Scrollbar(self.frame, orient="vertical", command=self.canvas.yview)
        
        self.canvas.configure(
            xscrollcommand=self.h_scroll.set, 
            yscrollcommand=self.v_scroll.set, 
            scrollregion=(0, 0, canvas_width, canvas_height)
        )
        
        # Layout with grid manager
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")
        
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=1)
        
        # Info section
        self.info_frame = ttk.Frame(self.frame)
        self.info_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="ew")
        
        # Info labels with better layout
        self.info_labels = {}
        info_texts = ["Phase", "Path", "Coins Collected", "Coins Stolen", "Searched States"]
        
        for i, text in enumerate(info_texts):
            label_text = f"{text}:"
            if i == 1:  # Path info might be long
                label_text = f"{text} (start → end):"
                
            label = ttk.Label(self.info_frame, text=label_text)
            label.grid(row=i, column=0, sticky="w", pady=2)
            
            # Value labels with reasonable width
            self.info_labels[text.lower().replace(" ", "_")] = ttk.Label(
                self.info_frame, 
                text="", 
                wraplength=600
            )
            self.info_labels[text.lower().replace(" ", "_")].grid(
                row=i, column=1, sticky="w", padx=10, pady=2
            )
        
        # Control buttons
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.grid(row=3, column=0, columnspan=2, pady=10, sticky="ew")
        
        self.step_button = ttk.Button(self.button_frame, text="Step", state="disabled")
        self.step_button.pack(side="left", padx=5)
        
        self.auto_button = ttk.Button(self.button_frame, text="Auto Run", state="disabled")
        self.auto_button.pack(side="left", padx=5)
        
        self.reset_button = ttk.Button(self.button_frame, text="Reset")
        self.reset_button.pack(side="left", padx=5)
        
        # Speed control
        self.speed_frame = ttk.Frame(self.button_frame)
        self.speed_frame.pack(side="left", padx=20)
        
        ttk.Label(self.speed_frame, text="Animation Speed:").pack(side="left")
        
        self.speed_var = tk.IntVar(value=500)
        self.speed_scale = ttk.Scale(
            self.speed_frame, 
            from_=100, to=2000, 
            orient="horizontal",
            variable=self.speed_var, 
            length=150
        )
        self.speed_scale.pack(side="left", padx=5)
        
        # Initialize grid
        self.draw_grid()
        self.cell_text_ids = {}
        self.cell_ids = {}
        self.fill_grid()
        
        # Animation state
        self.current_path = []
        self.current_step = 0
        self.simulation_running = False
        self.animation_speed = 500

    def draw_grid(self):
        for i in range(self.n + 1):
            # Horizontal lines
            self.canvas.create_line(
                0, i * self.cell_size, 
                self.n * self.cell_size, i * self.cell_size, 
                fill=self.colors["grid_line"], width=1
            )
            # Vertical lines
            self.canvas.create_line(
                i * self.cell_size, 0, 
                i * self.cell_size, self.n * self.cell_size, 
                fill=self.colors["grid_line"], width=1
            )

    def fill_grid(self):
        for i in range(self.n):
            for j in range(self.n):
                cell_value = self.grid[i][j]
                self.fill_cell(i, j, cell_value)

    def fill_cell(self, i, j, value, highlight=None):
        x1 = j * self.cell_size
        y1 = i * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        
        # Determine cell color based on value and highlight
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
        
        # Special colors for start and goal cells
        if i == 0 and j == 0:
            cell_color = self.colors["start"]
        elif i == self.n - 1 and j == self.n - 1:
            cell_color = self.colors["goal"]
        
        # Create or update cell rectangle
        cell_id = f"cell_{i}_{j}"
        if cell_id in self.cell_ids:
            self.canvas.itemconfig(self.cell_ids[cell_id], fill=cell_color)
        else:
            self.cell_ids[cell_id] = self.canvas.create_rectangle(
                x1, y1, x2, y2, 
                fill=cell_color, 
                outline=self.colors["grid_line"]
            )
        
        # Create or update cell text
        text_id = f"text_{i}_{j}"
        text_value = value if value != '!' else "!"
        
        # Adjust font size based on cell size
        font_size = max(8, min(12, self.cell_size // 5))
        
        if text_id in self.cell_text_ids:
            self.canvas.itemconfig(
                self.cell_text_ids[text_id], 
                text=str(text_value),
                font=("Arial", font_size, "bold")
            )
        else:
            self.cell_text_ids[text_id] = self.canvas.create_text(
                x1 + self.cell_size//2, 
                y1 + self.cell_size//2, 
                text=str(text_value), 
                font=("Arial", font_size, "bold")
            )

    def update_info(self, phase=None, path=None, coins_collected=None, coins_stolen=None, states_searched=None):
        if phase is not None:
            self.info_labels["phase"].config(text=f"Phase {phase}")
        
        if path is not None:
            # Truncate long paths for display
            if len(path) > 20:
                path_display = ", ".join(path[:10]) + " ... " + ", ".join(path[-10:])
            else:
                path_display = ", ".join(path)
            self.info_labels["path"].config(text=path_display)
        
        if coins_collected is not None:
            self.info_labels["coins_collected"].config(text=str(coins_collected))
        
        if coins_stolen is not None:
            self.info_labels["coins_stolen"].config(text=str(coins_stolen))
        
        if states_searched is not None:
            self.info_labels["searched_states"].config(text=str(states_searched))

    def visualize_path(self, path, phase, simulation_result):
        self.current_path = path
        self.current_step = 0
        self.reset_grid()
        
        # Update animation speed from slider
        self.animation_speed = self.speed_var.get()
        
        # Update info panel
        self.update_info(
            phase=phase,
            path=path,
            coins_collected=simulation_result["coins_collected"],
            coins_stolen=simulation_result["coins_stolen"],
            states_searched=simulation_result.get("states_searched", "N/A")
        )
        
        # Enable control buttons
        self.step_button.config(state="normal", command=self.step_forward)
        self.auto_button.config(state="normal", command=self.auto_run)
        self.reset_button.config(command=self.reset_visualization)
        
        # Center view on start position
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def step_forward(self):
        if self.current_step < len(self.current_path):
            # Calculate current position
            i, j = 0, 0
            for s in range(self.current_step):
                if self.current_path[s] == "down":
                    i += 1
                else:
                    j += 1
            
            # Mark current cell as visited
            self.fill_cell(i, j, self.grid[i][j], highlight="visited")
            
            # Move to next cell based on direction
            if self.current_path[self.current_step] == "down":
                i += 1
            else:
                j += 1
                
            # Highlight new position
            self.fill_cell(i, j, self.grid[i][j], highlight="current_pos")
            
            # Auto-scroll to keep current position visible
            self.scroll_to_cell(i, j)
            
            # Increment step counter
            self.current_step += 1
            
            # Disable step button if we've reached the end
            if self.current_step >= len(self.current_path):
                self.step_button.config(state="disabled")
                self.fill_cell(self.n - 1, self.n - 1, self.grid[self.n - 1][self.n - 1], highlight="goal")

    def scroll_to_cell(self, i, j):
        # Calculate the location to scroll to (as a fraction of canvas size)
        canvas_width = self.n * self.cell_size
        canvas_height = self.n * self.cell_size
        
        x_fraction = (j * self.cell_size) / canvas_width
        y_fraction = (i * self.cell_size) / canvas_height
        
        # Apply scrolling if needed
        self.canvas.xview_moveto(max(0, x_fraction - 0.4))
        self.canvas.yview_moveto(max(0, y_fraction - 0.4))

    def auto_run(self):
        self.simulation_running = True
        self.auto_button.config(state="disabled")
        self.step_button.config(state="disabled")
        
        # Update animation speed from slider
        self.animation_speed = self.speed_var.get()
        
        # Start auto-stepping
        self.auto_step()

    def auto_step(self):
        if self.simulation_running and self.current_step < len(self.current_path):
            self.step_forward()
            self.master.after(self.animation_speed, self.auto_step)
        elif self.simulation_running:
            self.simulation_running = False
            self.auto_button.config(state="normal")

    def reset_visualization(self):
        self.current_step = 0
        self.simulation_running = False
        self.reset_grid()
        self.step_button.config(state="normal")
        self.auto_button.config(state="normal")

    def reset_grid(self):
        for i in range(self.n):
            for j in range(self.n):
                self.fill_cell(i, j, self.grid[i][j])
def show_grid_visualization(grid, phase, path, simulation_result):
    root = tk.Tk()
    root.title(f"Phase {phase} Grid Visualization")
    grid_vis = GridGraphics(root, grid)
    grid_vis.visualize_path(path, phase, simulation_result)
    root.mainloop()

