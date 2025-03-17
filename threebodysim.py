import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import scipy.integrate
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import time
from collections import deque
import random


class ThreeBodySimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Three-Body Problem Simulator")
        self.root.geometry("1300x850")
        self.root.configure(bg="#282c34")

        # Make the root window responsive
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Set up the style for dark theme
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TFrame", background="#282c34")
        self.style.configure("TButton", background="#3e4451", foreground="#abb2bf", font=('Arial', 10))
        self.style.configure("TLabel", background="#282c34", foreground="#abb2bf", font=('Arial', 10))
        self.style.configure("TScale", background="#282c34", troughcolor="#3e4451")
        self.style.configure("TLabelframe", background="#282c34", foreground="#abb2bf")
        self.style.configure("TLabelframe.Label", background="#282c34", foreground="#abb2bf",
                             font=('Arial', 10, 'bold'))
        self.style.configure("TRadiobutton", background="#282c34", foreground="#abb2bf")
        self.style.configure("TProgressbar", background="#61afef")
        self.style.map('TButton', background=[('active', '#61afef')])

        # Create main frame with grid instead of pack for better responsiveness
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=3)  # Plot takes 3/4
        self.main_frame.grid_columnconfigure(1, weight=1)  # Controls take 1/4

        # Create frame for the plot (using grid)
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        # Create scrollable control panel
        self.control_frame_outer = ttk.Frame(self.main_frame)
        self.control_frame_outer.grid(row=0, column=1, sticky="nsew")
        self.control_frame_outer.grid_rowconfigure(0, weight=1)
        self.control_frame_outer.grid_columnconfigure(0, weight=1)

        # Add a canvas and scrollbar for scrolling controls
        self.control_canvas = tk.Canvas(self.control_frame_outer, bg="#282c34",
                                        highlightthickness=0)
        self.control_scrollbar = ttk.Scrollbar(self.control_frame_outer,
                                               orient="vertical",
                                               command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=self.control_scrollbar.set)

        # Place canvas and scrollbar
        self.control_scrollbar.grid(row=0, column=1, sticky="ns")
        self.control_canvas.grid(row=0, column=0, sticky="nsew")
        self.control_frame_outer.grid_columnconfigure(0, weight=1)

        # Create frame for controls inside canvas
        self.control_frame = ttk.Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=self.control_frame,
                                          anchor="nw", tags="self.control_frame")

        # Configure canvas scrolling
        self.control_frame.bind("<Configure>", self.on_frame_configure)
        self.control_canvas.bind("<Configure>", self.on_canvas_configure)

        # Bind mousewheel for scrolling
        self.control_canvas.bind_all("<MouseWheel>", self.on_mousewheel)

        # Set up the figure and canvas for the plot
        self.fig = Figure(figsize=(8, 8), facecolor="#282c34", constrained_layout=True)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor("#282c34")
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.xaxis.label.set_color("#abb2bf")
        self.ax.yaxis.label.set_color("#abb2bf")
        self.ax.zaxis.label.set_color("#abb2bf")
        self.ax.tick_params(colors="#abb2bf")
        self.ax.set_title("Three-Body Problem Simulation", color="#abb2bf")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Make the plot canvas responsive
        self.plot_frame.grid_rowconfigure(0, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        # Initialize neural network
        self.init_deep_learning()

        # Set up the control widgets
        self.setup_controls()

        # Create event log
        self.create_event_log()

        # Initialize simulation parameters and state variables
        self.collision_detected = False
        self.instability_detected = False
        self.instability_reason = ""
        self.time_since_last_event = 0

        # Initialize performance parameters
        self.max_fps = 30  # Target maximum frames per second
        self.target_fps = 60  # UI responsiveness target
        self.physics_steps_per_frame = 2  # Physics calculations per frame
        self.visualization_quality = 2  # Default medium quality
        self.last_render_time = time.time()
        self.last_frame_time = time.time()
        self.trail_counter = 0

        self.reset_simulation()

        # Start the animation
        self.paused = False
        self.update_plot()  # Reference, not call

    def on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame"""
        self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        """Resize the inner frame to match the canvas"""
        width = event.width
        self.control_canvas.itemconfig("self.control_frame", width=width)

    def on_mousewheel(self, event):
        """Handle mousewheel scrolling"""
        # Check if mouse is over the control frame
        x, y = self.control_canvas.winfo_pointerxy()
        widget_under_mouse = self.control_canvas.winfo_containing(x, y)
        if widget_under_mouse and (widget_under_mouse == self.control_canvas or
                                   widget_under_mouse.master == self.control_frame or
                                   widget_under_mouse.master.master == self.control_frame):
            self.control_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def create_event_log(self):
        """Create an event log panel to display simulation events"""
        log_frame = ttk.LabelFrame(self.control_frame, text="Event Log")
        log_frame.pack(fill=tk.X, pady=10, padx=5)

        self.event_log = scrolledtext.ScrolledText(log_frame, width=30, height=8,
                                                   background="#21252b", foreground="#abb2bf",
                                                   font=("Consolas", 9))
        self.event_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.event_log.config(state=tk.DISABLED)

    def log_event(self, message):
        """Optimized event logging to reduce UI updates"""
        # Only log important messages or throttle frequent updates
        if "CRITICAL" in message or "COLLISION" in message or "INSTABILITY" in message:
            important = True
        else:
            important = False

        # Throttle non-important messages
        current_time = time.time()
        if hasattr(self, 'last_log_time'):
            if not important and current_time - self.last_log_time < 0.5:  # Limit to 2 logs per second
                return

        self.last_log_time = current_time

        # Make this method thread-safe by checking if it's called from the main thread
        if threading.current_thread() is threading.main_thread():
            # Called from main thread, update directly
            self._update_log(message)
        else:
            # Called from background thread, schedule update on main thread
            self.schedule_ui_update(self._update_log, message)

    def _update_log(self, message):
        """Internal method to update the log (always called from main thread)"""
        self.event_log.config(state=tk.NORMAL)
        self.event_log.insert(tk.END, f"[t={self.t:.2f}] {message}\n")
        self.event_log.see(tk.END)
        self.event_log.config(state=tk.DISABLED)

    def schedule_ui_update(self, func, *args, **kwargs):
        """Schedule a UI update on the main thread to avoid thread-safety issues"""
        self.root.after(0, lambda: func(*args, **kwargs))

    def init_deep_learning(self):
        # Initialize deep learning components
        self.rl_model = None
        self.optimizer = None
        self.training_in_progress = False
        self.training_thread = None
        self.best_params = None
        self.is_model_trained = False
        self.replay_buffer = deque(maxlen=10000)
        self.memory_states = []
        self.gamma = 0.99  # Discount factor

        # Define neural network architecture
        class StabilizationNetwork(nn.Module):
            def __init__(self, input_dim=18, hidden_dim=128, output_dim=18):
                super(StabilizationNetwork, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return torch.tanh(self.fc3(x)) * 0.1  # Small corrections

        # Create the model
        try:
            self.rl_model = StabilizationNetwork()
            self.optimizer = optim.Adam(self.rl_model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()
            self.model_ready = True
        except Exception as e:
            print(f"Could not initialize deep learning model: {e}")
            self.model_ready = False

    def setup_controls(self):
        # Title
        ttk.Label(self.control_frame, text="Simulation Controls", font=('Arial', 12, 'bold')).pack(pady=10)

        # Simulation controls
        sim_control_frame = ttk.Frame(self.control_frame)
        sim_control_frame.pack(fill=tk.X, pady=5)

        ttk.Button(sim_control_frame, text="Start/Pause", command=self.toggle_pause).pack(side=tk.LEFT, fill=tk.X,
                                                                                          expand=True, padx=2)
        ttk.Button(sim_control_frame, text="Reset", command=self.reset_simulation).pack(side=tk.LEFT, fill=tk.X,
                                                                                        expand=True, padx=2)
        ttk.Button(sim_control_frame, text="Step", command=self.step_simulation).pack(side=tk.LEFT, fill=tk.X,
                                                                                      expand=True, padx=2)

        # Simulation parameters
        param_frame = ttk.LabelFrame(self.control_frame, text="Parameters")
        param_frame.pack(fill=tk.X, pady=10)

        # Create a notebook for tabbed parameters
        param_notebook = ttk.Notebook(param_frame)
        param_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Basic parameters tab
        basic_tab = ttk.Frame(param_notebook)
        param_notebook.add(basic_tab, text="Basic")

        # Time step slider
        ttk.Label(basic_tab, text="Time Step:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.time_step_var = tk.DoubleVar(value=0.01)
        time_step_scale = ttk.Scale(basic_tab, from_=0.001, to=0.05,
                                    variable=self.time_step_var, orient=tk.HORIZONTAL)
        time_step_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Mass sliders
        ttk.Label(basic_tab, text="Mass Ratio (Body 2):").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.mass2_var = tk.DoubleVar(value=1.0)
        mass2_scale = ttk.Scale(basic_tab, from_=0.1, to=5.0,
                                variable=self.mass2_var, orient=tk.HORIZONTAL)
        mass2_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Label(basic_tab, text="Mass Ratio (Body 3):").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.mass3_var = tk.DoubleVar(value=1.0)
        mass3_scale = ttk.Scale(basic_tab, from_=0.1, to=5.0,
                                variable=self.mass3_var, orient=tk.HORIZONTAL)
        mass3_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Trail length slider
        ttk.Label(basic_tab, text="Trail Length:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.trail_length_var = tk.IntVar(value=100)
        trail_length_scale = ttk.Scale(basic_tab, from_=10, to=500,
                                       variable=self.trail_length_var, orient=tk.HORIZONTAL)
        trail_length_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Advanced parameters tab
        adv_tab = ttk.Frame(param_notebook)
        param_notebook.add(adv_tab, text="Advanced")

        # Gravitational constant
        ttk.Label(adv_tab, text="G (Gravitational Constant):").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.g_constant_var = tk.DoubleVar(value=1.0)
        g_scale = ttk.Scale(adv_tab, from_=0.1, to=2.0,
                            variable=self.g_constant_var, orient=tk.HORIZONTAL)
        g_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Collision detection radius multiplier
        ttk.Label(adv_tab, text="Collision Radius (multiplier):").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.collision_radius_var = tk.DoubleVar(value=0.05)  # Reduced default
        collision_scale = ttk.Scale(adv_tab, from_=0.01, to=0.5,
                                    variable=self.collision_radius_var, orient=tk.HORIZONTAL)
        collision_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # System bounds
        ttk.Label(adv_tab, text="System Bounds:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.system_bounds_var = tk.DoubleVar(value=10.0)
        bounds_scale = ttk.Scale(adv_tab, from_=5.0, to=50.0,
                                 variable=self.system_bounds_var, orient=tk.HORIZONTAL)
        bounds_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Softening parameter slider
        ttk.Label(adv_tab, text="Gravitational Softening:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.softening_var = tk.DoubleVar(value=0.01)
        softening_scale = ttk.Scale(adv_tab, from_=0.001, to=0.1,
                                    variable=self.softening_var, orient=tk.HORIZONTAL)
        softening_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Warmup period slider
        ttk.Label(adv_tab, text="Collision Warmup Period:").pack(anchor=tk.W, padx=5, pady=(10, 0))
        self.warmup_var = tk.DoubleVar(value=0.1)
        warmup_scale = ttk.Scale(adv_tab, from_=0.0, to=1.0,
                                 variable=self.warmup_var, orient=tk.HORIZONTAL)
        warmup_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Enable/disable collision detection
        self.collision_detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv_tab, text="Enable Collision Detection",
                        variable=self.collision_detection_var).pack(anchor=tk.W, padx=5, pady=5)

        # Enable/disable instability detection
        self.instability_detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adv_tab, text="Enable Instability Detection",
                        variable=self.instability_detection_var).pack(anchor=tk.W, padx=5, pady=5)

        # Add show axes option
        self.show_axes_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv_tab, text="Show Coordinate Axes",
                        variable=self.show_axes_var).pack(anchor=tk.W, padx=5, pady=5)

        # Add option to disable collision detection popups
        self.silent_collisions_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv_tab, text="Silent Collision Detection (Log Only)",
                        variable=self.silent_collisions_var).pack(anchor=tk.W, padx=5, pady=5)

        # Add an option to adjust collision visualization
        vis_frame = ttk.Frame(adv_tab)
        vis_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(vis_frame, text="Collision Sphere Transparency:").pack(side=tk.LEFT)
        self.collision_transparency_var = tk.DoubleVar(value=0.05)
        ttk.Scale(vis_frame, from_=0.01, to=0.2,
                  variable=self.collision_transparency_var,
                  orient=tk.HORIZONTAL).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # Individual body controls tab
        bodies_tab = ttk.Frame(param_notebook)
        param_notebook.add(bodies_tab, text="Bodies")

        # Create a sub-notebook for each body
        bodies_notebook = ttk.Notebook(bodies_tab)
        bodies_notebook.pack(fill=tk.BOTH, expand=True)

        # Create tabs for each body
        self.body_tabs = []
        self.body_position_vars = []
        self.body_velocity_vars = []

        for i in range(3):
            body_tab = ttk.Frame(bodies_notebook)
            bodies_notebook.add(body_tab, text=f"Body {i + 1}")
            self.body_tabs.append(body_tab)

            # Position controls
            pos_frame = ttk.LabelFrame(body_tab, text="Position")
            pos_frame.pack(fill=tk.X, padx=5, pady=5)

            pos_vars = []
            for axis, label in enumerate(['X', 'Y', 'Z']):
                ttk.Label(pos_frame, text=f"{label}:").pack(anchor=tk.W, padx=5, pady=(5 if axis == 0 else 2, 0))
                var = tk.DoubleVar(value=0.0)
                pos_vars.append(var)
                ttk.Scale(pos_frame, from_=-2.0, to=2.0, variable=var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5,
                                                                                                  pady=(0,
                                                                                                        5 if axis == 2 else 0))

            self.body_position_vars.append(pos_vars)

            # Velocity controls
            vel_frame = ttk.LabelFrame(body_tab, text="Velocity")
            vel_frame.pack(fill=tk.X, padx=5, pady=5)

            vel_vars = []
            for axis, label in enumerate(['VX', 'VY', 'VZ']):
                ttk.Label(vel_frame, text=f"{label}:").pack(anchor=tk.W, padx=5, pady=(5 if axis == 0 else 2, 0))
                var = tk.DoubleVar(value=0.0)
                vel_vars.append(var)
                ttk.Scale(vel_frame, from_=-1.0, to=1.0, variable=var, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5,
                                                                                                  pady=(0,
                                                                                                        5 if axis == 2 else 0))

            self.body_velocity_vars.append(vel_vars)

        # Stabilization methods
        stab_frame = ttk.LabelFrame(self.control_frame, text="Stabilization Methods")
        stab_frame.pack(fill=tk.X, pady=10)

        self.stab_method_var = tk.StringVar(value="none")
        ttk.Radiobutton(stab_frame, text="None (Chaotic)", variable=self.stab_method_var,
                        value="none").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(stab_frame, text="Lagrange Points", variable=self.stab_method_var,
                        value="lagrange").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(stab_frame, text="Figure-8 Orbit", variable=self.stab_method_var,
                        value="figure8").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(stab_frame, text="Euler's Three-Body", variable=self.stab_method_var,
                        value="euler").pack(anchor=tk.W, padx=5, pady=2)
        ttk.Radiobutton(stab_frame, text="Deep Learning", variable=self.stab_method_var,
                        value="deep_learning").pack(anchor=tk.W, padx=5, pady=2)

        # Damping control
        damping_frame = ttk.Frame(stab_frame)
        damping_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(damping_frame, text="Damping Factor:").pack(side=tk.LEFT)
        self.damping_var = tk.DoubleVar(value=0.995)
        ttk.Scale(damping_frame, from_=0.9, to=1.0, variable=self.damping_var,
                  orient=tk.HORIZONTAL).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=5)

        # Description of selected stabilization method
        self.stab_desc_frame = ttk.LabelFrame(self.control_frame, text="Method Description")
        self.stab_desc_frame.pack(fill=tk.X, pady=10)

        self.stab_desc_var = tk.StringVar(
            value="Current: No stabilization applied.\nThe system will exhibit chaotic behavior.")
        ttk.Label(self.stab_desc_frame, textvariable=self.stab_desc_var, wraplength=250).pack(padx=5, pady=5)

        # Apply button
        ttk.Button(self.control_frame, text="Apply Changes",
                   command=self.apply_changes).pack(fill=tk.X, pady=10)

        # Status display
        status_frame = ttk.LabelFrame(self.control_frame, text="System Status")
        status_frame.pack(fill=tk.X, pady=10)

        self.energy_var = tk.StringVar(value="Total Energy: 0.0")
        ttk.Label(status_frame, textvariable=self.energy_var).pack(anchor=tk.W, padx=5, pady=2)

        self.angular_momentum_var = tk.StringVar(value="Angular Momentum: 0.0")
        ttk.Label(status_frame, textvariable=self.angular_momentum_var).pack(anchor=tk.W, padx=5, pady=2)

        self.stability_var = tk.StringVar(value="Stability: Unknown")
        ttk.Label(status_frame, textvariable=self.stability_var).pack(anchor=tk.W, padx=5, pady=2)

        # Deep Learning controls - create as a notebook with tabs
        self.dl_frame = ttk.LabelFrame(self.control_frame, text="Deep Learning Controls")
        self.dl_frame.pack(fill=tk.X, pady=10)

        dl_notebook = ttk.Notebook(self.dl_frame)
        dl_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Training tab
        training_tab = ttk.Frame(dl_notebook)
        dl_notebook.add(training_tab, text="Training")

        # Training parameters
        ttk.Label(training_tab, text="Training Epochs:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.epochs_entry_var = tk.StringVar(value="100")
        epochs_entry = ttk.Entry(training_tab, textvariable=self.epochs_entry_var, width=10)
        epochs_entry.pack(anchor=tk.W, padx=5, pady=(0, 5))

        # Batch size
        ttk.Label(training_tab, text="Batch Size:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Scale(training_tab, from_=8, to=128, variable=self.batch_size_var,
                  orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=(0, 5))

        # Learning rate slider
        ttk.Label(training_tab, text="Learning Rate:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.lr_var = tk.DoubleVar(value=0.001)
        lr_scale = ttk.Scale(training_tab, from_=0.0001, to=0.01,
                             variable=self.lr_var, orient=tk.HORIZONTAL)
        lr_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Random search iterations
        ttk.Label(training_tab, text="Random Configurations to Try:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.random_configs_var = tk.IntVar(value=50)
        random_configs_scale = ttk.Scale(training_tab, from_=10, to=200,
                                         variable=self.random_configs_var, orient=tk.HORIZONTAL)
        random_configs_scale.pack(fill=tk.X, padx=5, pady=(0, 5))

        # Training status
        self.training_status_var = tk.StringVar(value="Model Status: Not Trained")
        ttk.Label(training_tab, textvariable=self.training_status_var).pack(anchor=tk.W, padx=5, pady=5)

        # Training progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(training_tab, variable=self.progress_var,
                                            maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

        # Train button
        self.train_button = ttk.Button(training_tab, text="Train Model",
                                       command=self.start_training)
        self.train_button.pack(fill=tk.X, padx=5, pady=5)

        # Apply trained model button
        self.apply_model_button = ttk.Button(training_tab, text="Apply Best Solution",
                                             command=self.apply_best_solution, state=tk.DISABLED)
        self.apply_model_button.pack(fill=tk.X, padx=5, pady=5)

        # Results tab
        results_tab = ttk.Frame(dl_notebook)
        dl_notebook.add(results_tab, text="Results")

        # Best parameters found
        ttk.Label(results_tab, text="Best Initial Conditions:",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=5, pady=(5, 0))

        self.best_params_text = scrolledtext.ScrolledText(results_tab, width=30, height=8,
                                                          background="#21252b", foreground="#98c379",
                                                          font=("Consolas", 9))
        self.best_params_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.best_params_text.config(state=tk.DISABLED)

        # Model tab
        model_tab = ttk.Frame(dl_notebook)
        dl_notebook.add(model_tab, text="Model")

        # Model parameters
        ttk.Label(model_tab, text="Neural Network Parameters:",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, padx=5, pady=(5, 0))

        self.model_params_text = scrolledtext.ScrolledText(model_tab, width=30, height=8,
                                                           background="#21252b", foreground="#61afef",
                                                           font=("Consolas", 9))
        self.model_params_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.model_params_text.config(state=tk.DISABLED)

        # Add event binding for stabilization method change
        self.stab_method_var.trace_add("write", self.update_stab_description)

    def update_stab_description(self, *args):
        method = self.stab_method_var.get()

        if method == "none":
            desc = "Current: No stabilization applied.\nThe system will exhibit chaotic behavior."
        elif method == "lagrange":
            desc = "Lagrange Points: Bodies form an equilateral triangle configuration, which is a stable solution when bodies have equal mass."
        elif method == "figure8":
            desc = "Figure-8 Orbit: A special periodic solution where three equal-mass bodies follow a figure-8 path while maintaining zero angular momentum."
        elif method == "euler":
            desc = "Euler's Three-Body: A collinear configuration where three bodies lie on a straight line and orbit around their common center of mass."
        elif method == "deep_learning":
            desc = "Deep Learning: Uses reinforcement learning to discover and maintain stable configurations by applying small corrective forces."

        self.stab_desc_var.set(desc)

    def setup_axes(self):
        """Set up the plot axes once to avoid repeated configurations"""
        self.ax.set_facecolor("#282c34")
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.xaxis.label.set_color("#abb2bf")
        self.ax.yaxis.label.set_color("#abb2bf")
        self.ax.zaxis.label.set_color("#abb2bf")
        self.ax.tick_params(colors="#abb2bf")
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Three-Body Problem Simulation", color="#abb2bf")

    def reset_simulation(self):
        self.t = 0
        self.collision_detected = False
        self.instability_detected = False
        self.instability_reason = ""
        self.time_since_last_event = 0

        # Add performance-related initialization settings
        if not hasattr(self, 'max_fps'):
            self.max_fps = 30  # Target maximum frames per second

        if not hasattr(self, 'target_fps'):
            self.target_fps = 60  # UI responsiveness target

        if not hasattr(self, 'physics_steps_per_frame'):
            self.physics_steps_per_frame = 2  # Physics calculations per frame

        if not hasattr(self, 'visualization_quality'):
            self.visualization_quality = 2  # Default medium quality

        # Clear cached artists and data
        if hasattr(self, 'body_artists'):
            del self.body_artists

        if hasattr(self, 'trail_artists'):
            del self.trail_artists

        # Add a warm-up period to avoid immediate collisions
        self.collision_warmup = self.warmup_var.get() if hasattr(self, 'warmup_var') else 0.1

        # Initialize based on selected stabilization method
        method = self.stab_method_var.get()

        if method == "lagrange":
            # L4/L5 Lagrange point configuration (equilateral triangle)
            self.bodies = np.array([
                # position (x, y, z) and velocity (vx, vy, vz) for 3 bodies
                [-0.5, 0, 0, 0, -0.5, 0],  # body 1
                [0.5, 0, 0, 0, 0.5, 0],  # body 2
                [0, 0.866, 0, -1.0, 0, 0]  # body 3 (at top of equilateral triangle)
            ])
            self.masses = np.array([1.0, 1.0, 1.0])
            self.stability_var.set("Stability: Stable (Equal Masses)")

        elif method == "figure8":
            # Initial conditions for the figure-8 solution (Chenciner & Montgomery)
            self.bodies = np.array([
                [0.97000436, -0.24308753, 0, 0.466203685, 0.43236573, 0],
                [-0.97000436, 0.24308753, 0, 0.466203685, 0.43236573, 0],
                [0, 0, 0, -0.93240737, -0.86473146, 0]
            ])
            self.masses = np.array([1.0, 1.0, 1.0])
            self.stability_var.set("Stability: Periodic Solution")

        elif method == "euler":
            # Euler's collinear solution
            self.bodies = np.array([
                [-1, 0, 0, 0, -0.3, 0],
                [0, 0, 0, 0, 0.6, 0],
                [1, 0, 0, 0, -0.3, 0]
            ])
            self.masses = np.array([1.0, 4.0, 1.0])
            self.stability_var.set("Stability: Conditionally Stable")

        elif method == "deep_learning" and self.is_model_trained and self.best_params is not None:
            # Use the learned best parameters
            self.bodies = self.best_params["bodies"].copy()
            self.masses = self.best_params["masses"].copy()
            self.stability_var.set("Stability: ML-Optimized")

        else:  # Random/chaotic initial conditions
            # Use the individual body controls if they've been set
            if hasattr(self, 'body_position_vars') and hasattr(self, 'body_velocity_vars'):
                # Check if all positions are close to zero (default values)
                all_zeros = True
                for i in range(3):
                    for j in range(3):
                        if abs(self.body_position_vars[i][j].get()) > 0.1:
                            all_zeros = False
                            break
                    if not all_zeros:
                        break

                if all_zeros:
                    # If positions are all near zero, set some reasonable defaults
                    # Triangular formation with significant spacing
                    default_positions = [
                        [1.0, 0.0, 0.0],  # Body 1 at (1, 0, 0)
                        [-0.5, 0.866, 0.0],  # Body 2 at (-0.5, 0.866, 0) (120° from body 1)
                        [-0.5, -0.866, 0.0]  # Body 3 at (-0.5, -0.866, 0) (240° from body 1)
                    ]
                    default_velocities = [
                        [0.0, 0.3, 0.0],  # Body 1 velocity
                        [0.3, -0.15, 0.0],  # Body 2 velocity
                        [-0.3, -0.15, 0.0]  # Body 3 velocity
                    ]

                    # Update the UI sliders with these default values
                    for i in range(3):
                        for j in range(3):
                            self.body_position_vars[i][j].set(default_positions[i][j])
                            self.body_velocity_vars[i][j].set(default_velocities[i][j])

                # Set the body positions and velocities from the UI variables
                self.bodies = np.zeros((3, 6))
                for i in range(3):
                    for j in range(3):
                        self.bodies[i, j] = self.body_position_vars[i][j].get()
                        self.bodies[i, j + 3] = self.body_velocity_vars[i][j].get()
            else:
                # Spread the bodies further apart in default configuration
                self.bodies = np.array([
                    [1.2, 0, 0, 0, 0.5, 0],  # body 1
                    [-0.6, 1.0, 0, -0.5, 0, 0],  # body 2
                    [-0.6, -1.0, 0, 0.5, 0, 0]  # body 3
                ])

            self.masses = np.array([1.0, self.mass2_var.get(), self.mass3_var.get()])

            if method == "deep_learning" and not self.is_model_trained:
                self.stability_var.set("Stability: ML (Training Required)")
            else:
                self.stability_var.set("Stability: Chaotic")

        # Set mass values in UI to match the configuration
        self.mass2_var.set(self.masses[1])
        self.mass3_var.set(self.masses[2])

        # Update the body position and velocity sliders if they exist
        if hasattr(self, 'body_position_vars') and hasattr(self, 'body_velocity_vars'):
            for i in range(3):
                for j in range(3):
                    self.body_position_vars[i][j].set(self.bodies[i, j])
                    self.body_velocity_vars[i][j].set(self.bodies[i, j + 3])

        # Create trail data
        self.trails = [[] for _ in range(3)]

        # Clear the plot
        self.ax.clear()
        self.setup_axes()

        # Reset the RL state
        self.current_state = self.bodies.flatten()
        self.memory_states = []

        # Initial energy and angular momentum
        energy, momentum = self.calculate_conserved_quantities()
        self.initial_energy = energy
        self.initial_momentum = momentum

        # Clear event log
        self.event_log.config(state=tk.NORMAL)
        self.event_log.delete(1.0, tk.END)
        self.event_log.config(state=tk.DISABLED)

        self.log_event("Simulation reset")
        self.log_event(f"Initial energy: {energy:.4f}")
        self.log_event(f"Initial ang. momentum: {momentum:.4f}")
        if self.collision_warmup > 0:
            self.log_event(f"Collision warmup period: {self.collision_warmup:.2f} time units")

    def step_simulation(self):
        """Perform a single step of the simulation"""
        if not self.paused and not self.collision_detected and not self.instability_detected:
            self.update_simulation()
            self.update_visualization()

    def toggle_pause(self):
        self.paused = not self.paused
        status = "paused" if self.paused else "resumed"
        self.log_event(f"Simulation {status}")

    def apply_changes(self):
        # Update masses if not using a special configuration
        if self.stab_method_var.get() == "none":
            self.masses = np.array([1.0, self.mass2_var.get(), self.mass3_var.get()])

        # Update body positions and velocities from sliders
        if hasattr(self, 'body_position_vars') and hasattr(self, 'body_velocity_vars'):
            for i in range(3):
                for j in range(3):
                    self.bodies[i, j] = self.body_position_vars[i][j].get()
                    self.bodies[i, j + 3] = self.body_velocity_vars[i][j].get()

        # Log the changes
        self.log_event("Applied parameter changes")
        self.log_event(f"Masses: {self.masses[0]:.1f}, {self.masses[1]:.1f}, {self.masses[2]:.1f}")

        # Reset collision and instability flags
        self.collision_detected = False
        self.instability_detected = False

        # Update warmup period
        if hasattr(self, 'warmup_var'):
            self.collision_warmup = self.warmup_var.get()

        # Update visualization
        self.update_visualization()

    def calculate_derivatives(self, t, state):
        """Optimized derivative calculation for physics"""
        # Reshape the state vector to work with our 3 bodies
        state = np.array(state).reshape(3, 6)
        derivatives = np.zeros_like(state)

        # Extract positions and velocities
        positions = state[:, :3]
        velocities = state[:, 3:6]

        # Set velocity components of derivatives
        derivatives[:, :3] = velocities

        # Calculate forces and accelerations with gravitational softening
        G = self.g_constant_var.get() if hasattr(self, 'g_constant_var') else 1.0
        softening = self.softening_var.get() if hasattr(self, 'softening_var') else 0.01

        # Optimize the acceleration calculation loop
        for i in range(3):
            # Vectorized approach for accelerations
            acc = np.zeros(3)
            for j in range(3):
                if i != j:
                    # Vector from body i to body j
                    r_ij = positions[j] - positions[i]

                    # Distance calculation with softening
                    distance_squared = np.sum(r_ij ** 2) + softening ** 2
                    distance = np.sqrt(distance_squared)

                    # Safety check and acceleration calculation
                    if distance > 1e-10:
                        acc += G * self.masses[j] * r_ij / (distance * distance_squared)

            # Apply the acceleration
            derivatives[i, 3:6] = acc

        # Apply other stabilization methods
        method = self.stab_method_var.get()

        if method == "deep_learning" and self.is_model_trained and self.rl_model:
            # Use the neural network to stabilize the system
            try:
                with torch.no_grad():
                    # Convert state to tensor
                    state_tensor = torch.FloatTensor(state.flatten())
                    # Get corrective actions from the model
                    corrections = self.rl_model(state_tensor).numpy()
                    # Apply the corrections (reshaped to match derivatives)
                    corrections_reshaped = corrections.reshape(3, 6)
                    # Only apply to velocity components
                    derivatives[:, 3:6] += corrections_reshaped[:, 3:6]
            except Exception as e:
                print(f"Error applying model corrections: {e}")

        # Apply other stabilization methods with customizable damping
        elif method != "none":
            damping = self.damping_var.get() if hasattr(self, 'damping_var') else 0.995
            center_of_mass = np.sum(positions * self.masses[:, np.newaxis], axis=0) / np.sum(self.masses)

            # Apply a small correction force toward the ideal configuration
            if method == "lagrange":
                # Small correction to maintain equilateral triangle
                derivatives[:, 3:6] *= damping

            elif method == "figure8":
                # Ensure zero angular momentum for figure-8
                angular_momentum = np.zeros(3)
                for i in range(3):
                    p_i = self.masses[i] * velocities[i]
                    angular_momentum += np.cross(positions[i] - center_of_mass, p_i)

                # Apply a small correction if angular momentum is not close to zero
                if np.linalg.norm(angular_momentum) > 0.01:
                    derivatives[:, 3:6] *= damping

            elif method == "euler":
                # Small correction to maintain collinearity
                derivatives[:, 3:6] *= damping

        return derivatives.flatten()

    def update_simulation(self):
        """Optimized simulation update method"""
        # Skip if collision or instability already detected
        if self.collision_detected or self.instability_detected:
            return

        # Integrate the equations of motion
        dt = self.time_step_var.get()
        state = self.bodies.flatten()

        # Check for NaN values before integration
        if np.any(np.isnan(state)):
            self.instability_detected = True
            self.instability_reason = "Numerical instability detected (NaN values)"
            self.log_event("CRITICAL: " + self.instability_reason)
            self.paused = True
            messagebox.showwarning("Simulation Stopped",
                                   f"The simulation has become numerically unstable.\n\n{self.instability_reason}")
            return

        try:
            # Set stricter tolerances and use a more robust method
            solution = scipy.integrate.solve_ivp(
                self.calculate_derivatives,
                [self.t, self.t + dt],
                state,
                method='RK45',
                rtol=1e-6,  # Relaxed tolerance for speed
                atol=1e-6  # Relaxed tolerance for speed
            )

            # Check if integration was successful
            if solution.success:
                # Check for NaN values in the solution
                if np.any(np.isnan(solution.y[:, -1])):
                    self.instability_detected = True
                    self.instability_reason = "Numerical instability in solution (NaN values)"
                    self.log_event("CRITICAL: " + self.instability_reason)
                    self.paused = True
                    messagebox.showwarning("Simulation Stopped",
                                           f"The simulation has become numerically unstable.\n\n{self.instability_reason}")
                    return

                # Update time and state
                self.t += dt
                self.time_since_last_event += dt
                new_state = solution.y[:, -1].reshape(3, 6)
            else:
                # Integration failed
                self.instability_detected = True
                self.instability_reason = f"Integration failed: {solution.message}"
                self.log_event("CRITICAL: " + self.instability_reason)
                self.paused = True
                messagebox.showwarning("Simulation Stopped",
                                       f"Integration failed.\n\n{solution.message}")
                return

            # Check for collisions if enabled (throttled)
            if self.collision_detection_var.get() and (not hasattr(self, 'last_collision_check') or
                                                       self.t - self.last_collision_check >= dt):
                self.last_collision_check = self.t
                collision = self.check_collisions(new_state)
                if collision:
                    # Only report collision if not in warmup period
                    if self.t < self.collision_warmup:
                        # Still in warmup, just log but don't stop
                        self.log_event(
                            f"Note: Bodies {collision[0] + 1} and {collision[1] + 1} would collide, but in warmup period")
                    else:
                        self.collision_detected = True
                        self.paused = True
                        self.log_event(f"COLLISION: Bodies {collision[0] + 1} and {collision[1] + 1} collided!")

                        # Only show the message box if silent collisions is disabled
                        if not hasattr(self, 'silent_collisions_var') or not self.silent_collisions_var.get():
                            messagebox.showinfo("Collision Detected",
                                                f"Bodies {collision[0] + 1} and {collision[1] + 1} have collided at t={self.t:.2f}!")

            # Check for instabilities less frequently
            if self.instability_detection_var.get() and (not hasattr(self, 'last_instability_check') or
                                                         self.t - self.last_instability_check >= 5 * dt):
                self.last_instability_check = self.t
                instability = self.check_instabilities(new_state)
                if instability:
                    self.instability_detected = True
                    self.instability_reason = instability
                    self.paused = True
                    self.log_event("INSTABILITY: " + instability)
                    messagebox.showwarning("Instability Detected",
                                           f"The system has become unstable at t={self.t:.2f}!\n\n{instability}")

            # Update the body state
            self.bodies = new_state

            # Update trails with throttling
            max_trail_length = self.trail_length_var.get()
            for i in range(3):
                pos = self.bodies[i, :3]

                # Only add points to trail at certain intervals to reduce memory usage
                if not hasattr(self, 'trail_counter'):
                    self.trail_counter = 0

                # Update trails based on visualization quality
                trail_update_freq = 4 - self.visualization_quality  # 1-3 for quality 3-1

                if self.trail_counter % trail_update_freq == 0:
                    self.trails[i].append(pos.copy())
                    if len(self.trails[i]) > max_trail_length:
                        self.trails[i].pop(0)

            self.trail_counter = (self.trail_counter + 1) % 10

            # Log events at reasonable intervals
            if self.time_since_last_event >= 5.0:  # Log every 5 time units
                energy, momentum = self.calculate_conserved_quantities()
                energy_diff = abs((energy - self.initial_energy) / (self.initial_energy + 1e-10)) * 100
                momentum_diff = abs((momentum - self.initial_momentum) / (self.initial_momentum + 1e-10)) * 100

                self.log_event(f"Energy change: {energy_diff:.2f}%")
                self.log_event(f"Momentum change: {momentum_diff:.2f}%")

                # Log position of only the first body to reduce log clutter
                pos = self.bodies[0, :3]
                self.log_event(f"Body 1 at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

                self.time_since_last_event = 0

        except Exception as e:
            self.instability_detected = True
            self.instability_reason = f"Integration error: {str(e)}"
            self.log_event("ERROR: " + self.instability_reason)
            self.paused = True
            messagebox.showerror("Simulation Error",
                                 f"An error occurred during simulation:\n\n{str(e)}")

    def check_collisions(self, state):
        """Optimized collision detection"""
        # Skip collision detection during warm-up period
        if self.t < self.collision_warmup:
            return None

        # Quick distance check using NumPy vectorization
        positions = state[:, :3]

        # Get collision radius from UI or use a small default
        collision_radius = 0.05
        if hasattr(self, 'collision_radius_var'):
            collision_radius = self.collision_radius_var.get()

        # Calculate collision radii based on mass (cube root of mass)
        radii = collision_radius * np.cbrt(self.masses)

        # Check distances between all pairs of bodies
        for i in range(3):
            for j in range(i + 1, 3):
                # Vector from body i to body j
                r_ij = positions[j] - positions[i]

                # Distance between bodies
                distance = np.linalg.norm(r_ij)

                # Sum of collision radii (with a small buffer)
                collision_distance = (radii[i] + radii[j])

                if distance < collision_distance:
                    return (i, j)  # Return the colliding pair

        return None

    def check_instabilities(self, state):
        """Check for various types of instabilities"""
        positions = state[:, :3]
        velocities = state[:, 3:6]

        # Check for NaN values
        if np.any(np.isnan(positions)) or np.any(np.isnan(velocities)):
            return "Numerical instability detected (NaN values in state)"

        # Check for extremely close bodies (potential singularity)
        for i in range(3):
            for j in range(i + 1, 3):
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < 1e-3:  # Very close bodies
                    return f"Bodies {i + 1} and {j + 1} are extremely close ({distance:.6f} units)"

        # Check for bodies escaping the system bounds
        bounds = self.system_bounds_var.get() if hasattr(self, 'system_bounds_var') else 10.0
        for i in range(3):
            if np.any(np.abs(positions[i]) > bounds):
                return f"Body {i + 1} has escaped the system bounds (>{bounds} units from origin)"

        # Check for extreme velocities
        max_velocity = 20.0  # Increased threshold to allow more dynamic motion
        for i in range(3):
            vel_magnitude = np.linalg.norm(velocities[i])
            if vel_magnitude > max_velocity:
                return f"Body {i + 1} has reached extreme velocity ({vel_magnitude:.2f} units/time)"

        # Check for conservation law violations
        try:
            energy, momentum = self.calculate_conserved_quantities()

            # Skip the check if initial values weren't set
            if not hasattr(self, 'initial_energy') or not hasattr(self, 'initial_momentum'):
                return None

            # Add small epsilon to avoid division by zero
            energy_change = abs((energy - self.initial_energy) / (abs(self.initial_energy) + 1e-10))
            momentum_change = abs((momentum - self.initial_momentum) / (abs(self.initial_momentum) + 1e-10))

            # Consider large changes in conservation laws as instability
            # Increased threshold to 70% to allow for more dynamics
            if energy_change > 0.7:  # 70% change in energy
                return f"Energy conservation violated ({energy_change * 100:.1f}% change from initial)"

            if momentum_change > 0.7:  # 70% change in angular momentum
                return f"Angular momentum conservation violated ({momentum_change * 100:.1f}% change from initial)"
        except Exception as e:
            return f"Error checking conservation laws: {str(e)}"

        return None  # No instability detected

    def update_visualization(self):
        """Optimized visualization update method"""
        # Skip full redraw if nothing has changed
        if hasattr(self, 'last_bodies_state') and np.array_equal(self.bodies,
                                                                 self.last_bodies_state) and not self.paused:
            return

        self.last_bodies_state = self.bodies.copy()

        # Only clear necessary elements instead of the full plot
        if not hasattr(self, 'body_artists'):
            self.ax.clear()
            self.body_artists = [None, None, None]
            self.trail_artists = [None, None, None]
            self.setup_axes()

        # Plot the bodies and their trails
        colors = ['#e06c75', '#98c379', '#61afef']

        for i in range(3):
            # Update trail with throttling
            if len(self.trails[i]) > 1:
                trail = np.array(self.trails[i])

                if self.trail_artists[i]:
                    # Update existing trail line
                    self.trail_artists[i].set_data(trail[:, 0], trail[:, 1])
                    self.trail_artists[i].set_3d_properties(trail[:, 2])
                else:
                    # Create new trail line
                    self.trail_artists[i], = self.ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], '-',
                                                          color=colors[i], alpha=0.5, linewidth=1.5)

            # Update body position
            pos = self.bodies[i, :3]
            size = 100 * (self.masses[i] ** (1 / 3))

            if self.body_artists[i]:
                # Update existing scatter
                self.body_artists[i]._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            else:
                # Create new scatter
                self.body_artists[i] = self.ax.scatter(pos[0], pos[1], pos[2], s=size, color=colors[i],
                                                       label=f"Body {i + 1} (m={self.masses[i]:.1f})")

        # Update title without full redraw
        method_names = {
            "none": "Chaotic System",
            "lagrange": "Lagrange Points",
            "figure8": "Figure-8 Orbit",
            "euler": "Euler's Three-Body",
            "deep_learning": "ML-Stabilized System"
        }
        method_name = method_names.get(self.stab_method_var.get(), "Three-Body Problem")

        # Add status to title
        status = ""
        if self.collision_detected:
            status = "- COLLISION DETECTED"
        elif self.instability_detected:
            status = "- UNSTABLE"

        # Add warmup indicator
        warmup = ""
        if self.t < self.collision_warmup:
            warmup = " (Warmup)"

        self.ax.set_title(f"{method_name}{warmup} - t={self.t:.2f} {status}", color="#abb2bf")

        # Add a center of mass marker
        com_pos = np.sum(self.bodies[:, :3] * self.masses[:, np.newaxis], axis=0) / np.sum(self.masses)
        #self.ax.scatter(com_pos[0], com_pos[1], com_pos[2], color='#ffffff',marker='x', s=50, label='Center of Mass')

        # Add coordinate axes at origin for reference
        if hasattr(self, 'show_axes_var') and self.show_axes_var.get():
            # X axis (red)
            self.ax.plot([0, 0.5], [0, 0], [0, 0], color='r', linewidth=2)
            # Y axis (green)
            self.ax.plot([0, 0], [0, 0.5], [0, 0], color='g', linewidth=2)
            # Z axis (blue)
            self.ax.plot([0, 0], [0, 0], [0, 0.5], color='b', linewidth=2)

        # Update plot limits to fit the bodies
        positions = self.bodies[:, :3]
        min_pos = positions.min(axis=0)
        max_pos = positions.max(axis=0)

        # Add some margin
        margin = 0.5
        min_pos -= margin
        max_pos += margin

        # Ensure the plot has equal scale on all axes
        max_range = max(max_pos - min_pos)
        center = (max_pos + min_pos) / 2

        self.ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
        self.ax.set_ylim(center[1] - max_range / 2, center[1] + max_range / 2)
        self.ax.set_zlim(center[2] - max_range / 2, center[2] + max_range / 2)

        # Add or update legend
        self.ax.legend(loc='upper right', facecolor="#3e4451", edgecolor="#abb2bf",
                       labelcolor="#abb2bf")

        # Redraw the canvas
        self.canvas.draw_idle()

    def update_plot(self):
        """Optimized main update function with throttled rendering"""
        current_time = time.time()

        # Only perform simulation updates if not paused and no issues detected
        if not self.paused and not self.collision_detected and not self.instability_detected:
            # Run multiple physics steps per frame for smoother simulation
            for _ in range(self.physics_steps_per_frame):
                self.update_simulation()

        # Throttle visualization updates based on elapsed time (limit framerate)
        if hasattr(self, 'last_render_time'):
            time_since_render = current_time - self.last_render_time
            if time_since_render >= 1.0 / self.max_fps:
                self.update_visualization()
                self.last_render_time = current_time
        else:
            self.update_visualization()
            self.last_render_time = current_time

        # Use a dynamic update rate based on system performance
        if hasattr(self, 'last_frame_time'):
            frame_time = current_time - self.last_frame_time
            # If frame took too long, reduce complexity
            if frame_time > 0.05:  # More than 50ms per frame
                self.reduce_visualization_complexity()
            # If frame was very fast, can increase complexity
            elif frame_time < 0.02 and self.visualization_quality < 3:  # Less than 20ms per frame
                self.increase_visualization_complexity()

        self.last_frame_time = current_time

        # Schedule the next update with a more responsive approach
        delay = max(1, int(1000 / self.target_fps))  # Ensure minimum 1ms delay
        self.root.after(delay, self.update_plot)

    def reduce_visualization_complexity(self):
        """Reduce rendering complexity to improve performance"""
        if not hasattr(self, 'visualization_quality'):
            self.visualization_quality = 2  # Medium quality default

        if self.visualization_quality > 0:
            self.visualization_quality -= 1
            # Apply quality reductions
            if self.visualization_quality == 0:  # Lowest quality
                self.trail_length_var.set(min(50, self.trail_length_var.get()))
                self.max_fps = 20
                self.physics_steps_per_frame = 1
            elif self.visualization_quality == 1:  # Low quality
                self.trail_length_var.set(min(100, self.trail_length_var.get()))
                self.max_fps = 30
                self.physics_steps_per_frame = 1
            # No need to log minor quality adjustments to avoid cluttering the log

    def increase_visualization_complexity(self):
        """Increase rendering complexity if performance allows"""
        if not hasattr(self, 'visualization_quality'):
            self.visualization_quality = 2  # Medium quality default

        if self.visualization_quality < 3:
            self.visualization_quality += 1
            # Apply quality increases
            if self.visualization_quality == 2:  # Medium quality
                self.max_fps = 40
                self.physics_steps_per_frame = 2
            elif self.visualization_quality == 3:  # High quality
                self.max_fps = 60
                self.physics_steps_per_frame = 3

    def calculate_conserved_quantities(self):
        # Calculate total energy and angular momentum
        positions = self.bodies[:, :3]
        velocities = self.bodies[:, 3:6]

        # Kinetic energy: 0.5 * m * v^2
        kinetic_energy = 0
        for i in range(3):
            v_squared = np.sum(velocities[i] ** 2)
            kinetic_energy += 0.5 * self.masses[i] * v_squared

        # Potential energy: -G * m1 * m2 / r
        potential_energy = 0
        G = self.g_constant_var.get() if hasattr(self, 'g_constant_var') else 1.0
        softening = self.softening_var.get() if hasattr(self, 'softening_var') else 0.01

        for i in range(3):
            for j in range(i + 1, 3):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                # Use softened potential for consistency with force calculation
                distance_squared = r_ij ** 2 + softening ** 2
                distance = np.sqrt(distance_squared)
                potential_energy -= G * self.masses[i] * self.masses[j] / distance

        total_energy = kinetic_energy + potential_energy

        # Angular momentum: r × p
        center_of_mass = np.sum(positions * self.masses[:, np.newaxis], axis=0) / np.sum(self.masses)
        angular_momentum = np.zeros(3)

        for i in range(3):
            r_i = positions[i] - center_of_mass
            p_i = self.masses[i] * velocities[i]
            angular_momentum += np.cross(r_i, p_i)

        angular_momentum_magnitude = np.linalg.norm(angular_momentum)

        # Update the display
        if hasattr(self, 'energy_var'):
            self.energy_var.set(f"Total Energy: {total_energy:.4f}")
        if hasattr(self, 'angular_momentum_var'):
            self.angular_momentum_var.set(f"Angular Momentum: {angular_momentum_magnitude:.4f}")

        return total_energy, angular_momentum_magnitude

    def calculate_reward(self, total_energy, angular_momentum, prev_energy=None, prev_angular_momentum=None):
        """Calculate reward for reinforcement learning"""
        # Check for numerical stability
        if np.isnan(total_energy) or np.isnan(angular_momentum):
            return -100  # Large negative reward for unstable states

        # Check if bodies are too far apart (system falling apart)
        positions = self.bodies[:, :3]
        max_distance = 0
        for i in range(3):
            for j in range(i + 1, 3):
                distance = np.linalg.norm(positions[i] - positions[j])
                max_distance = max(max_distance, distance)

        if max_distance > 10:
            return -50  # Penalize bodies moving too far apart

        # Reward for conservation of energy and angular momentum
        stability_reward = 0

        if prev_energy is not None and prev_angular_momentum is not None:
            energy_change = abs(total_energy - prev_energy)
            momentum_change = abs(angular_momentum - prev_angular_momentum)

            # Reward stability (less change is better)
            stability_reward = 10.0 * (1.0 / (1.0 + energy_change + momentum_change))

        # Reward for keeping bodies in a bounded region
        compactness_reward = 5.0 / (1.0 + max_distance)

        # Total reward
        return stability_reward + compactness_reward

    def start_training(self):
        """Start the deep learning training process"""
        if self.training_in_progress:
            messagebox.showinfo("Training in Progress", "Training is already running!")
            return

        if not self.model_ready:
            messagebox.showerror("Model Error", "Deep learning model could not be initialized!")
            return

        # Get the number of epochs from the entry field
        try:
            epochs = int(self.epochs_entry_var.get())
            if epochs <= 0:
                raise ValueError("Epochs must be positive")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid number of epochs: {str(e)}")
            return

        # Set up fresh model with current learning rate
        class StabilizationNetwork(nn.Module):
            def __init__(self, input_dim=18, hidden_dim=128, output_dim=18):
                super(StabilizationNetwork, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return torch.tanh(self.fc3(x)) * 0.1  # Small corrections

        self.rl_model = StabilizationNetwork()
        self.optimizer = optim.Adam(self.rl_model.parameters(), lr=self.lr_var.get())

        # Reset the progress bar
        self.progress_var.set(0)
        self.training_status_var.set("Model Status: Training...")
        self.train_button.config(state=tk.DISABLED)

        # Clear previous results
        self.best_params_text.config(state=tk.NORMAL)
        self.best_params_text.delete(1.0, tk.END)
        self.best_params_text.config(state=tk.DISABLED)

        self.model_params_text.config(state=tk.NORMAL)
        self.model_params_text.delete(1.0, tk.END)
        self.model_params_text.config(state=tk.DISABLED)

        # Log training start
        self.log_event(f"Starting training with {epochs} epochs")
        self.log_event(f"Learning rate: {self.lr_var.get()}")
        self.log_event(f"Batch size: {self.batch_size_var.get()}")
        self.log_event(f"Testing {self.random_configs_var.get()} configurations")

        # Start training in a separate thread to avoid blocking the GUI
        self.training_in_progress = True
        self.training_thread = threading.Thread(target=self.train_model, args=(epochs,))
        self.training_thread.daemon = True
        self.training_thread.start()

    def train_model(self, epochs):
        """Train the reinforcement learning model to stabilize the system"""
        try:
            random_configs = self.random_configs_var.get()
            batch_size = self.batch_size_var.get()
            best_reward = -float('inf')
            best_state = None

            # Save the current state to restore after training
            original_state = self.bodies.copy()
            original_masses = self.masses.copy()
            original_t = self.t

            # List to collect initial conditions with rewards
            init_conditions = []

            # Temporarily pause the simulation
            original_paused = self.paused
            self.paused = True

            # Try different initial conditions
            for config_idx in range(random_configs):
                # Generate random initial conditions
                bodies = np.random.uniform(-1, 1, (3, 6))
                masses = np.array([1.0,
                                   np.random.uniform(0.5, 2.0),
                                   np.random.uniform(0.5, 2.0)])

                # Test these conditions
                self.bodies = bodies.copy()
                self.masses = masses.copy()
                self.t = 0

                # Run a simulation to evaluate
                total_reward = 0
                prev_energy = None
                prev_angular_momentum = None

                # Simulate for a number of steps
                for step in range(50):
                    # Integrate one step
                    dt = 0.01
                    state = self.bodies.flatten()

                    # Get model prediction for this state if it's not the first iteration
                    if self.rl_model is not None and step > 0:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state)
                            action = self.rl_model(state_tensor).numpy()
                            # Apply small corrections from the model
                            state = state + action

                    try:
                        solution = scipy.integrate.solve_ivp(
                            self.calculate_derivatives,
                            [self.t, self.t + dt],
                            state,
                            method='RK45',
                            rtol=1e-6
                        )

                        # Update time and state
                        self.t += dt
                        self.bodies = solution.y[:, -1].reshape(3, 6)

                        # Calculate energy and angular momentum
                        energy, angular_momentum = self.calculate_conserved_quantities()

                        # Calculate reward
                        step_reward = self.calculate_reward(energy, angular_momentum,
                                                            prev_energy, prev_angular_momentum)
                        total_reward += step_reward

                        # Save previous values
                        prev_energy = energy
                        prev_angular_momentum = angular_momentum

                    except Exception as e:
                        # Handle integration errors
                        total_reward = -1000
                        break

                    # Check for instability
                    if np.isnan(energy) or np.any(np.isnan(self.bodies)):
                        total_reward = -1000
                        break

                # Record this configuration and its reward
                init_conditions.append({
                    "bodies": bodies.copy(),
                    "masses": masses.copy(),
                    "reward": total_reward
                })

                # Update the best solution if this one is better
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_state = {
                        "bodies": bodies.copy(),
                        "masses": masses.copy(),
                        "reward": total_reward
                    }

                    # Update best parameters display
                    self.schedule_ui_update(self.update_best_params_display, best_state)

                # Update training progress (use thread-safe method)
                progress = (config_idx + 1) / random_configs * 50  # First 50% for exploration
                self.schedule_ui_update(self.progress_var.set, progress)
                self.schedule_ui_update(self.training_status_var.set,
                                        f"Exploring: {config_idx + 1}/{random_configs}")

                # Log every 10 configurations (thread-safe)
                if (config_idx + 1) % 10 == 0 or config_idx == 0:
                    self.schedule_ui_update(self.log_event,
                                            f"Tested {config_idx + 1}/{random_configs} configs")
                    self.schedule_ui_update(self.log_event,
                                            f"Best reward so far: {best_reward:.2f}")

                # Let the GUI update
                time.sleep(0.01)

            # Train the neural network on the collected data
            self.train_neural_network(init_conditions, epochs, batch_size)

            # Store the best configuration found
            self.best_params = best_state

            # Restore original state
            self.bodies = original_state
            self.masses = original_masses
            self.t = original_t

            # Update UI (thread-safe)
            self.schedule_ui_update(self.training_status_var.set,
                                    f"Model Status: Trained (Best Reward: {best_reward:.2f})")
            self.schedule_ui_update(lambda: self.apply_model_button.config(state=tk.NORMAL))
            self.is_model_trained = True

            # Restore pause state
            self.paused = original_paused

            # Final log (thread-safe)
            self.schedule_ui_update(self.log_event, f"Training complete! Best reward: {best_reward:.2f}")

        except Exception as e:
            # Handle errors (thread-safe)
            self.schedule_ui_update(self.training_status_var.set, f"Training Error: {str(e)[:30]}...")
            self.schedule_ui_update(self.log_event, f"ERROR during training: {str(e)}")
            print(f"Training error: {e}")

        finally:
            self.training_in_progress = False
            self.schedule_ui_update(lambda: self.train_button.config(state=tk.NORMAL))

    def update_best_params_display(self, best_state):
        """Update the display of best parameters found during training"""
        if not best_state:
            return

        # Format the best parameters as text
        text = "Bodies:\n"
        for i in range(3):
            pos = best_state["bodies"][i, :3]
            vel = best_state["bodies"][i, 3:6]
            text += f"Body {i + 1}:\n"
            text += f"  Position: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})\n"
            text += f"  Velocity: ({vel[0]:.4f}, {vel[1]:.4f}, {vel[2]:.4f})\n"

        text += "\nMasses:\n"
        for i, mass in enumerate(best_state["masses"]):
            text += f"  Body {i + 1}: {mass:.4f}\n"

        text += f"\nReward: {best_state['reward']:.4f}"

        # Update the text widget
        self.best_params_text.config(state=tk.NORMAL)
        self.best_params_text.delete(1.0, tk.END)
        self.best_params_text.insert(tk.END, text)
        self.best_params_text.config(state=tk.DISABLED)

    def update_model_params_display(self):
        """Update the display of neural network model parameters"""
        if not self.rl_model:
            return

        # Get model parameters
        model_info = "Network Architecture:\n"
        model_info += "  Input → 128 → 128 → Output\n\n"

        model_info += "Layer Weights (sample):\n"

        # Get some representative weights from each layer
        try:
            # Layer 1 (input to hidden)
            fc1_weights = self.rl_model.fc1.weight.data.flatten()
            model_info += "  Layer 1 mean: {:.6f}\n".format(torch.mean(fc1_weights).item())
            model_info += "  Layer 1 std: {:.6f}\n".format(torch.std(fc1_weights).item())
            model_info += "  Layer 1 min: {:.6f}\n".format(torch.min(fc1_weights).item())
            model_info += "  Layer 1 max: {:.6f}\n\n".format(torch.max(fc1_weights).item())

            # Layer 2 (hidden to hidden)
            fc2_weights = self.rl_model.fc2.weight.data.flatten()
            model_info += "  Layer 2 mean: {:.6f}\n".format(torch.mean(fc2_weights).item())
            model_info += "  Layer 2 std: {:.6f}\n\n".format(torch.std(fc2_weights).item())

            # Layer 3 (hidden to output)
            fc3_weights = self.rl_model.fc3.weight.data.flatten()
            model_info += "  Layer 3 mean: {:.6f}\n".format(torch.mean(fc3_weights).item())
            model_info += "  Layer 3 std: {:.6f}\n".format(torch.std(fc3_weights).item())

        except Exception as e:
            model_info += f"\nError getting weights: {str(e)}"

        # Update the text widget
        self.model_params_text.config(state=tk.NORMAL)
        self.model_params_text.delete(1.0, tk.END)
        self.model_params_text.insert(tk.END, model_info)
        self.model_params_text.config(state=tk.DISABLED)

    def train_neural_network(self, init_conditions, epochs, batch_size):
        """Train the neural network on collected data"""
        # Make sure we have data to train on
        if not init_conditions:
            self.schedule_ui_update(self.log_event, "No valid conditions found for training")
            return

        # Sort conditions by reward (best first)
        sorted_conditions = sorted(init_conditions, key=lambda x: x["reward"], reverse=True)

        # Keep only the top 50% of conditions for training
        top_conditions = sorted_conditions[:len(sorted_conditions) // 2]
        if not top_conditions:
            self.schedule_ui_update(self.log_event, "No positive reward conditions found")
            return

        # Prepare training data
        states = []
        targets = []

        for condition in top_conditions:
            # The state is the flattened initial position and velocity
            state = condition["bodies"].flatten()

            # The target is the same state (identity function for stable configurations)
            # or a slightly modified state for unstable ones
            target = state.copy()

            # Make sure we have valid reward data
            if "reward" not in condition:
                continue

            # For lower reward conditions, add some damping to stabilize
            try:
                max_reward = max(c["reward"] for c in top_conditions)
                if max_reward <= 0:
                    # All rewards are negative, normalize differently
                    reward_normalized = 1.0 - abs(condition["reward"] / min(c["reward"] for c in top_conditions))
                else:
                    reward_normalized = condition["reward"] / max_reward

                if reward_normalized < 0.8:
                    # Apply damping to velocities (elements 3, 4, 5, 9, 10, 11, 15, 16, 17)
                    velocity_indices = [3, 4, 5, 9, 10, 11, 15, 16, 17]
                    for idx in velocity_indices:
                        target[idx] *= (0.9 + 0.1 * reward_normalized)  # More damping for worse configurations
            except Exception as e:
                self.schedule_ui_update(self.log_event, f"Warning: {str(e)}")
                continue

            states.append(state)
            targets.append(target)

        # Check if we have any valid data
        if not states:
            self.schedule_ui_update(self.log_event, "No valid training data generated")
            return

        # Convert to tensors
        X = torch.FloatTensor(np.array(states))
        y = torch.FloatTensor(np.array(targets))

        # Create a simple dataset and dataloader
        try:
            dataset = torch.utils.data.TensorDataset(X, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        except Exception as e:
            self.schedule_ui_update(self.log_event, f"Error creating dataloader: {str(e)}")
            return

        # Check if we have a valid dataloader
        if len(dataloader) == 0:
            self.schedule_ui_update(self.log_event, "Empty dataloader, cannot train")
            return

        # Train the model
        self.rl_model.train()

        total_batches = epochs * len(dataloader)
        batch_count = 0
        running_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.rl_model(batch_X)
                loss = self.criterion(outputs, batch_y - batch_X)  # Learn the correction
                loss.backward()
                self.optimizer.step()

                # Update progress
                batch_count += 1
                epoch_loss += loss.item()
                running_loss = 0.9 * running_loss + 0.1 * loss.item() if running_loss > 0 else loss.item()

                # Update progress display (thread-safe)
                progress = 50 + (batch_count / total_batches * 50)  # Second 50% for training
                self.schedule_ui_update(self.progress_var.set, progress)
                self.schedule_ui_update(self.training_status_var.set,
                                        f"Training: Epoch {epoch + 1}/{epochs} (Loss: {running_loss:.6f})")

                # Allow GUI to update
                if batch_count % 5 == 0:
                    time.sleep(0.01)

            # Log epoch results (thread-safe)
            if len(dataloader) > 0:
                avg_loss = epoch_loss / len(dataloader)
                self.schedule_ui_update(self.log_event,
                                        f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        # Update model parameters display (thread-safe)
        self.schedule_ui_update(self.update_model_params_display)

    def apply_best_solution(self):
        """Apply the best solution found during training"""
        if self.best_params is None:
            messagebox.showinfo("No Solution",
                                "No stable solution has been found yet. Please train the model first.")
            return

        # Set the stabilization method to deep learning
        self.stab_method_var.set("deep_learning")

        # Reset the simulation with the best parameters
        self.reset_simulation()


if __name__ == "__main__":
    root = tk.Tk()
    app = ThreeBodySimulator(root)


    # Set up a proper exit handler
    def on_closing():
        if hasattr(app, 'training_thread') and app.training_thread and app.training_thread.is_alive():
            # Wait for training to complete or timeout
            app.training_thread.join(1.0)
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()