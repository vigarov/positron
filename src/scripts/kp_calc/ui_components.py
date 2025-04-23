import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def setup_ui(app):
    """Set up the user interface components for the application"""
    main_container = ttk.Frame(app.root)
    main_container.pack(fill=tk.BOTH, expand=True)
    
    app.canvas_main = tk.Canvas(main_container)
    scrollbar = ttk.Scrollbar(main_container, orient=tk.VERTICAL, command=app.canvas_main.yview)
    app.canvas_main.configure(yscrollcommand=scrollbar.set)
    
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    app.canvas_main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    app.main_frame = ttk.Frame(app.canvas_main)
    app.canvas_window = app.canvas_main.create_window((0, 0), window=app.main_frame, anchor=tk.NW)
    
    app.main_frame.bind("<Configure>", lambda event: on_frame_configure(app, event))
    app.canvas_main.bind("<Configure>", lambda event: on_canvas_configure(app, event))
    
    app.control_panel = ttk.Frame(app.main_frame)
    app.control_panel.pack(fill=tk.X, pady=5)
    
    ttk.Label(app.control_panel, text="Image Set:").pack(side=tk.LEFT, padx=5)
    app.set_var = tk.StringVar(value=str(app.current_set_id))
    set_values = list(app.image_mappings.keys())
    app.set_combo = ttk.Combobox(app.control_panel, textvariable=app.set_var, 
                                 values=set_values, width=5, state="readonly")
    app.set_combo.pack(side=tk.LEFT, padx=5)
    app.set_combo.bind("<<ComboboxSelected>>", app.on_set_changed)
    
    ttk.Label(app.control_panel, text="Source:").pack(side=tk.LEFT, padx=5)
    app.source_var = tk.StringVar(value="Digital (DA)")
    sources = ["Digital (DA)", "Negative 1", "Negative 2"]
    app.source_combo = ttk.Combobox(app.control_panel, textvariable=app.source_var, 
                                    values=sources, width=12, state="disabled")
    app.source_combo.pack(side=tk.LEFT, padx=5)
    
    app.clear_rect_btn = ttk.Button(app.control_panel, text="Clear Rectangle", 
                                   command=app.clear_rectangle)
    app.extract_pixels_btn = ttk.Button(app.control_panel, text="Extract Pixels", 
                                       command=app.extract_pixels)
    app.save_results_btn = ttk.Button(app.control_panel, text="Save Results", 
                                     command=app.save_results)
    
    app.status_var = tk.StringVar(value="Ready. Draw rectangle on the Digital (DA) image.")
    ttk.Label(app.control_panel, textvariable=app.status_var).pack(side=tk.LEFT, padx=20)
    
    app.img_frame = ttk.Frame(app.main_frame)
    app.img_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    app.canvas = tk.Canvas(app.img_frame, bg="#f0f0f0", width=app.display_width, height=app.display_height)
    app.canvas.pack(fill=tk.BOTH, expand=True)
    
    app.canvas.bind("<Configure>", lambda event: on_canvas_resize(app, event))
    
    app.canvas.bind("<ButtonPress-1>", app.on_mouse_down)
    app.canvas.bind("<B1-Motion>", app.on_mouse_move)
    app.canvas.bind("<ButtonRelease-1>", app.on_mouse_up)
    
    app.results_frame = ttk.Frame(app.main_frame)
    app.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    app.results_tabs = ttk.Notebook(app.results_frame)
    app.results_tabs.pack(fill=tk.BOTH, expand=True)
    
    app.hist_tab = ttk.Frame(app.results_tabs)
    app.results_tabs.add(app.hist_tab, text="Histograms")
    
    app.fig = plt.Figure(figsize=(15, 6), dpi=100)
    app.canvas_fig = FigureCanvasTkAgg(app.fig, master=app.hist_tab)
    app.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    app.waveform_tab = ttk.Frame(app.results_tabs)
    app.results_tabs.add(app.waveform_tab, text="Waveforms")
    
    app.waveform_fig = plt.Figure(figsize=(16, 10), dpi=150)
    app.waveform_canvas = FigureCanvasTkAgg(app.waveform_fig, master=app.waveform_tab)
    app.waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    setup_raw_data_tab(app)
    
    app.all_rectangles_drawn = False
    app.pixels_extracted = False
    
    app.results_tabs.pack_forget()

def setup_raw_data_tab(app):
    """Set up the raw data tab with CSV display"""
    app.raw_data_tab = ttk.Frame(app.results_tabs)
    app.results_tabs.add(app.raw_data_tab, text="Raw Data")
    
    raw_data_frame = ttk.Frame(app.raw_data_tab)
    raw_data_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    raw_data_scroll_y = ttk.Scrollbar(raw_data_frame, orient=tk.VERTICAL)
    raw_data_scroll_x = ttk.Scrollbar(raw_data_frame, orient=tk.HORIZONTAL)
    
    app.raw_data_text = tk.Text(raw_data_frame, wrap=tk.NONE, height=25, width=80,
                               yscrollcommand=raw_data_scroll_y.set,
                               xscrollcommand=raw_data_scroll_x.set,
                               font=("Courier New", 10))
    
    raw_data_scroll_y.config(command=app.raw_data_text.yview)
    raw_data_scroll_x.config(command=app.raw_data_text.xview)
    
    raw_data_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
    raw_data_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
    app.raw_data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    raw_data_controls = ttk.Frame(app.raw_data_tab)
    raw_data_controls.pack(fill=tk.X, pady=5)
    
    ttk.Button(raw_data_controls, text="Copy to Clipboard", 
              command=app.copy_raw_data_to_clipboard).pack(side=tk.LEFT, padx=5)
    
    ttk.Label(raw_data_controls, text="Data shows all three images side by side for direct comparison.", 
              font=("Arial", 10)).pack(side=tk.LEFT, padx=20)

def on_frame_configure(app, event):
    """Update the scroll region when the inner frame changes size"""
    app.canvas_main.configure(scrollregion=app.canvas_main.bbox("all"))

def on_canvas_configure(app, event):
    """Update the inner frame's width to fill the canvas"""
    canvas_width = event.width
    app.canvas_main.itemconfig(app.canvas_window, width=canvas_width)

def on_canvas_resize(app, event):
    """Handle canvas resize events to redraw the image at proper size"""
    # When the canvas is resized, reload the current image to center it properly
    if hasattr(app, 'current_img_rgb') and app.current_img_rgb is not None:
        from image_processor import load_current_image
        load_current_image(app) 