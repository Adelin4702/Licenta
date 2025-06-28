"""
Statistics panel component for Traffic Analyzer App
"""
import tkinter as tk
from tkinter import ttk
from UI.utils.constants import STATS_PANEL_WIDTH, CARD_HEADER_HEIGHT, FONT_SIZES

class StatsPanelComponent:
    """Scrollable statistics panel for displaying detailed text-based stats"""
    
    def __init__(self, parent, styles):
        self.parent = parent
        self.styles = styles
        self.colors = styles.colors
        
        self.create_stats_panel()
    
    def create_stats_panel(self):
        """Create statistics panel with scrollable content"""
        # Stats container with fixed width
        self.stats_container = tk.Frame(self.parent, bg=self.colors['surface'], width=STATS_PANEL_WIDTH)
        self.stats_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        self.stats_container.pack_propagate(False)
        
        # Stats header
        self._create_stats_header()
        
        # Scrollable content area
        self._create_scrollable_content()
        
        # Show default message
        self.show_default_message()
        
        return self.stats_container
    
    def _create_stats_header(self):
        """Create statistics panel header"""
        stats_header = tk.Frame(self.stats_container, bg=self.colors['primary'], height=CARD_HEADER_HEIGHT)
        stats_header.pack(fill=tk.X)
        stats_header.pack_propagate(False)
        
        tk.Label(
            stats_header, 
            text="📊 Statistici",
            font=('Segoe UI', FONT_SIZES['card_header'], 'bold'),
            bg=self.colors['primary'],
            fg='white'
        ).pack(expand=True)
    
    def _create_scrollable_content(self):
        """Create scrollable content area for statistics"""
        # Canvas and scrollbar setup
        self.canvas = tk.Canvas(self.stats_container, bg=self.colors['surface'], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.stats_container, orient="vertical", command=self.canvas.yview)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Stats content frame
        self.text_stats_frame = tk.Frame(self.canvas, bg=self.colors['surface'])
        self.canvas_window = self.canvas.create_window((0, 0), window=self.text_stats_frame, anchor="nw")
        
        # Configure scrolling
        self._setup_scroll_configuration()
        
        # Mouse wheel scrolling
        self._setup_mouse_wheel_scrolling()
    
    def _setup_scroll_configuration(self):
        """Setup scroll region configuration"""
        def configure_scroll(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.canvas.itemconfig(self.canvas_window, width=event.width)
        
        self.canvas.bind('<Configure>', configure_scroll)
        self.text_stats_frame.bind('<Configure>', 
                                  lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
    
    def _setup_mouse_wheel_scrolling(self):
        """Setup mouse wheel scrolling for the canvas"""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind to multiple widgets to ensure scrolling works
        widgets_to_bind = [self.canvas, self.text_stats_frame, self.stats_container]
        for widget in widgets_to_bind:
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))
            widget.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
    
    def show_default_message(self):
        """Show default message when no stats are available"""
        self.clear_stats()
        
        default_label = tk.Label(
            self.text_stats_frame,
            text="📈 Statisticile detaliate vor apărea aici după selectarea unei vizualizări.",
            font=('Segoe UI', FONT_SIZES['normal']),
            bg=self.colors['surface'],
            fg=self.colors['muted'],
            wraplength=220,
            justify=tk.CENTER
        )
        default_label.pack(expand=True, pady=20)
    
    def clear_stats(self):
        """Clear all statistics content"""
        for widget in self.text_stats_frame.winfo_children():
            widget.destroy()
    
    def display_stats(self, stats_text):
        """Display statistics text in the panel"""
        self.clear_stats()
        
        stats_label = tk.Label(
            self.text_stats_frame,
            text=stats_text,
            font=('Segoe UI', FONT_SIZES['small']),
            bg=self.colors['surface'],
            fg=self.colors['dark'],
            justify=tk.LEFT,
            anchor='nw',
            wraplength=220
        )
        stats_label.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Update scroll region after adding content
        self.text_stats_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def add_stats_section(self, title, content, title_color=None):
        """Add a formatted statistics section"""
        if not title_color:
            title_color = self.colors['primary']
        
        # Title
        title_label = tk.Label(
            self.text_stats_frame,
            text=title,
            font=('Segoe UI', FONT_SIZES['normal'], 'bold'),
            bg=self.colors['surface'],
            fg=title_color,
            wraplength=220,
            justify=tk.LEFT
        )
        title_label.pack(anchor='w', padx=5, pady=(10, 5))
        
        # Content
        content_label = tk.Label(
            self.text_stats_frame,
            text=content,
            font=('Segoe UI', FONT_SIZES['small']),
            bg=self.colors['surface'],
            fg=self.colors['dark'],
            justify=tk.LEFT,
            anchor='nw',
            wraplength=200
        )
        content_label.pack(anchor='w', padx=15, pady=(0, 5))
        
        # Update scroll region
        self.text_stats_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def add_metric_card(self, title, value, unit="", icon="📊", color=None):
        """Add a styled metric card"""
        if not color:
            color = self.colors['primary']
        
        # Card frame
        card_frame = tk.Frame(self.text_stats_frame, bg=self.colors['light'], relief='solid', bd=1)
        card_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Icon and title
        header_frame = tk.Frame(card_frame, bg=self.colors['light'])
        header_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        
        tk.Label(header_frame, text=f"{icon} {title}", 
                font=('Segoe UI', FONT_SIZES['small'], 'bold'),
                bg=self.colors['light'], fg=self.colors['dark']).pack(anchor='w')
        
        # Value
        value_frame = tk.Frame(card_frame, bg=self.colors['light'])
        value_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        tk.Label(value_frame, text=f"{value} {unit}", 
                font=('Segoe UI', FONT_SIZES['normal'], 'bold'),
                bg=self.colors['light'], fg=color).pack(anchor='w')
        
        # Update scroll region
        self.text_stats_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def add_divider(self):
        """Add a visual divider"""
        divider = tk.Frame(self.text_stats_frame, height=1, bg=self.colors['muted'])
        divider.pack(fill=tk.X, padx=10, pady=8)
    
    def get_stats_frame(self):
        """Get the stats frame for custom content"""
        return self.text_stats_frame
    
    def scroll_to_top(self):
        """Scroll to the top of the stats panel"""
        self.canvas.yview_moveto(0)