"""
Base visualization class for Traffic Analyzer App
"""
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from UI.utils.constants import MESSAGES

class BaseVisualization:
    """Base class for all visualizations with common functionality"""
    
    def __init__(self, graph_frame, stats_panel, styles, db):
        self.graph_frame = graph_frame
        self.stats_panel = stats_panel
        self.styles = styles
        self.colors = styles.colors
        self.db = db
    
    def clear_visualization(self):
        """Clear existing visualization content"""
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        if self.stats_panel:
            self.stats_panel.clear_stats()
    
    def clear_placeholder(self):
        """Clear any placeholder content from graph frame"""
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
    
    def show_no_data_message(self, custom_message=None):
        """Show modern no data message"""
        message = custom_message or MESSAGES['no_data']
        
        # Clear existing content
        self.clear_visualization()
        
        # Create modern no data card
        no_data_frame = tk.Frame(self.graph_frame, bg=self.colors['surface'], relief='solid', bd=1)
        no_data_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)
        
        # Icon and message
        icon_label = tk.Label(no_data_frame, text="📊",
                             font=('Segoe UI', 48),
                             bg=self.colors['surface'],
                             fg=self.colors['muted'])
        icon_label.pack(expand=True, pady=(20, 10))
        
        message_label = tk.Label(no_data_frame, text=message,
                               font=('Segoe UI', 14),
                               bg=self.colors['surface'],
                               fg=self.colors['muted'],
                               justify=tk.CENTER)
        message_label.pack(expand=True, pady=(0, 10))
    
    def create_chart_canvas(self, fig):
        """Create and display chart canvas"""
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        return canvas
    
    def style_axes(self, ax, xlabel, ylabel, title):
        """Apply common styling to matplotlib axes"""
        # Apply modern axes styling
        self.styles.apply_modern_axes_style(ax)
        
        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_title(title, fontsize=16, fontweight='bold', color=self.colors['primary'], pad=20)
        
        return ax
    
    def add_value_labels_to_bars(self, ax, bars, values):
        """Add value labels on top of bars"""
        max_value = max(values) if values else 1
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + max_value * 0.02,
                       f'{int(height):,}',
                       ha='center', va='bottom',
                       fontsize=10, fontweight='bold',
                       color=self.colors['dark'])
    
    def create_modern_legend(self, ax, loc='upper left'):
        """Create modern styled legend"""
        return self.styles.create_modern_legend(ax, loc)
    
    def generate_visualization(self, date, view_type):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_visualization method")
    
    def generate_stats(self, *args, **kwargs):
        """Abstract method to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement generate_stats method")