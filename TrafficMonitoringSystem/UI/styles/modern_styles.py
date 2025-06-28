"""
Modern styling configuration for Traffic Analyzer App
"""
import matplotlib.pyplot as plt
from tkinter import ttk

class ModernStyles:
    """Modern color scheme and styling configuration"""
    
    def __init__(self):
        # Modern color palette
        self.colors = {
            'primary': '#1e40af',      # Modern blue
            'secondary': '#64748b',    # Slate gray
            'success': '#059669',      # Emerald green
            'danger': '#dc2626',       # Red
            'warning': '#d97706',      # Amber
            'info': '#0ea5e9',         # Sky blue
            'light': '#f8fafc',        # Very light gray
            'dark': '#1e293b',         # Dark slate
            'background': '#f1f5f9',   # Light blue-gray
            'surface': '#ffffff',      # White
            'accent': '#8b5cf6',       # Purple
            'muted': '#6b7280'         # Gray
        }
    
    def setup_matplotlib_style(self):
        """Configure matplotlib with modern styling"""
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Segoe UI',
            'font.size': 10,
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'figure.facecolor': self.colors['surface'],
            'axes.facecolor': self.colors['light'],
            'axes.edgecolor': self.colors['muted'],
            'axes.linewidth': 0.8,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5
        })
    
    def setup_ttk_styles(self, root):
        """Configure ttk styles with modern theme"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure root background
        root.configure(bg=self.colors['background'])
        
        # Modern Combobox style
        style.configure('Modern.TCombobox',
                       font=('Segoe UI', 10),
                       padding=8,
                       fieldbackground=self.colors['light'],
                       borderwidth=1,
                       relief='solid')
        
        return style
    
    def get_button_hover_effects(self):
        """Return hover effect functions for buttons"""
        def on_enter(event, btn, color):
            btn.configure(bg=color)
        
        def on_leave(event, btn, original_color):
            btn.configure(bg=original_color)
        
        return on_enter, on_leave
    
    def create_modern_figure(self, width=8, height=6):
        """Create modern styled matplotlib figure"""
        from matplotlib.figure import Figure
        fig = Figure(figsize=(width, height), 
                    facecolor=self.colors['surface'],
                    edgecolor='none',
                    tight_layout=True)
        
        fig.patch.set_linewidth(0)
        return fig
    
    def apply_modern_axes_style(self, ax):
        """Apply modern styling to matplotlib axes"""
        ax.set_facecolor(self.colors['light'])
        
        # Modern grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['muted'])
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        
        return ax
    
    def create_modern_legend(self, ax, loc='upper left'):
        """Create modern styled legend"""
        legend = ax.legend(loc=loc, frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor=self.colors['muted'])
        legend.get_frame().set_facecolor(self.colors['surface'])
        return legend