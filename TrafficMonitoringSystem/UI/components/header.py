"""
Header component for Traffic Analyzer App
"""
import tkinter as tk
from utils.constants import APP_TITLE, APP_SUBTITLE, HEADER_HEIGHT, FONT_SIZES

class HeaderComponent:
    """Modern header component with title and status"""
    
    def __init__(self, parent, styles, dates_count=0):
        self.parent = parent
        self.styles = styles
        self.colors = styles.colors
        self.dates_count = dates_count
        
        self.create_header()
    
    def create_header(self):
        """Create modern header with gradient effect"""
        # Main header frame
        self.header_frame = tk.Frame(self.parent, bg=self.colors['surface'], height=HEADER_HEIGHT)
        self.header_frame.pack(fill=tk.X)
        self.header_frame.pack_propagate(False)
        
        # Add subtle shadow effect
        shadow_frame = tk.Frame(self.parent, bg='#e2e8f0', height=2)
        shadow_frame.pack(fill=tk.X)
        
        # Header content container
        header_content = tk.Frame(self.header_frame, bg=self.colors['surface'])
        header_content.pack(expand=True, fill=tk.BOTH, padx=40, pady=20)
        
        self._create_title_section(header_content)
        self._create_status_section(header_content)
        
        return self.header_frame
    
    def _create_title_section(self, parent):
        """Create title and subtitle"""
        # Main title
        self.title_label = tk.Label(
            parent, 
            text=APP_TITLE,
            font=(self.styles.colors.get('font_family', 'Segoe UI'), FONT_SIZES['title'], 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['primary']
        )
        self.title_label.pack(anchor='w')
        
        # Subtitle
        self.subtitle_label = tk.Label(
            parent,
            text=APP_SUBTITLE,
            font=(self.styles.colors.get('font_family', 'Segoe UI'), FONT_SIZES['subtitle']),
            bg=self.colors['surface'],
            fg=self.colors['secondary']
        )
        self.subtitle_label.pack(anchor='w', pady=(5, 0))
    
    def _create_status_section(self, parent):
        """Create status indicator"""
        status_frame = tk.Frame(parent, bg=self.colors['surface'])
        status_frame.pack(anchor='w', pady=(10, 0))
        
        # Status dot (green = connected)
        self.status_dot = tk.Label(
            status_frame, 
            text="●", 
            font=(self.styles.colors.get('font_family', 'Segoe UI'), 12),
            fg=self.colors['success'],
            bg=self.colors['surface']
        )
        self.status_dot.pack(side=tk.LEFT)
        
        # Status text
        self.status_text = tk.Label(
            status_frame, 
            text=f"Conectat | {self.dates_count} zile cu date",
            font=(self.styles.colors.get('font_family', 'Segoe UI'), 11),
            bg=self.colors['surface'],
            fg=self.colors['muted']
        )
        self.status_text.pack(side=tk.LEFT, padx=(5, 0))
    
    def update_status(self, dates_count, connected=True):
        """Update status information"""
        self.dates_count = dates_count
        
        # Update status dot color
        dot_color = self.colors['success'] if connected else self.colors['danger']
        self.status_dot.configure(fg=dot_color)
        
        # Update status text
        status_text = "Conectat" if connected else "Deconectat"
        self.status_text.configure(text=f"{status_text} | {dates_count} zile cu date")
    
    def set_title(self, new_title):
        """Update the main title"""
        self.title_label.configure(text=new_title)
    
    def set_subtitle(self, new_subtitle):
        """Update the subtitle"""
        self.subtitle_label.configure(text=new_subtitle)