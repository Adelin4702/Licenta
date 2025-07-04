"""
Statistics panel component for Traffic Analyzer App
"""
import tkinter as tk
from tkinter import ttk
from utils.constants import STATS_PANEL_WIDTH, CARD_HEADER_HEIGHT, FONT_SIZES

class StatsPanelComponent:
    """Scrollable statistics panel for displaying detailed text-based stats"""
    
    def __init__(self, parent, styles):
        self.parent = parent
        self.styles = styles
        self.colors = styles.colors
        
        self.create_stats_panel()
    
    def create_stats_panel(self):
        """Create statistics panel with enhanced scrollable content"""
        # Stats container with LARGER fixed width for more space
        self.stats_container = tk.Frame(self.parent, bg=self.colors['surface'], width=250)  # INCREASED from 250 to 320
        self.stats_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        # self.stats_container.pack_propagate(False)
        
        # Stats header
        self._create_stats_header()
        
        # Scrollable content area
        self._create_scrollable_content()
        
        # Show default message
        self.show_default_message()
        
        # CRITICAL: Force canvas to update size after a short delay
        self.stats_container.after(100, self._force_canvas_update)
        
        return self.stats_container
    
    def _create_stats_header(self):
        """Create statistics panel header"""
        stats_header = tk.Frame(self.stats_container, bg=self.colors['primary'], height=CARD_HEADER_HEIGHT)
        stats_header.pack(fill=tk.X)
        # stats_header.pack_propagate(False)
        
        tk.Label(
            stats_header, 
            text="📊 Statistici",
            font=('Segoe UI', FONT_SIZES['card_header'], 'bold'),
            bg=self.colors['primary'],
            fg='white'
        ).pack(expand=True)
    
    def _create_scrollable_content(self):
        """Create enhanced scrollable content area with better scrolling"""
        # Create main scroll frame
        scroll_frame = tk.Frame(self.stats_container, bg=self.colors['surface'])
        scroll_frame.pack(fill="both", expand=True, padx=3, pady=3)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(scroll_frame, bg=self.colors['surface'], highlightthickness=0)
        
        # Scrollbars
        self.v_scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(scroll_frame, orient="horizontal", command=self.canvas.xview)  # HORIZONTAL SCROLL
        
        # Configure canvas scrolling
        self.canvas.configure(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set  # HORIZONTAL SCROLL CONFIG
        )
        
        # Grid layout for proper scrollbar positioning
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")  # HORIZONTAL SCROLLBAR
        
        # Configure grid weights
        scroll_frame.grid_rowconfigure(0, weight=1)
        scroll_frame.grid_columnconfigure(0, weight=1)
        
        # Stats content frame with LARGER initial width
        self.text_stats_frame = tk.Frame(self.canvas, bg=self.colors['surface'])
        self.canvas_window = self.canvas.create_window((0, 0), window=self.text_stats_frame, anchor="nw")
        
        # Configure scrolling
        self._setup_enhanced_scroll_configuration()
        
        # Enhanced mouse wheel scrolling
        self._setup_enhanced_mouse_wheel_scrolling()
    
    def _setup_enhanced_scroll_configuration(self):
        """Setup enhanced scroll region configuration"""
        def on_frame_configure(event=None):
            min_width = 250  # Lățime minimă dorită    
            self.canvas.itemconfig(self.canvas_window, width=min_width)
        
        # Legătură la orice modificare de conținut
        self.text_stats_frame.bind("<Configure>", on_frame_configure)
        # Legătură la orice redimensionare a canvasului
        self.canvas.bind("<Configure>", on_frame_configure)
    
    def _setup_enhanced_mouse_wheel_scrolling(self):
        """Setup enhanced mouse wheel scrolling for Windows"""
        def _on_mousewheel(event):
            # Simple vertical scrolling - always try to scroll
            try:
                bbox = self.canvas.bbox("all")
                if bbox:
                    canvas_height = self.canvas.winfo_height()
                    content_height = bbox[3] - bbox[1]
                    if content_height > canvas_height:
                        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                        print(f"Scrolling: delta={event.delta}")
            except Exception as e:
                print(f"Scroll error: {e}")
        
        def _on_shift_mousewheel(event):
            # Simple horizontal scrolling - always try to scroll
            try:
                self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
            except:
                pass
        
        # Bind to ALL widgets in the stats panel
        def bind_to_widget_and_children(widget):
            try:
                widget.bind("<MouseWheel>", _on_mousewheel)
                widget.bind("<Shift-MouseWheel>", _on_shift_mousewheel)
                # Also bind clicking to set focus
                widget.bind("<Button-1>", lambda e: self.canvas.focus_set())
                
                # Recursively bind to all children
                for child in widget.winfo_children():
                    bind_to_widget_and_children(child)
            except:
                pass
        
        # Bind to the entire stats container
        bind_to_widget_and_children(self.stats_container)
    
    def show_default_message(self):
        """Show default message when no stats are available"""
        self.clear_stats()
        
        default_label = tk.Label(
            self.text_stats_frame,
            text="📈 Statisticile detaliate vor apărea aici după selectarea unei vizualizări.",
            font=('Segoe UI', FONT_SIZES['normal']),
            bg=self.colors['surface'],
            fg=self.colors['muted'],
            wraplength=230,  # INCREASED wrap length for wider panel
            justify=tk.CENTER
        )
        default_label.pack(expand=True, pady=20)
    
    def clear_stats(self):
        """Clear all statistics content"""
        for widget in self.text_stats_frame.winfo_children():
            widget.destroy()
        
        # CRITICAL: Reset scroll after clearing
        self.canvas.yview_moveto(0)
        self.canvas.xview_moveto(0)
        self.text_stats_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def display_stats(self, stats_text):
        """Display statistics text in the panel with better formatting"""
        self.clear_stats()
        
        stats_label = tk.Label(
            self.text_stats_frame,
            text=stats_text,
            font=('Segoe UI', FONT_SIZES['small']),
            bg=self.colors['surface'],
            fg=self.colors['dark'],
            justify=tk.LEFT,
            anchor='nw',
            wraplength=300  # INCREASED wrap length for better text flow
        )
        stats_label.pack(fill=tk.BOTH, padx=5, pady=5)
        
        # Update scroll region after adding content
        self.text_stats_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def add_stats_section(self, title, content, title_color=None):
        """Add a formatted statistics section with better spacing"""
        if not title_color:
            title_color = self.colors['primary']
        
        # Title
        title_label = tk.Label(
            self.text_stats_frame,
            text=title,
            font=('Segoe UI', FONT_SIZES['normal'], 'bold'),
            bg=self.colors['surface'],
            fg=title_color,
            wraplength=290,  # INCREASED wrap length
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
            wraplength=280  # INCREASED wrap length
        )
        content_label.pack(anchor='w', padx=15, pady=(0, 5))
        
        # CRITICAL: Force scroll update after EVERY addition
        self.text_stats_frame.update_idletasks()
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)
            # Debug info
            canvas_height = self.canvas.winfo_height()
            content_height = bbox[3] - bbox[1]
            print(f"Canvas height: {canvas_height}, Content height: {content_height}, Can scroll: {content_height > canvas_height}")
    
    def add_metric_card(self, title, value, unit="", icon="📊", color=None):
        """Add a styled metric card with better sizing"""
        if not color:
            color = self.colors['primary']
        
        # Card frame with better width
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
        
        # CRITICAL: Force scroll update after divider
        self.text_stats_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def get_stats_frame(self):
        """Get the stats frame for custom content"""
        return self.text_stats_frame
    
    def scroll_to_top(self):
        """Scroll to the top of the stats panel"""
        self.canvas.yview_moveto(0)
        self.canvas.xview_moveto(0)  # Also reset horizontal scroll
        # Set focus to canvas so scroll wheel works
        self.canvas.focus_set()
    
    def _force_canvas_update(self):
        """Force canvas to update its dimensions and scroll region"""
        self.canvas.update_idletasks()
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)
            canvas_height = self.canvas.winfo_height()
            content_height = bbox[3] - bbox[1]
            print(f"Force update - Canvas: {canvas_height}px, Content: {content_height}px")
            
            # If content is larger than canvas, scrollbars should be visible
            if content_height > canvas_height:
                print("Content overflows - scroll should work!")
            else:
                print("Content fits - no scroll needed")