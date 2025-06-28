"""
Controls panel component for Traffic Analyzer App
"""
import tkinter as tk
from tkinter import ttk, messagebox
from utils.constants import VISUALIZATION_OPTIONS, CARD_HEADER_HEIGHT, FONT_SIZES

class ControlsPanelComponent:
    """Controls panel with visualization options and action buttons"""
    
    def __init__(self, parent, styles, on_visualization_change=None, on_test_data_add=None):
        self.parent = parent
        self.styles = styles
        self.colors = styles.colors
        self.on_visualization_change = on_visualization_change
        self.on_test_data_add = on_test_data_add
        
        self.create_controls_panel()
    
    def create_controls_panel(self):
        """Create controls card with options and buttons"""
        # Main card frame
        self.card_frame = tk.Frame(self.parent, bg=self.colors['surface'], relief='solid', bd=1)
        self.card_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Card header
        self._create_card_header()
        
        # Controls content
        self._create_controls_content()
        
        return self.card_frame
    
    def _create_card_header(self):
        """Create compact card header"""
        header_frame = tk.Frame(self.card_frame, bg=self.colors['info'], height=28)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame, 
            text="⚙️ Opțiuni",
            font=('Segoe UI', FONT_SIZES['normal'], 'bold'),
            bg=self.colors['info'],
            fg='white'
        )
        header_label.pack(expand=True)
    
    def _create_controls_content(self):
        """Create compact controls content area"""
        controls_content = tk.Frame(self.card_frame, bg=self.colors['surface'])
        controls_content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # View type selector
        self._create_view_selector(controls_content)
        
        # Action buttons
        self._create_action_buttons(controls_content)
    
    def _create_view_selector(self, parent):
        """Create compact visualization type selector"""
        # Label
        view_label = tk.Label(
            parent, 
            text="Tip vizualizare:",
            font=('Segoe UI', FONT_SIZES['small'], 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['dark']
        )
        view_label.pack(anchor='w', pady=(0, 2))
        
        # Combobox
        self.view_type = ttk.Combobox(
            parent, 
            values=VISUALIZATION_OPTIONS,
            state="readonly",
            font=('Segoe UI', 8),
            style='Modern.TCombobox'
        )
        self.view_type.set(VISUALIZATION_OPTIONS[0])  # Default to first option
        self.view_type.pack(fill=tk.X, pady=(0, 8))
        
        # Bind change event
        self.view_type.bind("<<ComboboxSelected>>", self._on_view_type_change)
    
    def _create_action_buttons(self, parent):
        """Create compact action buttons with modern styling"""
        button_frame = tk.Frame(parent, bg=self.colors['surface'])
        button_frame.pack(fill=tk.X)
        
        # Update/Refresh button
        self.generate_btn = tk.Button(
            button_frame, 
            text="🔄 Actualizează",
            font=('Segoe UI', 8, 'bold'),
            bg=self.colors['primary'],
            fg='white',
            relief='flat',
            padx=8, 
            pady=4,
            cursor='hand2',
            command=self._on_generate_click
        )
        self.generate_btn.pack(fill=tk.X, pady=(0, 4))
        
        # Test data button
        self.test_data_btn = tk.Button(
            button_frame, 
            text="🔧 Date Test",
            font=('Segoe UI', 8, 'bold'),
            bg=self.colors['success'],
            fg='white',
            relief='flat',
            padx=8, 
            pady=4,
            cursor='hand2',
            command=self._on_test_data_click
        )
        self.test_data_btn.pack(fill=tk.X)
        
        # Add hover effects
        self._setup_button_hover_effects()
    
    def _setup_button_hover_effects(self):
        """Setup hover effects for buttons"""
        on_enter, on_leave = self.styles.get_button_hover_effects()
        
        # Generate button hover
        self.generate_btn.bind("<Enter>", 
                               lambda e: on_enter(e, self.generate_btn, '#1d4ed8'))
        self.generate_btn.bind("<Leave>", 
                               lambda e: on_leave(e, self.generate_btn, self.colors['primary']))
        
        # Test data button hover
        self.test_data_btn.bind("<Enter>", 
                                lambda e: on_enter(e, self.test_data_btn, '#047857'))
        self.test_data_btn.bind("<Leave>", 
                                lambda e: on_leave(e, self.test_data_btn, self.colors['success']))
    
    def _on_view_type_change(self, event):
        """Handle visualization type change"""
        if self.on_visualization_change:
            self.on_visualization_change(self.get_selected_view_type())
    
    def _on_generate_click(self):
        """Handle generate/update button click"""
        if self.on_visualization_change:
            self.on_visualization_change(self.get_selected_view_type())
    
    def _on_test_data_click(self):
        """Handle test data button click"""
        if self.on_test_data_add:
            success = self.on_test_data_add()
            
            if success:
                messagebox.showinfo("✅ Succes", "Date de test adăugate cu succes!")
            else:
                messagebox.showerror("❌ Eroare", "Eroare la adăugarea datelor de test!")
    
    def get_selected_view_type(self):
        """Get currently selected visualization type"""
        return self.view_type.get()
    
    def set_selected_view_type(self, view_type):
        """Set visualization type programmatically"""
        if view_type in VISUALIZATION_OPTIONS:
            self.view_type.set(view_type)
    
    def set_visualization_callback(self, callback):
        """Set callback for visualization changes"""
        self.on_visualization_change = callback
    
    def set_test_data_callback(self, callback):
        """Set callback for test data addition"""
        self.on_test_data_add = callback