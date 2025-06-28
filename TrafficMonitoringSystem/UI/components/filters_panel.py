"""
Filters panel component for Traffic Analyzer App - Database Filters
"""
import tkinter as tk
from tkinter import ttk
from utils.constants import CARD_HEADER_HEIGHT, FONT_SIZES

class FiltersPanelComponent:
    """Database filters panel for cities, locations, and cameras"""
    
    def __init__(self, parent, styles, cities_data, locations_data, cameras_data, on_filter_change_callback=None):
        self.parent = parent
        self.styles = styles
        self.colors = styles.colors
        self.cities_data = cities_data or {}
        self.locations_data = locations_data or {}
        self.cameras_data = cameras_data or {}
        self.on_filter_change_callback = on_filter_change_callback
        
        self.create_filters_panel()
    
    def create_filters_panel(self):
        """Create database filters card"""
        # Main card frame
        self.card_frame = tk.Frame(self.parent, bg=self.colors['surface'], relief='solid', bd=1)
        self.card_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Card header
        self._create_card_header()
        
        # Filters content
        self._create_filters_content()
        
        return self.card_frame
    
    def _create_card_header(self):
        """Create compact card header"""
        header_frame = tk.Frame(self.card_frame, bg=self.colors['accent'], height=28)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame, 
            text="🔍 Filtre",
            font=('Segoe UI', FONT_SIZES['normal'], 'bold'),
            bg=self.colors['accent'],
            fg='white'
        )
        header_label.pack(expand=True)
    
    def _create_filters_content(self):
        """Create filters content area"""
        filters_content = tk.Frame(self.card_frame, bg=self.colors['surface'])
        filters_content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # City filter
        self._create_city_filter(filters_content)
        
        # Location filter
        self._create_location_filter(filters_content)
        
        # Camera filter
        self._create_camera_filter(filters_content)
        
        # Reset button
        self._create_reset_button(filters_content)
    
    def _create_city_filter(self, parent):
        """Create city filter"""
        city_label = tk.Label(parent, text="Oraș:",
                             font=('Segoe UI', FONT_SIZES['small'], 'bold'),
                             bg=self.colors['surface'],
                             fg=self.colors['dark'])
        city_label.pack(anchor='w', pady=(0, 2))
        
        city_values = ["Toate orașele"] + list(self.cities_data.keys())
        self.city_filter = ttk.Combobox(parent, 
                                       values=city_values,
                                       state="readonly",
                                       font=('Segoe UI', 8),
                                       style='Modern.TCombobox')
        self.city_filter.set("Toate orașele")
        self.city_filter.pack(fill=tk.X, pady=(0, 6))
        self.city_filter.bind("<<ComboboxSelected>>", self.on_city_change)
    
    def _create_location_filter(self, parent):
        """Create location filter"""
        location_label = tk.Label(parent, text="Locație:",
                                 font=('Segoe UI', FONT_SIZES['small'], 'bold'),
                                 bg=self.colors['surface'],
                                 fg=self.colors['dark'])
        location_label.pack(anchor='w', pady=(0, 2))
        
        self.location_filter = ttk.Combobox(parent, 
                                           values=["Toate locațiile"],
                                           state="readonly",
                                           font=('Segoe UI', 8),
                                           style='Modern.TCombobox')
        self.location_filter.set("Toate locațiile")
        self.location_filter.pack(fill=tk.X, pady=(0, 6))
        self.location_filter.bind("<<ComboboxSelected>>", self.on_location_change)
    
    def _create_camera_filter(self, parent):
        """Create camera filter"""
        camera_label = tk.Label(parent, text="Cameră:",
                               font=('Segoe UI', FONT_SIZES['small'], 'bold'),
                               bg=self.colors['surface'],
                               fg=self.colors['dark'])
        camera_label.pack(anchor='w', pady=(0, 2))
        
        self.camera_filter = ttk.Combobox(parent, 
                                         values=["Toate camerele"],
                                         state="readonly",
                                         font=('Segoe UI', 8),
                                         style='Modern.TCombobox')
        self.camera_filter.set("Toate camerele")
        self.camera_filter.pack(fill=tk.X, pady=(0, 6))
        self.camera_filter.bind("<<ComboboxSelected>>", self.on_filter_change)
    
    def _create_reset_button(self, parent):
        """Create reset filters button"""
        self.reset_btn = tk.Button(parent, text="🔄 Reset",
                                  font=('Segoe UI', 8, 'bold'),
                                  bg=self.colors['secondary'],
                                  fg='white',
                                  relief='flat',
                                  padx=8, pady=4,
                                  cursor='hand2',
                                  command=self.reset_filters)
        self.reset_btn.pack(fill=tk.X)
        
        # Hover effects
        self._setup_button_hover_effects()
    
    def _setup_button_hover_effects(self):
        """Setup hover effects for reset button"""
        def on_enter(e):
            self.reset_btn.configure(bg='#475569')
        def on_leave(e):
            self.reset_btn.configure(bg=self.colors['secondary'])
            
        self.reset_btn.bind("<Enter>", on_enter)
        self.reset_btn.bind("<Leave>", on_leave)
    
    def on_city_change(self, event=None):
        """Handle city filter change"""
        selected_city = self.city_filter.get()
        
        if selected_city == "Toate orașele":
            # Reset location and camera filters
            self.location_filter['values'] = ["Toate locațiile"]
            self.location_filter.set("Toate locațiile")
            self.camera_filter['values'] = ["Toate camerele"]
            self.camera_filter.set("Toate camerele")
        else:
            # Update location filter based on selected city
            if selected_city in self.locations_data:
                location_values = ["Toate locațiile"] + list(self.locations_data[selected_city].keys())
                self.location_filter['values'] = location_values
                self.location_filter.set("Toate locațiile")
            else:
                self.location_filter['values'] = ["Toate locațiile"]
                self.location_filter.set("Toate locațiile")
            
            # Reset camera filter
            self.camera_filter['values'] = ["Toate camerele"]
            self.camera_filter.set("Toate camerele")
        
        self.on_filter_change()
    
    def on_location_change(self, event=None):
        """Handle location filter change"""
        selected_city = self.city_filter.get()
        selected_location = self.location_filter.get()
        
        if selected_city == "Toate orașele" or selected_location == "Toate locațiile":
            # Reset camera filter
            self.camera_filter['values'] = ["Toate camerele"]
            self.camera_filter.set("Toate camerele")
        else:
            # Update camera filter based on selected city and location
            if (selected_city in self.cameras_data and 
                selected_location in self.cameras_data[selected_city]):
                camera_values = ["Toate camerele"] + list(self.cameras_data[selected_city][selected_location].keys())
                self.camera_filter['values'] = camera_values
                self.camera_filter.set("Toate camerele")
            else:
                self.camera_filter['values'] = ["Toate camerele"]
                self.camera_filter.set("Toate camerele")
        
        self.on_filter_change()
    
    def on_filter_change(self, event=None):
        """Handle any filter change and notify callback"""
        if self.on_filter_change_callback:
            self.on_filter_change_callback()
    
    def reset_filters(self):
        """Reset all filters to default values"""
        self.city_filter.set("Toate orașele")
        self.location_filter['values'] = ["Toate locațiile"]
        self.location_filter.set("Toate locațiile")
        self.camera_filter['values'] = ["Toate camerele"]
        self.camera_filter.set("Toate camerele")
        self.on_filter_change()
    
    def get_selected_filters(self):
        """Get currently selected filter values"""
        return {
            'city': self.city_filter.get(),
            'location': self.location_filter.get(),
            'camera': self.camera_filter.get()
        }
    
    def update_filter_data(self, cities_data, locations_data, cameras_data):
        """Update filter data and refresh dropdowns"""
        self.cities_data = cities_data or {}
        self.locations_data = locations_data or {}
        self.cameras_data = cameras_data or {}
        
        # Refresh city values
        city_values = ["Toate orașele"] + list(self.cities_data.keys())
        self.city_filter['values'] = city_values
        
        # Reset all filters
        self.reset_filters()