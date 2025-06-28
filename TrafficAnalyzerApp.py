import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
from tkcalendar import Calendar
import numpy as np
from db_functions import TrafficDatabase

class BinaryTrafficAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚗 Analizator Trafic Inteligent")
        self.root.state('zoomed')
        
        # Modern color scheme
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
        
        # Configure matplotlib style
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
        
        # Database connection
        self.db = TrafficDatabase('traffic_binary.db')
        self.dates_with_data = self.db.get_dates_with_data()
        
        # Initialize filter data
        self.cities_data = {}
        self.locations_data = {}
        self.cameras_data = {}
        self.load_filter_data()
        
        # Setup modern styling
        self.setup_styles()
        
        # Create GUI
        self.create_modern_gui()
        
    def load_filter_data(self):
        """Load cities, locations and cameras data for filters"""
        try:
            # Get all cities
            self.db.cursor.execute("SELECT id, city_name FROM city ORDER BY city_name")
            cities = self.db.cursor.fetchall()
            self.cities_data = {city[1]: city[0] for city in cities}
            
            # Get all locations grouped by city
            self.db.cursor.execute("""
                SELECT l.id, l.road_name, l.road_km, l.city_id, c.city_name 
                FROM location l 
                JOIN city c ON l.city_id = c.id 
                ORDER BY c.city_name, l.road_name
            """)
            locations = self.db.cursor.fetchall()
            
            self.locations_data = {}
            for loc in locations:
                loc_id, road_name, road_km, city_id, city_name = loc
                location_display = f"{road_name} (km {road_km})" if road_km else road_name
                
                if city_name not in self.locations_data:
                    self.locations_data[city_name] = {}
                self.locations_data[city_name][location_display] = loc_id
            
            # Get all cameras grouped by location and city
            self.db.cursor.execute("""
                SELECT cam.id, cam.model, l.road_name, l.road_km, c.city_name 
                FROM camera cam
                JOIN location l ON cam.location_id = l.id
                JOIN city c ON l.city_id = c.id 
                ORDER BY c.city_name, l.road_name, cam.model
            """)
            cameras = self.db.cursor.fetchall()
            
            self.cameras_data = {}
            for cam in cameras:
                cam_id, model, road_name, road_km, city_name = cam
                location_display = f"{road_name} (km {road_km})" if road_km else road_name
                camera_display = f"{model}"
                
                if city_name not in self.cameras_data:
                    self.cameras_data[city_name] = {}
                if location_display not in self.cameras_data[city_name]:
                    self.cameras_data[city_name][location_display] = {}
                self.cameras_data[city_name][location_display][camera_display] = cam_id
                
        except Exception as e:
            print(f"Error loading filter data: {e}")
            self.cities_data = {}
            self.locations_data = {}
            self.cameras_data = {}
        
    def setup_styles(self):
        """Configure modern ttk styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure root background
        self.root.configure(bg=self.colors['background'])
        
        # Combobox style
        self.style.configure('Modern.TCombobox',
                           font=('Segoe UI', 10),
                           padding=8,
                           fieldbackground=self.colors['light'],
                           borderwidth=1,
                           relief='solid')
        
    def create_modern_gui(self):
        """Create modern, professional GUI optimized for 1200x800"""
        # Main container with reduced padding
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header section
        self.create_header(main_frame)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel (controls) - optimized width
        left_panel = tk.Frame(content_frame, bg=self.colors['background'], width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel (visualization)
        right_panel = tk.Frame(content_frame, bg=self.colors['background'])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create panels
        self.create_control_panel(left_panel)
        self.create_visualization_panel(right_panel)
        
        # Initialize
        self.selected_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    def create_header(self, parent):
        """Create compact header"""
        header_frame = tk.Frame(parent, bg=self.colors['surface'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Add subtle shadow effect
        shadow_frame = tk.Frame(parent, bg='#e2e8f0', height=1)
        shadow_frame.pack(fill=tk.X)
        
        # Header content
        header_content = tk.Frame(header_frame, bg=self.colors['surface'])
        header_content.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Title and status in same line
        title_status_frame = tk.Frame(header_content, bg=self.colors['surface'])
        title_status_frame.pack(fill=tk.X)
        
        # Title on left
        title_label = tk.Label(title_status_frame, 
                              text="🚗 Analizator Trafic Inteligent",
                              font=('Segoe UI', 18, 'bold'),
                              bg=self.colors['surface'],
                              fg=self.colors['primary'])
        title_label.pack(side=tk.LEFT)
        
        # Subtitle
        subtitle_label = tk.Label(header_content,
                                 text="Analiză avansată pentru vehicule mari și mici",
                                 font=('Segoe UI', 11),
                                 bg=self.colors['surface'],
                                 fg=self.colors['secondary'])
        subtitle_label.pack(anchor='w', pady=(2, 0))
        
    def create_control_panel(self, parent):
        """Create modern control panel"""
        # Calendar card
        self.create_calendar_card(parent)
        
        # Filters card
        self.create_filters_card(parent)
        
        # Controls card
        self.create_controls_card(parent)
        
    def create_filters_card(self, parent):
        """Create compact filters card for city, location, camera selection"""
        card_frame = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        card_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Card header
        header_frame = tk.Frame(card_frame, bg=self.colors['accent'], height=28)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="🔍 Filtre",
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.colors['accent'],
                               fg='white')
        header_label.pack(expand=True)
        
        # Filters content
        filters_content = tk.Frame(card_frame, bg=self.colors['surface'])
        filters_content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # City filter
        city_label = tk.Label(filters_content, text="Oraș:",
                             font=('Segoe UI', 9, 'bold'),
                             bg=self.colors['surface'],
                             fg=self.colors['dark'])
        city_label.pack(anchor='w', pady=(0, 2))
        
        city_values = ["Toate orașele"] + list(self.cities_data.keys())
        self.city_filter = ttk.Combobox(filters_content, 
                                       values=city_values,
                                       state="readonly",
                                       font=('Segoe UI', 8),
                                       style='Modern.TCombobox')
        self.city_filter.set("Toate orașele")
        self.city_filter.pack(fill=tk.X, pady=(0, 6))
        self.city_filter.bind("<<ComboboxSelected>>", self.on_city_change)
        
        # Location filter
        location_label = tk.Label(filters_content, text="Locație:",
                                 font=('Segoe UI', 9, 'bold'),
                                 bg=self.colors['surface'],
                                 fg=self.colors['dark'])
        location_label.pack(anchor='w', pady=(0, 2))
        
        self.location_filter = ttk.Combobox(filters_content, 
                                           values=["Toate locațiile"],
                                           state="readonly",
                                           font=('Segoe UI', 8),
                                           style='Modern.TCombobox')
        self.location_filter.set("Toate locațiile")
        self.location_filter.pack(fill=tk.X, pady=(0, 6))
        self.location_filter.bind("<<ComboboxSelected>>", self.on_location_change)
        
        # Camera filter
        camera_label = tk.Label(filters_content, text="Cameră:",
                               font=('Segoe UI', 9, 'bold'),
                               bg=self.colors['surface'],
                               fg=self.colors['dark'])
        camera_label.pack(anchor='w', pady=(0, 2))
        
        self.camera_filter = ttk.Combobox(filters_content, 
                                         values=["Toate camerele"],
                                         state="readonly",
                                         font=('Segoe UI', 8),
                                         style='Modern.TCombobox')
        self.camera_filter.set("Toate camerele")
        self.camera_filter.pack(fill=tk.X, pady=(0, 6))
        self.camera_filter.bind("<<ComboboxSelected>>", self.on_filter_change)
        
        # Reset filters button
        reset_btn = tk.Button(filters_content, text="🔄 Reset",
                             font=('Segoe UI', 8, 'bold'),
                             bg=self.colors['secondary'],
                             fg='white',
                             relief='flat',
                             padx=8, pady=4,
                             cursor='hand2',
                             command=self.reset_filters)
        reset_btn.pack(fill=tk.X)
        
        # Hover effects for reset button
        def on_enter(e):
            reset_btn.configure(bg='#475569')
        def on_leave(e):
            reset_btn.configure(bg=self.colors['secondary'])
            
        reset_btn.bind("<Enter>", on_enter)
        reset_btn.bind("<Leave>", on_leave)
        
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
        """Handle any filter change and refresh visualization"""
        self.generate_visualization()
    
    def reset_filters(self):
        """Reset all filters to default values"""
        self.city_filter.set("Toate orașele")
        self.location_filter['values'] = ["Toate locațiile"]
        self.location_filter.set("Toate locațiile")
        self.camera_filter['values'] = ["Toate camerele"]
        self.camera_filter.set("Toate camerele")
        self.generate_visualization()
    
    def get_filter_conditions(self):
        """Get SQL conditions based on current filter selections"""
        conditions = []
        params = []
        
        selected_city = self.city_filter.get()
        selected_location = self.location_filter.get()
        selected_camera = self.camera_filter.get()
        
        if selected_city != "Toate orașele":
            if selected_city in self.cities_data:
                city_id = self.cities_data[selected_city]
                conditions.append("city.id = ?")
                params.append(city_id)
        
        if selected_location != "Toate locațiile" and selected_city != "Toate orașele":
            if (selected_city in self.locations_data and 
                selected_location in self.locations_data[selected_city]):
                location_id = self.locations_data[selected_city][selected_location]
                conditions.append("l.id = ?")
                params.append(location_id)
        
        if (selected_camera != "Toate camerele" and 
            selected_city != "Toate orașele" and 
            selected_location != "Toate locațiile"):
            if (selected_city in self.cameras_data and 
                selected_location in self.cameras_data[selected_city] and
                selected_camera in self.cameras_data[selected_city][selected_location]):
                camera_id = self.cameras_data[selected_city][selected_location][selected_camera]
                conditions.append("c.id = ?")
                params.append(camera_id)
        
        return conditions, params
    
    def get_filtered_hourly_data(self, date):
        """Get hourly data with filters applied"""
        try:
            conditions, params = self.get_filter_conditions()
            
            base_query = '''
                SELECT strftime('%H', r.timestamp) as hour,
                       SUM(r.nr_of_large_vehicles) as large_vehicles,
                       SUM(r.nr_of_small_vehicles) as small_vehicles
                FROM record r
                JOIN camera c ON r.camera_id = c.id
                JOIN location l ON c.location_id = l.id
                JOIN city ON l.city_id = city.id
                WHERE date(r.timestamp) = ?
            '''
            
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            base_query += '''
                GROUP BY strftime('%H', r.timestamp)
                ORDER BY hour
            '''
            
            params.insert(0, date)  # Add date as first parameter
            
            self.db.cursor.execute(base_query, params)
            return self.db.cursor.fetchall()
        except Exception as e:
            print(f"Error getting filtered hourly data: {e}")
            return []
    
    def get_filtered_daily_totals(self, date):
        """Get daily totals with filters applied"""
        try:
            conditions, params = self.get_filter_conditions()
            
            base_query = '''
                SELECT SUM(r.nr_of_large_vehicles) as total_large,
                       SUM(r.nr_of_small_vehicles) as total_small
                FROM record r
                JOIN camera c ON r.camera_id = c.id
                JOIN location l ON c.location_id = l.id
                JOIN city ON l.city_id = city.id
                WHERE date(r.timestamp) = ?
            '''
            
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            params.insert(0, date)  # Add date as first parameter
            
            self.db.cursor.execute(base_query, params)
            result = self.db.cursor.fetchone()
            return (result[0] or 0, result[1] or 0)
        except Exception as e:
            print(f"Error getting filtered daily totals: {e}")
            return (0, 0)
    
    def get_filtered_week_data_by_range(self, start_date, end_date):
        """Get week data with filters applied"""
        try:
            conditions, params = self.get_filter_conditions()
            
            base_query = '''
                SELECT date(r.timestamp) as day,
                       SUM(r.nr_of_large_vehicles) as large_vehicles,
                       SUM(r.nr_of_small_vehicles) as small_vehicles
                FROM record r
                JOIN camera c ON r.camera_id = c.id
                JOIN location l ON c.location_id = l.id
                JOIN city ON l.city_id = city.id
                WHERE date(r.timestamp) BETWEEN ? AND ?
            '''
            
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            base_query += '''
                GROUP BY date(r.timestamp)
                ORDER BY day
            '''
            
            params.insert(0, start_date)  # Add start_date as first parameter
            params.insert(1, end_date)    # Add end_date as second parameter
            
            self.db.cursor.execute(base_query, params)
            return self.db.cursor.fetchall()
        except Exception as e:
            print(f"Error getting filtered week data: {e}")
            return []
    
    def get_filtered_monthly_trend(self, month):
        """Get monthly trend with filters applied"""
        try:
            conditions, params = self.get_filter_conditions()
            
            base_query = '''
                SELECT strftime('%H', r.timestamp) as hour,
                       AVG(r.nr_of_large_vehicles) as avg_large,
                       AVG(r.nr_of_small_vehicles) as avg_small
                FROM record r
                JOIN camera c ON r.camera_id = c.id
                JOIN location l ON c.location_id = l.id
                JOIN city ON l.city_id = city.id
                WHERE strftime('%Y-%m', r.timestamp) = ?
            '''
            
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            base_query += '''
                GROUP BY strftime('%H', r.timestamp)
                ORDER BY hour
            '''
            
            params.insert(0, month)  # Add month as first parameter
            
            self.db.cursor.execute(base_query, params)
            return self.db.cursor.fetchall()
        except Exception as e:
            print(f"Error getting filtered monthly trend: {e}")
            return []
    
    def get_filtered_peak_hours_data(self, date):
        """Get peak hours data with filters applied"""
        try:
            conditions, params = self.get_filter_conditions()
            
            base_query = '''
                SELECT strftime('%H', r.timestamp) as hour,
                       SUM(r.nr_of_large_vehicles) as large_vehicles,
                       SUM(r.nr_of_small_vehicles) as small_vehicles
                FROM record r
                JOIN camera c ON r.camera_id = c.id
                JOIN location l ON c.location_id = l.id
                JOIN city ON l.city_id = city.id
                WHERE date(r.timestamp) = ?
            '''
            
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            base_query += '''
                GROUP BY strftime('%H', r.timestamp)
                ORDER BY hour
            '''
            
            params.insert(0, date)  # Add date as first parameter
            
            self.db.cursor.execute(base_query, params)
            return self.db.cursor.fetchall()
        except Exception as e:
            print(f"Error getting filtered peak hours data: {e}")
            return []
    
    def create_calendar_card(self, parent):
        """Create compact calendar card"""
        card_frame = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        card_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Compact card header
        header_frame = tk.Frame(card_frame, bg=self.colors['primary'], height=28)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="📅 Data",
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.colors['primary'],
                               fg='white')
        header_label.pack(expand=True)
        
        # Calendar content - more compact
        calendar_content = tk.Frame(card_frame, bg=self.colors['surface'])
        calendar_content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Smaller calendar widget
        self.calendar = Calendar(
            calendar_content,
            selectmode='day',
            year=datetime.datetime.now().year,
            month=datetime.datetime.now().month,
            day=datetime.datetime.now().day,
            date_pattern='yyyy-mm-dd',
            background=self.colors['primary'],
            foreground='white',
            selectbackground=self.colors['accent'],
            selectforeground='white',
            bordercolor=self.colors['primary'],
            headersbackground=self.colors['dark'],
            headersforeground='white',
            normalbackground=self.colors['light'],
            normalforeground=self.colors['dark'],
            weekendbackground=self.colors['surface'],
            weekendforeground=self.colors['muted'],
            font=('Segoe UI', 7)  # Even smaller font
        )
        self.calendar.pack(pady=2)
        
        self.update_calendar_marks()
        self.calendar.bind("<<CalendarSelected>>", self.on_date_select)
        
    def create_controls_card(self, parent):
        """Create compact controls card"""
        card_frame = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        card_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Compact card header
        header_frame = tk.Frame(card_frame, bg=self.colors['info'], height=28)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="⚙️ Opțiuni",
                               font=('Segoe UI', 10, 'bold'),
                               bg=self.colors['info'],
                               fg='white')
        header_label.pack(expand=True)
        
        # Controls content - more compact
        controls_content = tk.Frame(card_frame, bg=self.colors['surface'])
        controls_content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # View type selector with compact styling
        view_label = tk.Label(controls_content, text="Tip vizualizare:",
                             font=('Segoe UI', 9, 'bold'),
                             bg=self.colors['surface'],
                             fg=self.colors['dark'])
        view_label.pack(anchor='w', pady=(0, 2))
        
        self.view_type = ttk.Combobox(controls_content, 
                                     values=[
                                         "📊 Trafic orar",
                                         "🥧 Distribuție procentuală", 
                                         "🔥 Comparație ore de vârf",
                                         "📈 Trafic săptămânal",
                                         "📅 Tendință lunară"
                                     ],
                                     state="readonly",
                                     font=('Segoe UI', 8),
                                     style='Modern.TCombobox')
        self.view_type.set("📊 Trafic orar")
        self.view_type.pack(fill=tk.X, pady=(0, 8))
        self.view_type.bind("<<ComboboxSelected>>", self.generate_visualization)
        
        # Compact action buttons
        button_frame = tk.Frame(controls_content, bg=self.colors['surface'])
        button_frame.pack(fill=tk.X)
        
        generate_btn = tk.Button(button_frame, text="🔄 Actualizează",
                               font=('Segoe UI', 8, 'bold'),
                               bg=self.colors['primary'],
                               fg='white',
                               relief='flat',
                               padx=8, pady=4,
                               cursor='hand2',
                               command=self.generate_visualization)
        generate_btn.pack(fill=tk.X, pady=(0, 4))
        
        test_data_btn = tk.Button(button_frame, text="🔧 Date Test",
                                font=('Segoe UI', 8, 'bold'),
                                bg=self.colors['success'],
                                fg='white',
                                relief='flat',
                                padx=8, pady=4,
                                cursor='hand2',
                                command=self.add_test_data)
        test_data_btn.pack(fill=tk.X)
        
        # Hover effects
        def on_enter(e, btn, color):
            btn.configure(bg=color)
        def on_leave(e, btn, color):
            btn.configure(bg=color)
            
        generate_btn.bind("<Enter>", lambda e: on_enter(e, generate_btn, '#1d4ed8'))
        generate_btn.bind("<Leave>", lambda e: on_leave(e, generate_btn, self.colors['primary']))
        test_data_btn.bind("<Enter>", lambda e: on_enter(e, test_data_btn, '#047857'))
        test_data_btn.bind("<Leave>", lambda e: on_leave(e, test_data_btn, self.colors['success']))
        
    def create_visualization_panel(self, parent):
        """Create visualization panel optimized for 1200x800"""
        # Main visualization card
        viz_card = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        viz_card.pack(fill=tk.BOTH, expand=True)
        
        # Card header
        header_frame = tk.Frame(viz_card, bg=self.colors['dark'], height=35)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        self.viz_title = tk.Label(header_frame, text="📈 Vizualizare Trafic",
                                 font=('Segoe UI', 12, 'bold'),
                                 bg=self.colors['dark'],
                                 fg='white')
        self.viz_title.pack(expand=True)
        
        # Content container with horizontal split
        content_container = tk.Frame(viz_card, bg=self.colors['surface'])
        content_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Graph area - takes most of the width
        self.graph_frame = tk.Frame(content_container, bg=self.colors['light'])
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 0))
        
        # Initial placeholder
        placeholder_label = tk.Label(self.graph_frame,
                                    text="🎯\nSelectează o opțiune de vizualizare\npentru a începe analiza",
                                    font=('Segoe UI', 12),
                                    bg=self.colors['light'],
                                    fg=self.colors['muted'],
                                    justify=tk.CENTER)
        placeholder_label.pack(expand=True)
        
        # Text-based stats area - compact for 1200x800
        self.create_compact_stats_panel(content_container)
        
    def create_compact_stats_panel(self, parent):
        """Create compact stats panel on the right side"""
        # Stats frame with fixed width - optimized for 1200x800
        stats_container = tk.Frame(parent, bg=self.colors['surface'], width=200)
        stats_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        stats_container.pack_propagate(False)
        
        # Stats header
        stats_header = tk.Frame(stats_container, bg=self.colors['primary'], height=28)
        stats_header.pack(fill=tk.X)
        stats_header.pack_propagate(False)
        
        tk.Label(stats_header, text="📊 Statistici",
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['primary'],
                fg='white').pack(expand=True)
        
        # Create frame for scrollbars and canvas
        scroll_frame = tk.Frame(stats_container, bg=self.colors['surface'])
        scroll_frame.pack(fill="both", expand=True, padx=3, pady=3)
        
        # Create canvas and scrollbars
        canvas = tk.Canvas(scroll_frame, bg=self.colors['surface'], highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(scroll_frame, orient="horizontal", command=canvas.xview)
        
        # Create scrollable frame
        self.text_stats_frame = tk.Frame(canvas, bg=self.colors['surface'])
        
        # Configure scrollbars
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout for proper scrollbar positioning
        canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configure grid weights
        scroll_frame.grid_rowconfigure(0, weight=1)
        scroll_frame.grid_columnconfigure(0, weight=1)
        
        # Create window in canvas
        canvas_window = canvas.create_window((0, 0), window=self.text_stats_frame, anchor="nw")
        
        # Update scroll region when frame size changes
        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Update canvas window size when canvas size changes
        def on_canvas_configure(event):
            # Get the current size of the text frame
            canvas.update_idletasks()
            bbox = canvas.bbox("all")
            if bbox:
                # Set minimum width to canvas width, but allow frame to be wider
                frame_width = max(bbox[2], event.width - v_scrollbar.winfo_width())
                canvas.itemconfig(canvas_window, width=frame_width)
        
        # Bind events
        self.text_stats_frame.bind('<Configure>', on_frame_configure)
        canvas.bind('<Configure>', on_canvas_configure)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            
        def _on_shift_mousewheel(event):
            canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        
        # Bind mouse wheel to canvas and its children
        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", _on_mousewheel)
            widget.bind("<Shift-MouseWheel>", _on_shift_mousewheel)
            for child in widget.winfo_children():
                bind_mousewheel(child)
        
        bind_mousewheel(canvas)
        bind_mousewheel(self.text_stats_frame)
        
        # Default message
        default_label = tk.Label(self.text_stats_frame,
                               text="📈 Statisticile detaliate vor apărea aici după selectarea unei vizualizări.",
                               font=('Segoe UI', 9),
                               bg=self.colors['surface'],
                               fg=self.colors['muted'],
                               justify=tk.CENTER)
        default_label.pack(expand=True, pady=15)
            
    def update_calendar_marks(self):
        """Mark days with data in calendar"""
        for date_str in self.dates_with_data:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                self.calendar.calevent_create(date_obj, "Date disponibile", "highlight")
            except ValueError:
                continue
        
        self.calendar.tag_config('highlight', background='#10b981', foreground='white')
    
    def on_date_select(self, event):
        """Handle date selection"""
        self.selected_date = self.calendar.get_date()
        self.generate_visualization()
        
    def add_test_data(self):
        """Add test data to database"""
        success = self.db.generate_test_data(days=7)
        if success:
            self.dates_with_data = self.db.get_dates_with_data()
            self.update_calendar_marks()
            messagebox.showinfo("✅ Succes", "Date de test adăugate cu succes!")
        else:
            messagebox.showerror("❌ Eroare", "Eroare la adăugarea datelor de test!")
            
    def format_date(self, date_str):
        """Format date string yyyy-mm-dd to dd/mm/yyyy"""
        try:
            return datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
        except Exception:
            return date_str
        
    def get_filter_description(self):
        """Get description of current filters for display in titles"""
        parts = []
        
        selected_city = self.city_filter.get()
        selected_location = self.location_filter.get()
        selected_camera = self.camera_filter.get()
        
        if selected_city != "Toate orașele":
            parts.append(f"Oraș: {selected_city}")
            
        if selected_location != "Toate locațiile":
            parts.append(f"Locație: {selected_location}")
            
        if selected_camera != "Toate camerele":
            parts.append(f"Cameră: {selected_camera}")
        
        if parts:
            return " | " + " | ".join(parts)
        else:
            return " | Toate datele"
        
    def generate_visualization(self, event=None):
        """Generate selected visualization with modern styling"""
        # Clear existing content
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        for widget in self.text_stats_frame.winfo_children():
            widget.destroy()
            
        view_type = self.view_type.get()
        date = self.selected_date
        formatted_date = self.format_date(date)
        filter_desc = self.get_filter_description()
        self.viz_title.configure(text=f"{view_type}{filter_desc}")
        
        if "Trafic orar" in view_type:
            self.generate_modern_hourly_view(date, formatted_date)
        elif "Trafic săptămânal" in view_type:
            self.generate_modern_weekly_view(date, formatted_date)
        elif "Tendință lunară" in view_type:
            self.generate_modern_monthly_trend(date, formatted_date)
        elif "Distribuție procentuală" in view_type:
            self.generate_modern_percentage_distribution(date, formatted_date)
        elif "Comparație ore de vârf" in view_type:
            self.generate_modern_peak_hours_comparison(date, formatted_date)

    def create_modern_figure(self, width=8, height=6):
        """Create modern styled figure with optimized dimensions"""
        from matplotlib.figure import Figure
        fig = Figure(figsize=(width, height), 
                    facecolor=self.colors['surface'],
                    edgecolor='none',
                    tight_layout=True)
        
        # Add subtle border
        fig.patch.set_linewidth(0)
        
        return fig
        
    def generate_modern_hourly_view(self, date, formatted_date):
        """Generate modern hourly view"""
        data = self.get_filtered_hourly_data(date)
        
        if not data:
            self.show_no_data_message()
            return
        
        hours = [row[0] for row in data]
        vehicule_mari = [row[1] for row in data]
        vehicule_mici = [row[2] for row in data]
        
        # Create modern figure
        fig = self.create_modern_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        # Modern bar chart with gradients
        width = 0.35
        x = np.arange(len(hours))
        
        # Create bars with modern colors
        bars1 = ax.bar([i - width/2 for i in x], vehicule_mari, width,
                      label='Vehicule Mari', 
                      color=self.colors['danger'],
                      alpha=0.9,
                      edgecolor='white',
                      linewidth=1)
        
        bars2 = ax.bar([i + width/2 for i in x], vehicule_mici, width,
                      label='Vehicule Mici',
                      color=self.colors['success'], 
                      alpha=0.9,
                      edgecolor='white',
                      linewidth=1)
        
        # Add value labels with modern styling
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(vehicule_mari + vehicule_mici) * 0.02,
                           f'{int(height)}',
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold',
                           color=self.colors['dark'])
        
        # Modern styling
        ax.set_xlabel('Ora zilei', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_ylabel('Numărul de vehicule', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_title(f'Analiza traficului orar\n{formatted_date}', fontsize=16, fontweight='bold', 
                    color=self.colors['primary'], pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}:00" for h in hours], rotation=45, ha='right')
        
        # Modern legend
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor=self.colors['muted'])
        legend.get_frame().set_facecolor(self.colors['surface'])
        
        # Modern grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['muted'])
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        
        # Adjust layout
        fig.tight_layout(pad=1.0)
        
        # Display chart with optimized sizing
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=1)
        
        # Generate modern stats with text format
        self.generate_modern_hourly_stats_text(vehicule_mari, vehicule_mici, hours, formatted_date)
        
    def generate_modern_hourly_stats_text(self, vehicule_mari, vehicule_mici, hours, formatted_date):
        """Generate compact text-based statistics for hourly view"""
        # Clear existing stats
        for widget in self.text_stats_frame.winfo_children():
            widget.destroy()
        
        total_mari = sum(vehicule_mari)
        total_mici = sum(vehicule_mici)
        total = total_mari + total_mici
        
        # Create compact text-based detailed stats
        stats_text = f"""📊 STATISTICI {formatted_date}

🚦 TOTALURI:
    Total: {total:,} vehicule
    Mari: {total_mari:,} ({(total_mari/total*100) if total > 0 else 0:.1f}%)
    Mici: {total_mici:,} ({(total_mici/total*100) if total > 0 else 0:.1f}%)

⏰ ORE DE VÂRF:"""
        
        if vehicule_mici and vehicule_mari:
            peak_hour_mici_idx = vehicule_mici.index(max(vehicule_mici))
            peak_hour_mari_idx = vehicule_mari.index(max(vehicule_mari))
            peak_hour_mici = hours[peak_hour_mici_idx]
            peak_hour_mari = hours[peak_hour_mari_idx]
            
            stats_text += f"""
    🚗 Vehicule mici: {peak_hour_mici}:00 ({max(vehicule_mici)})
    🚛 Vehicule mari: {peak_hour_mari}:00 ({max(vehicule_mari)})
    📈 Medie: {total/len(hours):.1f}/oră

📋 DISTRIBUȚIA ORARĂ:"""
            
            # Show only non-zero hours to save space
            for i, hour in enumerate(hours):
                if i < len(vehicule_mari) and i < len(vehicule_mici):
                    hour_total = vehicule_mari[i] + vehicule_mici[i]
                    if hour_total > 0:
                        stats_text += f"""
    {hour} --> {vehicule_mari[i]} 🚛, {vehicule_mici[i]} 🚗 (total: {hour_total})"""
        
        # Display stats in scrollable text with smaller font
        stats_label = tk.Label(self.text_stats_frame,
                              text=stats_text,
                              font=('Segoe UI', 8),
                              bg=self.colors['surface'],
                              fg=self.colors['dark'],
                              justify=tk.LEFT,
                              anchor='nw')
        stats_label.pack(fill=tk.BOTH, padx=3, pady=3)
    
    def generate_modern_weekly_view(self, date, formatted_date):
        """Generate modern weekly view"""
        selected_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        days_since_monday = selected_date.weekday()
        week_start = selected_date - datetime.timedelta(days=days_since_monday)
        week_end = week_start + datetime.timedelta(days=6)
        
        data = self.get_filtered_week_data_by_range(week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d"))
        
        if not data:
            self.show_no_data_message(f"Nu există date pentru săptămâna {week_start.strftime('%d/%m')} - {week_end.strftime('%d/%m/%Y')}")
            return
        
        # Prepare data
        week_data_map = {row[0]: (row[1], row[2]) for row in data}
        vehicule_mari = []
        vehicule_mici = []
        day_labels = []
        
        for i in range(7):
            current_day = week_start + datetime.timedelta(days=i)
            day_str = current_day.strftime("%Y-%m-%d")
            day_label = current_day.strftime("%a\n%d/%m")
            
            if day_str in week_data_map:
                mari, mici = week_data_map[day_str]
                vehicule_mari.append(mari)
                vehicule_mici.append(mici)
            else:
                vehicule_mari.append(0)
                vehicule_mici.append(0)
            
            day_labels.append(day_label)
        
        # Create modern line chart
        fig = self.create_modern_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        # Modern line plots with gradients
        x_pos = range(len(day_labels))
        
        line1 = ax.plot(x_pos, vehicule_mari, marker='o', label='Vehicule Mari',
                       color=self.colors['danger'], linewidth=4, markersize=10,
                       markeredgecolor='white', markeredgewidth=2,
                       markerfacecolor=self.colors['danger'])
        
        line2 = ax.plot(x_pos, vehicule_mici, marker='s', label='Vehicule Mici',
                       color=self.colors['success'], linewidth=4, markersize=10,
                       markeredgecolor='white', markeredgewidth=2,
                       markerfacecolor=self.colors['success'])
        
        # Add gradient fill
        ax.fill_between(x_pos, vehicule_mari, alpha=0.2, color=self.colors['danger'])
        ax.fill_between(x_pos, vehicule_mici, alpha=0.2, color=self.colors['success'])
        
        # Modern styling
        ax.set_xlabel('Ziua săptămânii', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_ylabel('Numărul de vehicule', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_title(f'Analiza săptămânală a traficului\n{week_start.strftime("%d/%m")} - {week_end.strftime("%d/%m/%Y")}',
                    fontsize=16, fontweight='bold', color=self.colors['primary'], pad=20)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(day_labels, ha='center')
        
        # Modern legend and grid
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor=self.colors['muted'])
        legend.get_frame().set_facecolor(self.colors['surface'])
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['muted'])
        ax.set_axisbelow(True)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        
        fig.tight_layout(pad=2.0)
        
        # Display chart with optimized sizing
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Generate weekly stats with text format
        self.generate_modern_weekly_stats_text(vehicule_mari, vehicule_mici, day_labels, week_start, week_end, formatted_date)
    
    def generate_modern_weekly_stats_text(self, vehicule_mari, vehicule_mici, day_labels, week_start, week_end, formatted_date):
        """Generate compact text-based weekly statistics"""
        # Clear existing stats
        for widget in self.text_stats_frame.winfo_children():
            widget.destroy()
        
        total_mari_week = sum(vehicule_mari)
        total_mici_week = sum(vehicule_mici)
        total_week = total_mari_week + total_mici_week
        
        # Weekly analysis
        total_daily = [m + s for m, s in zip(vehicule_mari, vehicule_mici)]
        max_day_idx = total_daily.index(max(total_daily)) if total_daily else 0
        min_day_idx = total_daily.index(min(total_daily)) if total_daily else 0
        
        # Weekday vs weekend analysis
        weekdays_total = sum(total_daily[0:5])
        weekend_total = sum(total_daily[5:7])
        day_labels = [label.replace('\n', ' ') for label in day_labels]
        stats_text = f"""📅 SĂPTĂMÂNA {week_start.strftime('%d/%m')}-{week_end.strftime('%d/%m')}

📊 TOTALURI:
    Total: {total_week:,} vehicule
    Mari: {total_mari_week:,} ({(total_mari_week/total_week*100) if total_week > 0 else 0:.1f}%)
    Mici: {total_mici_week:,} ({(total_mici_week/total_week*100) if total_week > 0 else 0:.1f}%)
    Medie zilnică: {total_week/7:.0f}

🏆 EXTREME:
    Cel mai mult:  {day_labels[max_day_idx]} ({total_daily[max_day_idx]:,})
    Cel mai puțin: {day_labels[min_day_idx]} ({total_daily[min_day_idx]:,})

💼 LUCRĂTOARE vs WEEKEND:
    Luni-Vineri: {weekdays_total:,} ({weekdays_total/5:.0f}/zi)
    Sâmbătă-Duminică: {weekend_total:,} ({weekend_total/2:.0f}/zi)

📈 ZILNIC:"""
        
        for i, day_label in enumerate(day_labels):
            if i < len(total_daily) and total_daily[i] > 0:
                stats_text += f"""
    {day_label}: {total_daily[i]:,} (M:{vehicule_mari[i]}, m:{vehicule_mici[i]})"""
        
        # Display stats
        stats_label = tk.Label(self.text_stats_frame,
                              text=stats_text,
                              font=('Segoe UI', 9),
                              bg=self.colors['surface'],
                              fg=self.colors['dark'],
                              justify=tk.LEFT,
                              anchor='nw')
        stats_label.pack(fill=tk.BOTH, padx=5, pady=5)
    
    def generate_modern_monthly_trend(self, date, formatted_date):
        """Generate modern monthly trend view"""
        data = self.get_filtered_monthly_trend(date[:7])
        
        if not data:
            self.show_no_data_message("Nu există date pentru luna selectată!")
            return
        
        hours = [row[0] for row in data]
        avg_mari = [row[1] for row in data]
        avg_mici = [row[2] for row in data]
        
        # Create modern stacked area chart
        fig = self.create_modern_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        # Create gradient stacked area
        ax.stackplot(hours, avg_mari, avg_mici,
                    labels=['Vehicule Mari', 'Vehicule Mici'],
                    colors=[self.colors['danger'], self.colors['success']],
                    alpha=0.8)
        
        # Add trend lines
        ax.plot(hours, avg_mari, color=self.colors['danger'], linewidth=2, alpha=0.9)
        ax.plot(hours, avg_mici, color=self.colors['success'], linewidth=2, alpha=0.9)
        
        # Modern styling
        ax.set_xlabel('Ora zilei', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_ylabel('Medie vehicule', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_title(f'Tendința lunară a traficului\nLuna {date[5:7]}/{date[:4]}',
                    fontsize=16, fontweight='bold', color=self.colors['primary'], pad=20)
        
        # Modern legend and grid
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor=self.colors['muted'])
        legend.get_frame().set_facecolor(self.colors['surface'])
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['muted'])
        ax.set_axisbelow(True)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        
        fig.tight_layout(pad=2.0)
        
        # Display chart with optimized sizing
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        self.generate_monthly_stats_text(avg_mari, avg_mici, hours, formatted_date)
        
    def generate_monthly_stats_text(self, avg_mari, avg_mici, hours, date):
        """Generate compact text-based monthly statistics"""
                
        # Clear existing stats
        for widget in self.text_stats_frame.winfo_children():
            widget.destroy()
        
        # Calculate totals and averages properly
        total_mari = sum(avg_mari) if avg_mari else 0
        total_mici = sum(avg_mici) if avg_mici else 0
        total = total_mari + total_mici
        
        # Calculate average per hour across the month
        avg_mari_per_hour = total_mari / len(avg_mari) if avg_mari else 0
        avg_mici_per_hour = total_mici / len(avg_mici) if avg_mici else 0
        
        # Find peak hours
        peak_mari_idx = avg_mari.index(max(avg_mari)) if avg_mari and max(avg_mari) > 0 else 0
        peak_mici_idx = avg_mici.index(max(avg_mici)) if avg_mici and max(avg_mici) > 0 else 0
        
        date_to_print = date[-7:] if len(date) >= 7 else date
        
        stats_text = f"""
LUNA {date_to_print}

📊 MEDII LUNARE:
    Total: {(avg_mici_per_hour + avg_mari_per_hour):.1f}/oră
    Mari: {avg_mari_per_hour:.1f}/oră ({(total_mari/total*100) if total > 0 else 0:.1f}%)
    Mici: {avg_mici_per_hour:.1f}/oră ({(total_mici/total*100) if total > 0 else 0:.1f}%)

⏰ ORE DE VÂRF:
    🚛 Mari: {hours[peak_mari_idx] if hours and peak_mari_idx < len(hours) else 0}:00 ({avg_mari[peak_mari_idx] if avg_mari and peak_mari_idx < len(avg_mari) else 0:.1f})
    🚗 Mici: {hours[peak_mici_idx] if hours and peak_mici_idx < len(hours) else 0}:00 ({avg_mici[peak_mici_idx] if avg_mici and peak_mici_idx < len(avg_mici) else 0:.1f})

📈 DISTRIBUȚIA ORARĂ:"""

                    
        # Show only significant hours
        for i, hour in enumerate(hours):
            if i < len(avg_mari) and i < len(avg_mici):
                hour_total = avg_mari[i] + avg_mici[i]
                stats_text += f"""
    {hour} --> {avg_mari[i]:.1f} 🚛, {avg_mici[i]:.1f} 🚗 (total: {hour_total:.1f})"""
        
        # Display stats
        stats_label = tk.Label(self.text_stats_frame,
                            text=stats_text,
                            font=('Segoe UI', 9),
                            bg=self.colors['surface'],
                            fg=self.colors['dark'],
                            justify=tk.LEFT,
                            anchor='nw')
        stats_label.pack(fill=tk.BOTH, padx=5, pady=5)
    
    def generate_modern_percentage_distribution(self, date, formatted_date):
        """Generate modern percentage distribution"""
        data = self.get_filtered_daily_totals(date)
        
        if not data:
            self.show_no_data_message()
            return
        
        total_mari, total_mici = data
        total = total_mari + total_mici
        
        if total == 0:
            self.show_no_data_message("Nu există vehicule înregistrate pentru data selectată!")
            return
        
        # Create modern donut chart
        fig = self.create_modern_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        # Data for pie chart
        sizes = [total_mari, total_mici]
        labels = ['Vehicule Mari', 'Vehicule Mici']
        colors = [self.colors['danger'], self.colors['success']]
        explode = (0.05, 0)
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total):,})',
                                         shadow=True, startangle=90,
                                         textprops={'fontsize': 12, 'fontweight': 'bold'},
                                         pctdistance=0.85, labeldistance=1.1)
        
        # Create donut hole
        centre_circle = plt.Circle((0,0), 0.60, fc=self.colors['surface'], linewidth=2, edgecolor=self.colors['muted'])
        ax.add_artist(centre_circle)
        
        # Add center text
        ax.text(0, 0, f'TOTAL\n{total:,}\nvehicule', ha='center', va='center',
               fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax.axis('equal')
        ax.set_title(f'Distribuția vehiculelor\n{formatted_date}',
                    fontsize=16, fontweight='bold', color=self.colors['primary'], pad=20)
        
        fig.tight_layout(pad=2.0)
        
        # Display chart with optimized sizing
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Generate distribution stats with text format
        self.generate_distribution_stats_text(total, total_mari, total_mici, formatted_date)
    
    def generate_distribution_stats_text(self, total, total_mari, total_mici, date):
        """Generate compact text-based distribution statistics"""
        # Clear existing stats
        for widget in self.text_stats_frame.winfo_children():
            widget.destroy()
        
        percent_mari = (total_mari / total) * 100
        percent_mici = (total_mici / total) * 100
        
        # Generate insights
        if percent_mari > 60:
            dominant = "🚛 Dominanță vehicule mari"
            traffic_type = "Transport comercial"
        elif percent_mici > 60:
            dominant = "🚗 Dominanță vehicule mici"
            traffic_type = "Trafic personal"
        else:
            dominant = "⚖️ Distribuție echilibrată"
            traffic_type = "Trafic mixt"

        stats_text = f"""🥧 DISTRIBUȚIE {date}

📊 VEHICULE:
    Total: {total:,}
    Mari: {total_mari:,} ({percent_mari:.1f}%)
    Mici: {total_mici:,} ({percent_mici:.1f}%)

💡 ANALIZĂ:
    {dominant}
    Tipul de trafic: {traffic_type}
"""
        
        # Display stats
        stats_label = tk.Label(self.text_stats_frame,
                              text=stats_text,
                              font=('Segoe UI', 9),
                              bg=self.colors['surface'],
                              fg=self.colors['dark'],
                              justify=tk.LEFT,
                              anchor='nw')
        stats_label.pack(fill=tk.BOTH, padx=5, pady=5)
    
    def generate_modern_peak_hours_comparison(self, date, formatted_date):
        """Generate modern peak hours comparison"""
        peak_data = self.get_filtered_peak_hours_data(date)
        
        if not peak_data:
            self.show_no_data_message("Nu există date pentru analiza orelor de vârf!")
            return
        
        # Define peak hours
        peak_hours = [7, 8, 9, 16, 17, 18]
        peak_mari = sum(row[1] for row in peak_data if int(row[0]) in peak_hours)
        peak_mici = sum(row[2] for row in peak_data if int(row[0]) in peak_hours)
        normal_mari = sum(row[1] for row in peak_data if int(row[0]) not in peak_hours)
        normal_mici = sum(row[2] for row in peak_data if int(row[0]) not in peak_hours)
        
        # Create modern comparison chart
        fig = self.create_modern_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        categories = ['Ore de vârf\n(7-9, 16-18)', 'Ore normale']
        mari_data = [peak_mari, normal_mari]
        mici_data = [peak_mici, normal_mici]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars with modern styling
        bars1 = ax.bar(x - width/2, mari_data, width, label='Vehicule Mari',
                      color=self.colors['danger'], alpha=0.9,
                      edgecolor='white', linewidth=2)
        bars2 = ax.bar(x + width/2, mici_data, width, label='Vehicule Mici',
                      color=self.colors['success'], alpha=0.9,
                      edgecolor='white', linewidth=2)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(mari_data + mici_data) * 0.02,
                           f'{int(height):,}',
                           ha='center', va='bottom',
                           fontsize=12, fontweight='bold',
                           color=self.colors['dark'])
        
        # Modern styling
        ax.set_xlabel('Perioada zilei', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_ylabel('Numărul de vehicule', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_title(f'Comparația orelor de vârf cu orele normale\n{formatted_date}',
                    fontsize=16, fontweight='bold', color=self.colors['primary'], pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, ha='center')
        
        # Modern legend and grid
        legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor=self.colors['muted'])
        legend.get_frame().set_facecolor(self.colors['surface'])
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['muted'])
        ax.set_axisbelow(True)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        
        fig.tight_layout(pad=2.0)
        
        # Display chart with optimized sizing
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Generate peak hours analysis with text format
        total_peak = peak_mari + peak_mici
        total_normal = normal_mari + normal_mici
        self.generate_peak_analysis_text(total_peak, total_normal, peak_mari, peak_mici, normal_mari, normal_mici, formatted_date)
    
    def generate_peak_analysis_text(self, total_peak, total_normal, peak_mari, peak_mici, normal_mari, normal_mici, date):
        """Generate compact text-based peak hours analysis"""
        # Clear existing stats
        for widget in self.text_stats_frame.winfo_children():
            widget.destroy()
        
        # Calculate statistics
        factor = total_peak / total_normal if total_normal > 0 else 0
        
        stats_text = f"""🔥 ORE DE VÂRF {date}

📊 COMPARAȚIE:
    Ore de vârf: {total_peak:,}
    Ore normale: {total_normal:,}
    Factor: {factor:.1f}x mai intens la ore de vârf

🚛 VEHICULE MARI:
    Ore de vârf: {peak_mari:,} ({(peak_mari/total_peak*100) if total_peak > 0 else 0:.1f}%)
    Normal: {normal_mari:,} ({(normal_mari/total_normal*100) if total_normal > 0 else 0:.1f}%)

🚗 VEHICULE MICI:
    Ore de vârf: {peak_mici:,} ({(peak_mici/total_peak*100) if total_peak > 0 else 0:.1f}%)
    Normal: {normal_mici:,} ({(normal_mici/total_normal*100) if total_normal > 0 else 0:.1f}%)
"""
        
        # Display stats
        stats_label = tk.Label(self.text_stats_frame,
                              text=stats_text,
                              font=('Segoe UI', 9),
                              bg=self.colors['surface'],
                              fg=self.colors['dark'],
                              justify=tk.LEFT,
                              anchor='nw')
        stats_label.pack(fill=tk.BOTH, padx=5, pady=5)
    
    def show_no_data_message(self, custom_message=None):
        """Show modern no data message"""
        message = custom_message or "Nu există date pentru data selectată!"
        
        # Clear existing content
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
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

    
    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'db'):
            self.db.close()

# Application entry point with error handling
if __name__ == "__main__":
    try:
        root = tk.Tk()
        
        # Set window icon and properties
        root.iconname("Traffic Analyzer")
        root.minsize(1200, 800)
        
        # Initialize app
        app = BinaryTrafficAnalyzerApp(root)
        
        # Add error handling for main loop
        def on_closing():
            try:
                if hasattr(app, 'db'):
                    app.db.close()
            except:
                pass
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start application
        root.mainloop()
        
    except Exception as e:
        import tkinter.messagebox as mb
        mb.showerror("Eroare de Inițializare", 
                    f"Nu s-a putut inițializa aplicația:\n{str(e)}\n\nVerificați dacă toate dependențele sunt instalate.")
    finally:
        try:
            root.destroy()
        except:
            pass