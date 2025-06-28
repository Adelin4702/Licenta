"""
Main application file for Traffic Analyzer App - WITH ALL DATABASE FILTERS
Entry point and application orchestration
"""
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
from tkcalendar import Calendar
import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from styles.modern_styles import ModernStyles
from components.calendar_panel import CalendarPanelComponent
from components.controls_panel import ControlsPanelComponent
from components.stats_panel import StatsPanelComponent
from components.filters_panel import FiltersPanelComponent
from visualization.hourly_viz import HourlyVisualization
from visualization.weekly_viz import WeeklyVisualization
from visualization.monthly_viz import MonthlyVisualization
from visualization.distribution_viz import DistributionVisualization
from visualization.peak_hours_viz import PeakHoursVisualization
from utils.date_utils import DateUtils
from utils.constants import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from MonitoringApp.db_functions import TrafficDatabase

from visualization.yearly_viz import YearlyVisualization 

class TrafficAnalyzerApp:
    """Main application class for Traffic Analyzer with ALL DATABASE FILTERS"""
    
    def __init__(self, root):
        self.root = root
        self.setup_window()
        
        # Initialize styling
        self.styles = ModernStyles()
        self.styles.setup_matplotlib_style()
        self.ttk_style = self.styles.setup_ttk_styles(root)
        
        # Initialize database
        self.db = TrafficDatabase(DATABASE_NAME)
        self.dates_with_data = self.db.get_dates_with_data()
        
        # Initialize filter data from database
        self.cities_data = {}
        self.locations_data = {}
        self.cameras_data = {}
        self.load_filter_data()
        
        # Initialize components
        self.selected_date = DateUtils.get_current_date()
        self.visualizations = {}
        
        # Create GUI
        self.create_gui()
        
        # Setup visualizations
        self.setup_visualizations()
    
    def setup_window(self):
        """Configure main window properties"""
        self.root.title(APP_TITLE)
        self.root.state('zoomed')
        self.root.minsize(*MIN_WINDOW_SIZE)
        self.root.iconname("Traffic Analyzer")
    
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
    
    def create_gui(self):
        """Create the main GUI layout with optimized spacing"""
        # Main container with reduced padding for 1200x800
        main_frame = tk.Frame(self.root, bg=self.styles.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Compact header
        self.create_compact_header(main_frame)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg=self.styles.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel (controls) - optimized width for 1200x800
        self.create_left_panel(content_frame)
        
        # Right panel (visualization)
        self.create_right_panel(content_frame)
    
    def create_compact_header(self, parent):
        """Create compact header optimized for 1200x800"""
        header_frame = tk.Frame(parent, bg=self.styles.colors['surface'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Add subtle shadow effect
        shadow_frame = tk.Frame(parent, bg='#e2e8f0', height=1)
        shadow_frame.pack(fill=tk.X)
        
        # Header content
        header_content = tk.Frame(header_frame, bg=self.styles.colors['surface'])
        header_content.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)
        
        # Title and status in same line
        title_status_frame = tk.Frame(header_content, bg=self.styles.colors['surface'])
        title_status_frame.pack(fill=tk.X)
        
        # Title on left
        title_label = tk.Label(title_status_frame, 
                              text="🚗 Analizator Trafic Inteligent",
                              font=('Segoe UI', 18, 'bold'),
                              bg=self.styles.colors['surface'],
                              fg=self.styles.colors['primary'])
        title_label.pack(side=tk.LEFT)
        
        # Subtitle
        subtitle_label = tk.Label(header_content,
                                 text="Analiză avansată pentru vehicule mari și mici",
                                 font=('Segoe UI', 11),
                                 bg=self.styles.colors['surface'],
                                 fg=self.styles.colors['secondary'])
        subtitle_label.pack(anchor='w', pady=(2, 0))
    
    def create_left_panel(self, parent):
        """Create left control panel with all filters and controls"""
        left_panel = tk.Frame(parent, bg=self.styles.colors['background'], width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Calendar component
        self.calendar_panel = CalendarPanelComponent(
            left_panel, 
            self.styles, 
            self.dates_with_data, 
            self.on_date_select
        )
        
        # Filters component - THE IMPORTANT PART
        self.filters_panel = FiltersPanelComponent(
            left_panel,
            self.styles,
            self.cities_data,
            self.locations_data,
            self.cameras_data,
            self.on_filter_change
        )
        
        # Controls component
        self.controls_panel = ControlsPanelComponent(
            left_panel, 
            self.styles, 
            self.on_visualization_change, 
            self.add_test_data
        )
    
    def create_right_panel(self, parent):
        """Create right visualization panel"""
        right_panel = tk.Frame(parent, bg=self.styles.colors['background'])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Visualization card
        viz_card = tk.Frame(right_panel, bg=self.styles.colors['surface'], relief='solid', bd=1)
        viz_card.pack(fill=tk.BOTH, expand=True)
        
        # Card header
        header_frame = tk.Frame(viz_card, bg=self.styles.colors['dark'], height=35)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        self.viz_title = tk.Label(
            header_frame, 
            text="📈 Vizualizare Trafic",
            font=('Segoe UI', FONT_SIZES['card_header'], 'bold'),
            bg=self.styles.colors['dark'],
            fg='white'
        )
        self.viz_title.pack(expand=True)
        
        # Content container with horizontal split
        content_container = tk.Frame(viz_card, bg=self.styles.colors['surface'])
        content_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Graph area - takes most of the width
        self.graph_frame = tk.Frame(content_container, bg=self.styles.colors['light'])
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Stats panel - compact for 1200x800
        self.stats_panel = StatsPanelComponent(content_container, self.styles)
        
        # Initial placeholder
        self.show_initial_placeholder()
    
    def setup_visualizations(self):
        """Initialize visualization objects with filter support"""
        self.visualizations = {
            "📊 Trafic orar": HourlyVisualization(self.graph_frame, self.stats_panel, self.styles, self.db),
            "🥧 Distribuție procentuală": DistributionVisualization(self.graph_frame, self.stats_panel, self.styles, self.db),
            "🔥 Comparație ore de vârf": PeakHoursVisualization(self.graph_frame, self.stats_panel, self.styles, self.db),
            "📈 Trafic săptămânal": WeeklyVisualization(self.graph_frame, self.stats_panel, self.styles, self.db),
            "📅 Tendință lunară": MonthlyVisualization(self.graph_frame, self.stats_panel, self.styles, self.db),
            "🗓️ Evoluție anuală": YearlyVisualization(self.graph_frame, self.stats_panel, self.styles, self.db)
        }
        
        # Set filter capability for all visualizations
        for viz in self.visualizations.values():
            if hasattr(viz, 'set_filter_method'):
                viz.set_filter_method(self.get_filtered_data)
    
    def show_initial_placeholder(self):
        """Show initial placeholder message"""
        placeholder_label = tk.Label(
            self.graph_frame,
            text="🎯\nSelectează o opțiune de vizualizare\npentru a începe analiza",
            font=('Segoe UI', 12),
            bg=self.styles.colors['light'],
            fg=self.styles.colors['muted'],
            justify=tk.CENTER
        )
        placeholder_label.pack(expand=True)
    
    def clear_initial_placeholder(self):
        """Clear the initial placeholder - FIXES BANNER ISSUE"""
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
    
    def on_filter_change(self):
        """Handle filter changes - regenerate current visualization"""
        self.generate_current_visualization()
    
    def on_date_select(self, selected_date):
        """Handle date selection from calendar"""
        self.selected_date = selected_date
        self.generate_current_visualization()
    
    def on_visualization_change(self, view_type):
        """Handle visualization type change"""
        self.generate_visualization(view_type)
    
    def generate_current_visualization(self):
        """Generate currently selected visualization"""
        current_view = self.controls_panel.get_selected_view_type()
        self.generate_visualization(current_view)
    
    def generate_visualization(self, view_type):
        """Generate selected visualization with filters applied"""
        # Clear any existing content first (THIS FIXES THE BANNER ISSUE)
        self.clear_initial_placeholder()
        
        # Get filter description for title
        filter_desc = self.get_filter_description()
        self.viz_title.configure(text=f"{view_type}{filter_desc}")
        
        # Get formatted date
        formatted_date = DateUtils.format_date_for_display(self.selected_date)
        
        # Generate appropriate visualization using filtered data methods
        if view_type in self.visualizations:
            try:
                # Replace db methods with filtered versions temporarily
                self._override_db_methods()
                
                viz = self.visualizations[view_type]
                viz.generate_visualization(self.selected_date, formatted_date)
                
                # Restore original db methods
                self._restore_db_methods()
                
            except Exception as e:
                self._restore_db_methods()  # Ensure methods are restored
                self.show_error_message(f"Eroare la generarea vizualizării: {str(e)}")
        else:
            self.show_error_message(f"Tip de vizualizare necunoscut: {view_type}")
    
    def _override_db_methods(self):
        """Temporarily override db methods with filtered versions"""
        # Store original methods
        self._original_methods = {
            'get_hourly_data': self.db.get_hourly_data,
            'get_daily_totals': self.db.get_daily_totals,
            'get_week_data_by_range': self.db.get_week_data_by_range,
            'get_monthly_trend': self.db.get_monthly_trend,
            'get_peak_hours_data': self.db.get_peak_hours_data
        }
        
        # Replace with filtered versions
        self.db.get_hourly_data = self.get_filtered_hourly_data
        self.db.get_daily_totals = self.get_filtered_daily_totals
        self.db.get_week_data_by_range = self.get_filtered_week_data_by_range
        self.db.get_monthly_trend = self.get_filtered_monthly_trend
        self.db.get_peak_hours_data = self.get_filtered_peak_hours_data
        
        # Add yearly method override for the new visualization
        if hasattr(self.db, 'get_yearly_data'):
            self._original_methods['get_yearly_data'] = self.db.get_yearly_data
        self.db.get_yearly_data = self.get_filtered_yearly_data
    
    def _restore_db_methods(self):
        """Restore original db methods"""
        if hasattr(self, '_original_methods'):
            for method_name, original_method in self._original_methods.items():
                setattr(self.db, method_name, original_method)
    
    def get_filter_conditions(self):
        """Get SQL conditions based on current filter selections"""
        conditions = []
        params = []
        
        filters = self.filters_panel.get_selected_filters()
        selected_city = filters['city']
        selected_location = filters['location']
        selected_camera = filters['camera']
        
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
    
    def get_filter_description(self):
        """Get description of current filters for display in titles"""
        parts = []
        filters = self.filters_panel.get_selected_filters()
        
        if filters['city'] != "Toate orașele":
            parts.append(f"Oraș: {filters['city']}")
            
        if filters['location'] != "Toate locațiile":
            parts.append(f"Locație: {filters['location']}")
            
        if filters['camera'] != "Toate camerele":
            parts.append(f"Cameră: {filters['camera']}")
        
        if parts:
            return " | " + " | ".join(parts)
        else:
            return " | Toate datele"
    
    # FILTERED DATA METHODS - The core filtering functionality
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
            
            params.insert(0, date)
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
            
            params.insert(0, date)
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
            
            params.insert(0, start_date)
            params.insert(1, end_date)
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
            
            params.insert(0, month)
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
            
            params.insert(0, date)
            self.db.cursor.execute(base_query, params)
            return self.db.cursor.fetchall()
        except Exception as e:
            print(f"Error getting filtered peak hours data: {e}")
            return []
    
    def get_filtered_yearly_data(self, year):
        """Get yearly data with filters applied - NEW METHOD"""
        try:
            conditions, params = self.get_filter_conditions()
            
            base_query = '''
                SELECT strftime('%m', r.timestamp) as month,
                       SUM(r.nr_of_large_vehicles) as large_vehicles,
                       SUM(r.nr_of_small_vehicles) as small_vehicles
                FROM record r
                JOIN camera c ON r.camera_id = c.id
                JOIN location l ON c.location_id = l.id
                JOIN city ON l.city_id = city.id
                WHERE strftime('%Y', r.timestamp) = ?
            '''
            
            if conditions:
                base_query += " AND " + " AND ".join(conditions)
            
            base_query += '''
                GROUP BY strftime('%m', r.timestamp)
                ORDER BY month
            '''
            
            params.insert(0, year)
            self.db.cursor.execute(base_query, params)
            return self.db.cursor.fetchall()
        except Exception as e:
            print(f"Error getting filtered yearly data: {e}")
            return []
    
    def add_test_data(self):
        """Add test data to database"""
        try:
            success = self.db.generate_test_data(days=7)
            if success:
                self.dates_with_data = self.db.get_dates_with_data()
                self.calendar_panel.set_dates_with_data(self.dates_with_data)
                # Reload filter data in case new cities/locations/cameras were added
                self.load_filter_data()
                self.filters_panel.update_filter_data(self.cities_data, self.locations_data, self.cameras_data)
            return success
        except Exception as e:
            self.show_error_message(f"Eroare la adăugarea datelor de test: {str(e)}")
            return False
    
    def show_error_message(self, message):
        """Show error message in visualization area"""
        # Clear existing content
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        error_label = tk.Label(
            self.graph_frame,
            text=f"⚠️\n{message}",
            font=('Segoe UI', FONT_SIZES['normal']),
            bg=self.styles.colors['light'],
            fg=self.styles.colors['danger'],
            justify=tk.CENTER
        )
        error_label.pack(expand=True)
    
    def on_closing(self):
        """Handle application closing"""
        try:
            self._restore_db_methods()  # Ensure methods are restored
            if hasattr(self, 'db'):
                self.db.close()
        except:
            pass
        self.root.destroy()

def main():
    """Main application entry point"""
    try:
        root = tk.Tk()
        app = TrafficAnalyzerApp(root)
        
        # Setup close handler
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        # Start application
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror(
            "Eroare de Inițializare", 
            f"{MESSAGES['init_error']}:\n{str(e)}\n\n{MESSAGES['check_dependencies']}"
        )
    finally:
        try:
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    main()