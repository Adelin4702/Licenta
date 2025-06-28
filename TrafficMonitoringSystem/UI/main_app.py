"""
Main application file for Traffic Analyzer App
Entry point and application orchestration
"""
import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from UI.styles.modern_styles import ModernStyles
from UI.components.header import HeaderComponent
from UI.components.calendar_panel import CalendarPanelComponent
from UI.components.controls_panel import ControlsPanelComponent
from UI.components.stats_panel import StatsPanelComponent
from UI.visualization.hourly_viz import HourlyVisualization
from UI.visualization.weekly_viz import WeeklyVisualization
from UI.visualization.monthly_viz import MonthlyVisualization
from UI.visualization.distribution_viz import DistributionVisualization
from UI.visualization.peak_hours_viz import PeakHoursVisualization
from UI.utils.constants import *
from UI.utils.date_utils import DateUtils
from MonitoringApp.db_functions import TrafficDatabase

class TrafficAnalyzerApp:
    """Main application class for Traffic Analyzer"""
    
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
    
    def create_gui(self):
        """Create the main GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.styles.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        self.header = HeaderComponent(main_frame, self.styles, len(self.dates_with_data))
        
        # Content area
        content_frame = tk.Frame(main_frame, bg=self.styles.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left panel (controls)
        self.create_left_panel(content_frame)
        
        # Right panel (visualization)
        self.create_right_panel(content_frame)
    
    def create_left_panel(self, parent):
        """Create left control panel"""
        left_panel = tk.Frame(parent, bg=self.styles.colors['background'], width=CALENDAR_PANEL_WIDTH)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        left_panel.pack_propagate(False)
        
        # Calendar component
        self.calendar_panel = CalendarPanelComponent(
            left_panel, 
            self.styles, 
            self.dates_with_data, 
            self.on_date_select
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
        header_frame = tk.Frame(viz_card, bg=self.styles.colors['dark'], height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        self.viz_title = tk.Label(
            header_frame, 
            text="📈 Vizualizare Trafic",
            font=('Segoe UI', FONT_SIZES['header'], 'bold'),
            bg=self.styles.colors['dark'],
            fg='white'
        )
        self.viz_title.pack(expand=True)
        
        # Content container
        content_container = tk.Frame(viz_card, bg=self.styles.colors['surface'])
        content_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Graph area
        self.graph_frame = tk.Frame(content_container, bg=self.styles.colors['light'])
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Stats panel
        self.stats_panel = StatsPanelComponent(content_container, self.styles)
        
        # Initial placeholder
        self.show_initial_placeholder()
    
    def setup_visualizations(self):
        """Initialize visualization objects"""
        self.visualizations = {
            "📊 Trafic orar": HourlyVisualization(self.graph_frame, self.stats_panel, self.styles, self.db),
            "📈 Trafic săptămânal": WeeklyVisualization(self.graph_frame, self.stats_panel, self.styles, self.db),
            "📅 Tendință lunară": MonthlyVisualization(self.graph_frame, self.stats_panel, self.styles, self.db),
            "🥧 Distribuție procentuală": DistributionVisualization(self.graph_frame, self.stats_panel, self.styles, self.db),
            "🔥 Comparație ore de vârf": PeakHoursVisualization(self.graph_frame, self.stats_panel, self.styles, self.db)
        }
    
    def show_initial_placeholder(self):
        """Show initial placeholder message"""
        placeholder_label = tk.Label(
            self.graph_frame,
            text="🎯\nSelectează o opțiune de vizualizare\npentru a începe analiza",
            font=('Segoe UI', FONT_SIZES['large']),
            bg=self.styles.colors['light'],
            fg=self.styles.colors['muted'],
            justify=tk.CENTER
        )
        placeholder_label.pack(expand=True)
    
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
        """Generate selected visualization"""
        # Update title
        self.viz_title.configure(text=f"{view_type}")
        
        # Get formatted date
        formatted_date = DateUtils.format_date_for_display(self.selected_date)
        
        # Generate appropriate visualization
        if view_type in self.visualizations:
            try:
                self.visualizations[view_type].generate_visualization(self.selected_date, formatted_date)
            except Exception as e:
                self.show_error_message(f"Eroare la generarea vizualizării: {str(e)}")
        else:
            self.show_error_message(f"Tip de vizualizare necunoscut: {view_type}")
    
    def add_test_data(self):
        """Add test data to database"""
        try:
            success = self.db.generate_test_data(days=7)
            if success:
                self.dates_with_data = self.db.get_dates_with_data()
                self.calendar_panel.set_dates_with_data(self.dates_with_data)
                self.header.update_status(len(self.dates_with_data))
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