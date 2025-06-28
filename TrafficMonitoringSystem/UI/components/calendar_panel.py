"""
Calendar panel component for Traffic Analyzer App
"""
import tkinter as tk
import datetime
from tkcalendar import Calendar
from UI.utils.constants import CARD_HEADER_HEIGHT, FONT_SIZES
from UI.utils.date_utils import DateUtils

class CalendarPanelComponent:
    """Calendar widget with date selection functionality"""
    
    def __init__(self, parent, styles, dates_with_data=None, on_date_select_callback=None):
        self.parent = parent
        self.styles = styles
        self.colors = styles.colors
        self.dates_with_data = dates_with_data or []
        self.on_date_select_callback = on_date_select_callback
        self.selected_date = DateUtils.get_current_date()
        
        self.create_calendar_panel()
    
    def create_calendar_panel(self):
        """Create calendar card with date selection"""
        # Main card frame
        self.card_frame = tk.Frame(self.parent, bg=self.colors['surface'], relief='solid', bd=1)
        self.card_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Card header
        self._create_card_header()
        
        # Calendar content
        self._create_calendar_content()
        
        return self.card_frame
    
    def _create_card_header(self):
        """Create compact card header"""
        header_frame = tk.Frame(self.card_frame, bg=self.colors['primary'], height=CARD_HEADER_HEIGHT)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame, 
            text="📅 Data",
            font=('Segoe UI', FONT_SIZES['card_header'], 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        header_label.pack(expand=True)
    
    def _create_calendar_content(self):
        """Create calendar widget"""
        calendar_content = tk.Frame(self.card_frame, bg=self.colors['surface'])
        calendar_content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Calendar widget with modern styling
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
            font=('Segoe UI', 8)  # Smaller font for compact view
        )
        self.calendar.pack(pady=5)
        
        # Bind calendar selection event
        self.calendar.bind("<<CalendarSelected>>", self._on_calendar_select)
        
        # Mark dates with data
        self.update_calendar_marks()
    
    def _on_calendar_select(self, event):
        """Handle calendar date selection"""
        self.selected_date = self.calendar.get_date()
        
        # Call external callback if provided
        if self.on_date_select_callback:
            self.on_date_select_callback(self.selected_date)
    
    def update_calendar_marks(self):
        """Mark days with data in calendar"""
        # Clear existing marks
        for event_id in self.calendar.get_calevents():
            self.calendar.calevent_remove(event_id)
        
        # Add marks for dates with data
        for date_str in self.dates_with_data:
            try:
                date_obj = DateUtils.parse_date_safely(date_str)
                if date_obj:
                    self.calendar.calevent_create(date_obj, "Date disponibile", "highlight")
            except ValueError:
                continue
        
        # Configure highlight style
        self.calendar.tag_config('highlight', background='#10b981', foreground='white')
    
    def set_dates_with_data(self, dates_list):
        """Update the list of dates with data"""
        self.dates_with_data = dates_list
        self.update_calendar_marks()
    
    def get_selected_date(self):
        """Get currently selected date"""
        return self.selected_date
    
    def set_selected_date(self, date_str):
        """Set the selected date programmatically"""
        try:
            date_obj = DateUtils.parse_date_safely(date_str)
            if date_obj:
                self.calendar.selection_set(date_obj)
                self.selected_date = date_str
        except Exception:
            pass
    
    def set_date_select_callback(self, callback):
        """Set callback function for date selection"""
        self.on_date_select_callback = callback