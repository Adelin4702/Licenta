"""
Date utility functions for Traffic Analyzer App
"""
import datetime
from .constants import DATE_FORMAT_INPUT, DATE_FORMAT_DISPLAY, DATE_FORMAT_SHORT

class DateUtils:
    """Utility class for date operations"""
    
    @staticmethod
    def format_date_for_display(date_str):
        """Format date string yyyy-mm-dd to dd/mm/yyyy"""
        try:
            return datetime.datetime.strptime(date_str, DATE_FORMAT_INPUT).strftime(DATE_FORMAT_DISPLAY)
        except Exception:
            return date_str
    
    @staticmethod
    def format_date_short(date_str):
        """Format date string yyyy-mm-dd to dd/mm"""
        try:
            return datetime.datetime.strptime(date_str, DATE_FORMAT_INPUT).strftime(DATE_FORMAT_SHORT)
        except Exception:
            return date_str
    
    @staticmethod
    def get_week_range(date_str):
        """Get start and end dates for the week containing the given date"""
        try:
            selected_date = datetime.datetime.strptime(date_str, DATE_FORMAT_INPUT)
            days_since_monday = selected_date.weekday()
            week_start = selected_date - datetime.timedelta(days=days_since_monday)
            week_end = week_start + datetime.timedelta(days=6)
            
            return (
                week_start.strftime(DATE_FORMAT_INPUT),
                week_end.strftime(DATE_FORMAT_INPUT),
                week_start,
                week_end
            )
        except Exception:
            return None, None, None, None
    
    @staticmethod
    def get_current_date():
        """Get current date in standard format"""
        return datetime.datetime.now().strftime(DATE_FORMAT_INPUT)
    
    @staticmethod
    def get_month_year(date_str):
        """Extract month/year from date string"""
        try:
            date_obj = datetime.datetime.strptime(date_str, DATE_FORMAT_INPUT)
            return date_obj.strftime("%m/%Y")
        except Exception:
            return ""
    
    @staticmethod
    def get_week_day_labels(week_start):
        """Generate day labels for a week starting from week_start"""
        day_labels = []
        for i in range(7):
            current_day = week_start + datetime.timedelta(days=i)
            day_label = current_day.strftime("%a\n%d/%m")
            day_labels.append(day_label)
        return day_labels
    
    @staticmethod
    def parse_date_safely(date_str, format_str=DATE_FORMAT_INPUT):
        """Safely parse date string"""
        try:
            return datetime.datetime.strptime(date_str, format_str)
        except ValueError:
            return None