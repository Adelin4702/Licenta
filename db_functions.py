import sqlite3
import datetime

class TrafficDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        """Connect to SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
    
    def create_tables(self):
        """Create tables for traffic data"""
        # Create table for normal (4-class) model
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ora DATETIME,
                numar_masini INTEGER,
                numar_autoutilitare INTEGER,
                numar_camioane INTEGER,
                numar_autobuze INTEGER
            )
        ''')
        
        # Create table for binary model
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data_binary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ora DATETIME,
                numar_vehicule_mari INTEGER,
                numar_vehicule_mici INTEGER
            )
        ''')
        
        self.conn.commit()
    
    def save_normal_data(self, timestamp, cars, vans, trucks, buses):
        """Save data from 4-class model"""
        try:
            self.cursor.execute('''
                INSERT INTO traffic_data (ora, numar_masini, numar_autoutilitare, numar_camioane, numar_autobuze)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, cars, vans, trucks, buses))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving normal data: {e}")
    
    def save_binary_data(self, timestamp, large_vehicles, small_vehicles):
        """Save data from binary model"""
        try:
            self.cursor.execute('''
                INSERT INTO traffic_data_binary (ora, numar_vehicule_mari, numar_vehicule_mici)
                VALUES (?, ?, ?)
            ''', (timestamp, large_vehicles, small_vehicles))
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error saving binary data: {e}")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def get_next_hour_timestamp(current_time):
    """Calculate next hour timestamp (e.g., 12:00:00)"""
    next_hour = current_time.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    return next_hour