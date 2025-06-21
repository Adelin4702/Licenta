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
            -- Tabelul City
            CREATE TABLE city (
                id INTEGER PRIMARY KEY,
                city_name VARCHAR(255)
            );

            -- Tabelul Location
            CREATE TABLE location (
                id INTEGER PRIMARY KEY,
                city_id INTEGER,
                road_name VARCHAR(255),
                road_km INTEGER,
                FOREIGN KEY (CityId) REFERENCES City(id)
            );

            -- Tabelul Camera
            CREATE TABLE camera (
                id INTEGER PRIMARY KEY,
                location_id INTEGER,
                model VARCHAR(255),
                frame_width INTEGER,
                frame_height INTEGER,
                fps INTEGER,
                FOREIGN KEY (LocationId) REFERENCES Location(id)
            );

            -- Tabelul Record
            CREATE TABLE record (
                id INTEGER PRIMARY KEY,
                camera_id INTEGER,
                date TIMESTAMP,
                nr_of_small_vehicles INTEGER,
                nr_of_large_vehicles INTEGER,
                FOREIGN KEY (CameraId) REFERENCES Camera(id)
            );
                            
            -- Inserează date de test
            INSERT INTO City (city_name) VALUES 
                ('Cluj-Napoca'),
                ('București'),
                ('Timișoara');

            INSERT INTO Location (CityId, road_name, road_km) VALUES 
                (1, 'Calea Turzii', 15),
                (1, 'Strada Memorandumului', 3),
                (2, 'Șoseaua Kiseleff', 8);

            INSERT INTO Camera (LocationId, model, frame_width, frame_height, fps) VALUES 
                (1, 'Hikvision DS-TCG405', 960, 540, 30),
                (2, 'Dahua ITC413-RW1F-Z', 1920, 1080, 25);

            INSERT INTO Record (CameraId, date, nr_of_cars, nr_of_vans, nr_of_trucks) VALUES 
                (1, '2024-01-15 10:30:00', 25, 8, 3),
                (1, '2024-01-15 11:30:00', 30, 12, 5),
                (2, '2024-01-15 10:45:00', 18, 6, 2);
        ''')
        
        self.conn.commit()
    
    def save_binary_data(self, cameraId, timestamp, large_vehicles, small_vehicles):
        """Save data from binary model"""
        try:
            self.cursor.execute('''
                INSERT INTO traffic_data_binary (camera_id, 
                                                date, 
                                                nr_of_small_vehicles, 
                                                nr_of_large_vehicles)
                VALUES (?, ?, ?)
            ''', (cameraId, timestamp, large_vehicles, small_vehicles))
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