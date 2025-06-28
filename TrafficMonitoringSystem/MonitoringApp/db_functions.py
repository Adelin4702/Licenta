import sqlite3
import datetime
import random


class TrafficDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.connect()
        self.create_tables()
    
    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            # Enable foreign keys
            self.cursor.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
    
    def create_tables(self):
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS city (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    city_name VARCHAR(255) NOT NULL UNIQUE
                )
            ''')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS location (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    city_id INTEGER NOT NULL,
                    road_name VARCHAR(255) NOT NULL,
                    road_km INTEGER,
                    FOREIGN KEY (city_id) REFERENCES city(id)
                )
            ''')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS camera (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    location_id INTEGER NOT NULL,
                    model VARCHAR(255),
                    frame_width INTEGER,
                    frame_height INTEGER,
                    fps INTEGER,
                    FOREIGN KEY (location_id) REFERENCES location(id)
                )
            ''')

            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS record (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    nr_of_large_vehicles INTEGER DEFAULT 0,
                    nr_of_small_vehicles INTEGER DEFAULT 0,
                    FOREIGN KEY (camera_id) REFERENCES camera(id)
                )
            ''')

            self.insert_initial_data()
            self.conn.commit()
            
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
    
    def insert_initial_data(self):
        try:
            self.cursor.execute("SELECT COUNT(*) FROM city")
            if self.cursor.fetchone()[0] == 0:
                cities = [
                    ('Cluj-Napoca',),
                    ('București',),
                    ('Timișoara',),
                    ('Iași',),
                    ('Constanța',)
                ]
                self.cursor.executemany("INSERT INTO city (city_name) VALUES (?)", cities)

                locations = [
                    (1, 'Calea Turzii', 15),
                    (1, 'Strada Memorandumului', 3),
                    (1, 'Calea Florești', 8),
                    (2, 'Șoseaua Kiseleff', 12),
                    (2, 'Calea Victoriei', 5),
                    (3, 'Calea Aradului', 20),
                ]
                self.cursor.executemany(
                    "INSERT INTO location (city_id, road_name, road_km) VALUES (?, ?, ?)", 
                    locations
                )

                cameras = [
                    (1, 'Hikvision DS-TCG405', 960, 540, 30),
                    (2, 'Dahua ITC413-RW1F-Z', 1920, 1080, 25),
                    (3, 'Axis P1448-LE', 1920, 1080, 30),
                    (4, 'Bosch MIC IP starlight 7000i', 1920, 1080, 25),
                    (5, 'Hanwha XNP-6320H', 1920, 1080, 30),
                ]
                self.cursor.executemany(
                    "INSERT INTO camera (location_id, model, frame_width, frame_height, fps) VALUES (?, ?, ?, ?, ?)", 
                    cameras
                )
                
        except sqlite3.Error as e:
            print(f"Error inserting initial data: {e}")
    
    def save_traffic_data(self, camera_id, timestamp, large_vehicles, small_vehicles):
        try:
            self.cursor.execute('''
                INSERT INTO record (camera_id, timestamp, nr_of_large_vehicles, nr_of_small_vehicles)
                VALUES (?, ?, ?, ?)
            ''', (camera_id, timestamp, large_vehicles, small_vehicles))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error saving traffic data: {e}")
            return False
    
    def get_dates_with_data(self):
        try:
            self.cursor.execute('''
                SELECT DISTINCT date(timestamp) as date_only
                FROM record 
                ORDER BY date_only DESC
            ''')
            return [row[0] for row in self.cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error getting dates with data: {e}")
            return []
    
    def get_hourly_data(self, date):
        try:
            self.cursor.execute('''
                SELECT strftime('%H', timestamp) as hour,
                       SUM(nr_of_large_vehicles) as large_vehicles,
                       SUM(nr_of_small_vehicles) as small_vehicles
                FROM record
                WHERE date(timestamp) = ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''', (date,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting hourly data: {e}")
            return []
    
    def get_week_data_by_range(self, start_date, end_date):
        try:
            self.cursor.execute('''
                SELECT date(timestamp) as day,
                       SUM(nr_of_large_vehicles) as large_vehicles,
                       SUM(nr_of_small_vehicles) as small_vehicles
                FROM record
                WHERE date(timestamp) BETWEEN ? AND ?
                GROUP BY date(timestamp)
                ORDER BY day
            ''', (start_date, end_date))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting week data by range: {e}")
            return []
    
    def get_weekly_data(self, end_date):
        try:
            start_date = (datetime.datetime.strptime(end_date, "%Y-%m-%d") - 
                         datetime.timedelta(days=6)).strftime("%Y-%m-%d")
            
            self.cursor.execute('''
                SELECT date(timestamp) as day,
                       SUM(nr_of_large_vehicles) as large_vehicles,
                       SUM(nr_of_small_vehicles) as small_vehicles
                FROM record
                WHERE date(timestamp) BETWEEN ? AND ?
                GROUP BY date(timestamp)
                ORDER BY day
            ''', (start_date, end_date))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting weekly data: {e}")
            return []
    
    def get_monthly_trend(self, month):
        try:
            self.cursor.execute('''
                SELECT strftime('%H', timestamp) as hour,
                       AVG(nr_of_large_vehicles) as avg_large,
                       AVG(nr_of_small_vehicles) as avg_small
                FROM record
                WHERE strftime('%Y-%m', timestamp) = ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''', (month,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting monthly trend: {e}")
            return []
    
    def get_daily_totals(self, date):
        try:
            self.cursor.execute('''
                SELECT SUM(nr_of_large_vehicles) as total_large,
                       SUM(nr_of_small_vehicles) as total_small
                FROM record
                WHERE date(timestamp) = ?
            ''', (date,))
            result = self.cursor.fetchone()
            return (result[0] or 0, result[1] or 0)
        except sqlite3.Error as e:
            print(f"Error getting daily totals: {e}")
            return (0, 0)
    
    def get_peak_hours_data(self, date):
        try:
            self.cursor.execute('''
                SELECT strftime('%H', timestamp) as hour,
                       SUM(nr_of_large_vehicles) as large_vehicles,
                       SUM(nr_of_small_vehicles) as small_vehicles
                FROM record
                WHERE date(timestamp) = ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            ''', (date,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting peak hours data: {e}")
            return []
    
    def get_cameras_info(self):
        try:
            self.cursor.execute('''
                SELECT c.id, c.model, c.frame_width, c.frame_height, c.fps,
                       l.road_name, l.road_km, city.city_name
                FROM camera c
                JOIN location l ON c.location_id = l.id
                JOIN city ON l.city_id = city.id
                ORDER BY city.city_name, l.road_name
            ''')
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting cameras info: {e}")
            return []
    
    def get_statistics_summary(self, start_date=None, end_date=None):
        try:
            base_query = '''
                SELECT 
                    COUNT(*) as total_records,
                    SUM(nr_of_large_vehicles) as total_large,
                    SUM(nr_of_small_vehicles) as total_small,
                    AVG(nr_of_large_vehicles) as avg_large,
                    AVG(nr_of_small_vehicles) as avg_small,
                    MAX(nr_of_large_vehicles + nr_of_small_vehicles) as max_total,
                    MIN(timestamp) as first_record,
                    MAX(timestamp) as last_record
                FROM record
            '''
            
            if start_date and end_date:
                query = base_query + " WHERE date(timestamp) BETWEEN ? AND ?"
                self.cursor.execute(query, (start_date, end_date))
            elif start_date:
                query = base_query + " WHERE date(timestamp) >= ?"
                self.cursor.execute(query, (start_date,))
            elif end_date:
                query = base_query + " WHERE date(timestamp) <= ?"
                self.cursor.execute(query, (end_date,))
            else:
                self.cursor.execute(base_query)
            
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            print(f"Error getting statistics summary: {e}")
            return None
    
    def generate_test_data(self, days=7, camera_id=1):
        try:
            base_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            for day in range(days):
                current_date = base_date + datetime.timedelta(days=day)
                for hour in range(24):
                    current_datetime = current_date.replace(hour=hour, minute=0, second=0, microsecond=0)

                    if 7 <= hour <= 9 or 16 <= hour <= 18:
                        large_vehicles = random.randint(15, 45)
                        small_vehicles = random.randint(200, 400)
                    elif 22 <= hour or hour <= 5:
                        large_vehicles = random.randint(2, 8)
                        small_vehicles = random.randint(10, 30)
                    else:
                        large_vehicles = random.randint(5, 20)
                        small_vehicles = random.randint(50, 150)

                    large_vehicles += random.randint(-3, 3)
                    small_vehicles += random.randint(-20, 20)

                    large_vehicles = max(0, large_vehicles)
                    small_vehicles = max(0, small_vehicles)
                    
                    self.save_traffic_data(camera_id, current_datetime, large_vehicles, small_vehicles)
            
            return True
        except Exception as e:
            print(f"Error generating test data: {e}")
            return False
    
    def get_database_info(self):
        try:
            info = {}

            self.cursor.execute("SELECT COUNT(*) FROM record")
            info['total_records'] = self.cursor.fetchone()[0]

            self.cursor.execute("SELECT COUNT(*) FROM camera")
            info['total_cameras'] = self.cursor.fetchone()[0]

            self.cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM record")
            period = self.cursor.fetchone()
            info['first_record'] = period[0]
            info['last_record'] = period[1]

            self.cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            info['database_size'] = self.cursor.fetchone()[0]
            
            return info
        except sqlite3.Error as e:
            print(f"Error getting database info: {e}")
            return {}
    
    def close(self):
        if self.conn:
            self.conn.close()


def get_next_hour_timestamp(current_time):
    next_hour = current_time.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    return next_hour


def format_timestamp(timestamp):
    if isinstance(timestamp, str):
        timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    return timestamp.strftime("%d/%m/%Y %H:%M")


def validate_traffic_data(large_vehicles, small_vehicles):
    if not isinstance(large_vehicles, int) or not isinstance(small_vehicles, int):
        return False, "Data should be an int"
    
    if large_vehicles < 0 or small_vehicles < 0:
        return False, "The number of vehicles can not be negative"

    return True, "Valid data"


def save_tracker_data(db_path, camera_id, timestamp, large_count, small_count):
    db = TrafficDatabase(db_path)
    success = db.save_traffic_data(camera_id, timestamp, large_count, small_count)
    db.close()
    return success
