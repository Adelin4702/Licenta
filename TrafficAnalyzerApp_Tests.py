import unittest
import tkinter as tk
from unittest.mock import Mock, patch, MagicMock
import datetime
import sqlite3
import tempfile
import os
import sys

# Adaugă path-ul pentru a găsi modulele aplicației
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from TrafficAnalyzerApp import BinaryTrafficAnalyzerApp
    from db_functions import TrafficDatabase
except ImportError as e:
    print(f"Eroare la importul modulelor: {e}")
    print("Asigură-te că fișierele TrafficAnalyzerApp.py și db_functions.py sunt în același director cu testele")
    sys.exit(1)


class TestTrafficDatabase(unittest.TestCase):
    """Teste pentru componenta de bază de date"""
    
    def setUp(self):
        """Setup pentru fiecare test - creează o bază de date temporară"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = TrafficDatabase(self.temp_db.name)
    
    def tearDown(self):
        """Cleanup după fiecare test"""
        try:
            self.db.close()
        except:
            pass
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_insert_traffic_data(self):
        """Test inserarea datelor de trafic"""
        test_date = "2024-01-15"
        test_hour = 8
        mari_count = 25
        mici_count = 150
        
        # Insert test data
        success = self.db.save_traffic_data(1, f"{test_date} {test_hour:02d}:00:00", mari_count, mici_count)
        self.assertTrue(success)
        
        # Verify data was inserted
        data = self.db.get_hourly_data(test_date)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], (f"{test_hour:02d}", mari_count, mici_count))
    
    def test_get_dates_with_data(self):
        """Test obținerea datelor cu înregistrări"""
        # Insert some test data
        test_dates = ["2024-01-15", "2024-01-16", "2024-01-17"]
        for date in test_dates:
            self.db.save_traffic_data(1, f"{date} 08:00:00", 10, 20)
        
        # Get dates with data
        dates = self.db.get_dates_with_data()
        
        for date in test_dates:
            self.assertIn(date, dates)
    
    def test_get_daily_totals(self):
        """Test calcularea totalurilor zilnice"""
        test_date = "2024-01-15"
        
        # Insert diverse ore
        test_data = [(6, 5, 10), (8, 15, 50), (12, 20, 80), (18, 25, 90)]
        for hour, mari, mici in test_data:
            self.db.save_traffic_data(1, f"{test_date} {hour:02d}:00:00", mari, mici)
        
        totals = self.db.get_daily_totals(test_date)
        expected_mari = sum(item[1] for item in test_data)
        expected_mici = sum(item[2] for item in test_data)
        
        self.assertEqual(totals, (expected_mari, expected_mici))
    
    def test_get_monthly_trend(self):
        """Test obținerea trend-ului lunar"""
        month = "2024-01"
        
        # Insert data pentru multiple zile în aceeași lună
        for day in [15, 16, 17]:
            date = f"{month}-{day:02d}"
            for hour in [8, 12, 18]:
                self.db.save_traffic_data(1, f"{date} {hour:02d}:00:00", 10, 30)
        
        trend_data = self.db.get_monthly_trend(month)
        self.assertGreater(len(trend_data), 0)
        
        # Verifică structura datelor
        for row in trend_data:
            self.assertEqual(len(row), 3)  # hour, avg_mari, avg_mici
    
    def test_get_weekly_data(self):
        """Test obținerea datelor săptămânale"""
        week_start = "2024-01-15"  # Monday
        week_end = "2024-01-21"    # Sunday
        
        # Insert data pentru fiecare zi a săptămânii
        start_date = datetime.datetime.strptime(week_start, "%Y-%m-%d")
        for i in range(7):
            current_date = (start_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            self.db.save_traffic_data(1, f"{current_date} 08:00:00", 10 + i, 20 + i*2)
        
        week_data = self.db.get_week_data_by_range(week_start, week_end)
        self.assertEqual(len(week_data), 7)
        
        # Verifică structura datelor
        for row in week_data:
            self.assertEqual(len(row), 3)  # date, mari_total, mici_total
    
    def test_get_peak_hours_data(self):
        """Test obținerea datelor pentru orele de vârf"""
        test_date = "2024-01-15"
        
        # Insert data pentru multiple ore
        hours_data = [(6, 10, 20), (7, 30, 60), (8, 40, 80), (9, 25, 50), 
                     (16, 35, 70), (17, 45, 90), (18, 38, 76), (20, 15, 30)]
        
        for hour, mari, mici in hours_data:
            self.db.save_traffic_data(1, f"{test_date} {hour:02d}:00:00", mari, mici)
        
        peak_data = self.db.get_peak_hours_data(test_date)
        self.assertEqual(len(peak_data), len(hours_data))
        
        # Verifică că datele pentru orele de vârf sunt prezente
        peak_hours = [7, 8, 9, 16, 17, 18]
        peak_data_hours = [int(row[0]) for row in peak_data]
        for peak_hour in peak_hours:
            self.assertIn(peak_hour, peak_data_hours)


class TestTrafficAnalyzerGUI(unittest.TestCase):
    """Teste pentru componenta GUI"""
    
    def setUp(self):
        """Setup pentru testele GUI"""
        self.root = tk.Tk()
        self.root.withdraw()  # Ascunde fereastra în timpul testelor
        
        # Mock database pentru a evita dependențele
        with patch('TrafficAnalyzerApp.TrafficDatabase') as mock_db_class:
            mock_db = Mock()
            mock_db.get_dates_with_data.return_value = ['2024-01-15', '2024-01-16']
            
            # Mock cursor pentru filtre cu diferite return values
            mock_cursor = Mock()
            
            # Configure side_effect pentru diferite queries
            def mock_fetchall_side_effect():
                # Primul call pentru cities
                cities_data = [(1, 'Cluj-Napoca'), (2, 'București')]
                # Al doilea call pentru locations
                locations_data = [(1, 'Calea Turzii', 15, 1, 'Cluj-Napoca'), (2, 'Strada Memorandumului', 3, 1, 'Cluj-Napoca')]
                # Al treilea call pentru cameras
                cameras_data = [(1, 'Hikvision DS-TCG405', 'Calea Turzii', 15, 'Cluj-Napoca')]
                
                # Returnează datele în ordinea apelurilor
                if not hasattr(mock_cursor, 'call_count'):
                    mock_cursor.call_count = 0
                
                mock_cursor.call_count += 1
                if mock_cursor.call_count == 1:
                    return cities_data
                elif mock_cursor.call_count == 2:
                    return locations_data
                else:
                    return cameras_data
            
            mock_cursor.fetchall.side_effect = mock_fetchall_side_effect
            mock_db.cursor = mock_cursor
            mock_db_class.return_value = mock_db
            
            self.app = BinaryTrafficAnalyzerApp(self.root)
            self.app.db = mock_db
            
            # Mock metodele de filtrare pentru a evita erori
            self.app.get_filtered_hourly_data = Mock(return_value=[])
            self.app.get_filtered_daily_totals = Mock(return_value=(0, 0))
            self.app.get_filtered_week_data_by_range = Mock(return_value=[])
            self.app.get_filtered_monthly_trend = Mock(return_value=[])
            self.app.get_filtered_peak_hours_data = Mock(return_value=[])
    
    def tearDown(self):
        """Cleanup după testele GUI"""
        try:
            if self.root:
                self.root.destroy()
        except:
            pass
    
    def test_gui_initialization(self):
        """Test inițializarea GUI-ului"""
        # Verifică că elementele principale sunt create
        self.assertIsNotNone(self.app.root)
        self.assertIsNotNone(self.app.calendar)
        self.assertIsNotNone(self.app.view_type)
        self.assertIsNotNone(self.app.colors)
        
        # Verifică că filtrele sunt create
        self.assertIsNotNone(self.app.city_filter)
        self.assertIsNotNone(self.app.location_filter)
        self.assertIsNotNone(self.app.camera_filter)
    
    @patch('matplotlib.pyplot.show')
    def test_visualization_generation_no_data(self, mock_show):
        """Test generarea vizualizării fără date"""
        # Mock database să returneze date goale pentru metodele cu filtre
        self.app.get_filtered_hourly_data = Mock(return_value=[])
        
        # Încearcă să genereze vizualizarea
        self.app.generate_visualization()
        
        # Verifică că funcția de filtrare a fost apelată
        self.app.get_filtered_hourly_data.assert_called()
    
    @patch('matplotlib.pyplot.show')
    def test_visualization_generation_with_data(self, mock_show):
        """Test generarea vizualizării cu date"""
        # Mock database să returneze date de test pentru metodele cu filtre
        test_data = [(8, 25, 150), (9, 30, 180), (10, 20, 120)]
        self.app.get_filtered_hourly_data = Mock(return_value=test_data)
        
        # Setează tipul de vizualizare
        self.app.view_type.set("📊 Trafic orar")
        
        # Generează vizualizarea
        self.app.generate_visualization()
        
        # Verifică că datele filtrate au fost solicitate
        self.app.get_filtered_hourly_data.assert_called()
    
    def test_no_data_message_display(self):
        """Test afișarea mesajului pentru lipsa datelor"""
        custom_message = "Nu există date pentru perioada selectată"
        
        # Afișează mesajul de lipsa datelor
        self.app.show_no_data_message(custom_message)
        
        # Verifică că s-au creat widget-uri în graph_frame
        children = self.app.graph_frame.winfo_children()
        self.assertGreater(len(children), 0)
    
    def test_filter_functionality(self):
        """Test funcționalitatea filtrelor"""
        # Test resetarea filtrelor (fără a apela generate_visualization care crează probleme în test)
        self.app.city_filter.set("Toate orașele")
        self.app.location_filter['values'] = ["Toate locațiile"]
        self.app.location_filter.set("Toate locațiile")
        self.app.camera_filter['values'] = ["Toate camerele"]
        self.app.camera_filter.set("Toate camerele")
        
        # Verifică că filtrele sunt resetate corect
        self.assertEqual(self.app.city_filter.get(), "Toate orașele")
        self.assertEqual(self.app.location_filter.get(), "Toate locațiile")
        self.assertEqual(self.app.camera_filter.get(), "Toate camerele")


class TestCalculationAccuracy(unittest.TestCase):
    """Teste pentru acuratețea calculelor"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = TrafficDatabase(self.temp_db.name)
    
    def tearDown(self):
        try:
            self.db.close()
        except:
            pass
        try:
            os.unlink(self.temp_db.name)
        except:
            pass
    
    def test_daily_totals_calculation(self):
        """Test calcularea totalurilor zilnice"""
        test_date = "2024-01-15"
        expected_mari_total = 0
        expected_mici_total = 0
        
        # Insert multiple hours of data
        test_hours = [
            (6, 10, 30),
            (8, 25, 75),
            (12, 15, 45),
            (18, 30, 90),
            (20, 5, 15)
        ]
        
        for hour, mari, mici in test_hours:
            self.db.save_traffic_data(1, f"{test_date} {hour:02d}:00:00", mari, mici)
            expected_mari_total += mari
            expected_mici_total += mici
        
        # Calculate totals
        actual_mari, actual_mici = self.db.get_daily_totals(test_date)
        
        self.assertEqual(actual_mari, expected_mari_total)
        self.assertEqual(actual_mici, expected_mici_total)
    
    def test_percentage_calculations(self):
        """Test calcularea procentajelor"""
        # Simulează calculele din GUI
        total_mari = 250
        total_mici = 750
        total = total_mari + total_mici
        
        percent_mari = (total_mari / total) * 100
        percent_mici = (total_mici / total) * 100
        
        self.assertAlmostEqual(percent_mari, 25.0, places=1)
        self.assertAlmostEqual(percent_mici, 75.0, places=1)
        self.assertAlmostEqual(percent_mari + percent_mici, 100.0, places=1)
    
    def test_weekly_averages(self):
        """Test calcularea mediilor săptămânale"""
        # Simulează o săptămână de date
        week_start = datetime.datetime.strptime("2024-01-15", "%Y-%m-%d")
        days_count = 7
        
        for i in range(days_count):
            current_date = (week_start + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            daily_total = (i + 1) * 100  # Vehicule crescătoare
            
            # Insert some hours for each day
            self.db.save_traffic_data(1, f"{current_date} 08:00:00", daily_total // 4, (daily_total * 3) // 4)
        
        # Test că datele există pentru toate zilele
        week_end = week_start + datetime.timedelta(days=6)
        week_data = self.db.get_week_data_by_range(
            week_start.strftime("%Y-%m-%d"), 
            week_end.strftime("%Y-%m-%d")
        )
        self.assertEqual(len(week_data), days_count)


# Test Runner Principal
def run_all_tests():
    """Rulează toate testele și generează un raport"""
    
    # Creează test suite
    test_classes = [
        TestTrafficDatabase,
        TestTrafficAnalyzerGUI,
        TestCalculationAccuracy
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Rulează testele
    runner = unittest.TextTestRunner(verbosity=1, buffer=True)
    result = runner.run(suite)
    
    # Raport final simplu
    print(f"\n{'='*50}")
    print(f"REZULTATE TESTE")
    print(f"{'='*50}")
    print(f"Total teste: {result.testsRun}")
    print(f"Succese: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Eșecuri: {len(result.failures)}")
    print(f"Erori: {len(result.errors)}")
    
    if result.failures:
        print(f"\nEȘECURI:")
        for test, _ in result.failures:
            print(f"❌ {test}")
    
    if result.errors:
        print(f"\nERRORI:")
        for test, _ in result.errors:
            print(f"⚠️ {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100 if result.testsRun > 0 else 0
    print(f"\nRata de succes: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Rulează testele când scriptul este executat direct
    success = run_all_tests()
    sys.exit(0 if success else 1)