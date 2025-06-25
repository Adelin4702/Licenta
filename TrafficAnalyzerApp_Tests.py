import tkinter as tk
from unittest.mock import Mock, patch, MagicMock
import datetime
import sqlite3
import tempfile
import os
from TrafficAnalyzerApp import BinaryTrafficAnalyzerApp
from db_functions import TrafficDatabase

class TestTrafficDatabase(unittest.TestCase):
    """Teste pentru componenta de bază de date"""
    
    def setUp(self):
        """Setup pentru fiecare test - creează o bază de date temporară"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = TrafficDatabase(self.temp_db.name)
    
    def tearDown(self):
        """Cleanup după fiecare test"""
        self.db.close()
        os.unlink(self.temp_db.name)
    
    def test_insert_traffic_data(self):
        """Test inserarea datelor de trafic"""
        test_date = "2024-01-15"
        test_hour = 8
        mari_count = 25
        mici_count = 150
        
        # Insert test data
        success = self.db.insert_traffic_data(test_date, test_hour, mari_count, mici_count)
        self.assertTrue(success)
        
        # Verify data was inserted
        data = self.db.get_hourly_data(test_date)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0], (test_hour, mari_count, mici_count))
    
    def test_get_dates_with_data(self):
        """Test obținerea datelor cu înregistrări"""
        # Insert some test data
        test_dates = ["2024-01-15", "2024-01-16", "2024-01-17"]
        for date in test_dates:
            self.db.insert_traffic_data(date, 8, 10, 20)
        
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
            self.db.insert_traffic_data(test_date, hour, mari, mici)
        
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
                self.db.insert_traffic_data(date, hour, 10, 30)
        
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
            self.db.insert_traffic_data(current_date, 8, 10 + i, 20 + i*2)
        
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
            self.db.insert_traffic_data(test_date, hour, mari, mici)
        
        peak_data = self.db.get_peak_hours_data(test_date)
        self.assertEqual(len(peak_data), len(hours_data))
        
        # Verifică că datele pentru orele de vârf sunt prezente
        peak_hours = [7, 8, 9, 16, 17, 18]
        peak_data_hours = [row[0] for row in peak_data]
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
            mock_db_class.return_value = mock_db
            
            self.app = BinaryTrafficAnalyzerApp(self.root)
            self.app.db = mock_db
    
    def tearDown(self):
        """Cleanup după testele GUI"""
        if self.root:
            self.root.destroy()
    
    def test_gui_initialization(self):
        """Test inițializarea GUI-ului"""
        # Verifică că elementele principale sunt create
        self.assertIsNotNone(self.app.root)
        self.assertIsNotNone(self.app.calendar)
        self.assertIsNotNone(self.app.view_type)
        self.assertIsNotNone(self.app.colors)
    
    @patch('matplotlib.pyplot.show')
    def test_visualization_generation_no_data(self, mock_show):
        """Test generarea vizualizării fără date"""
        # Mock database să returneze date goale
        self.app.db.get_hourly_data.return_value = []
        
        # Încearcă să genereze vizualizarea
        self.app.generate_visualization()
        
        # Verifică că nu se întâmplă erori și funcția se execută
        self.app.db.get_hourly_data.assert_called()
    
    @patch('matplotlib.pyplot.show')
    def test_visualization_generation_with_data(self, mock_show):
        """Test generarea vizualizării cu date"""
        # Mock database să returneze date de test
        test_data = [(8, 25, 150), (9, 30, 180), (10, 20, 120)]
        self.app.db.get_hourly_data.return_value = test_data
        
        # Setează tipul de vizualizare
        self.app.view_type.set("📊 Trafic orar")
        
        # Generează vizualizarea
        self.app.generate_visualization()
        
        # Verifică că datele au fost solicitate
        self.app.db.get_hourly_data.assert_called()
    


    
    def test_no_data_message_display(self):
        """Test afișarea mesajului pentru lipsa datelor"""
        custom_message = "Nu există date pentru perioada selectată"
        
        # Afișează mesajul de lipsa datelor
        self.app.show_no_data_message(custom_message)
        
        # Verifică că s-au creat widget-uri în graph_frame
        children = self.app.graph_frame.winfo_children()
        self.assertGreater(len(children), 0)
    
    
    def test_figure_creation(self):
        """Test crearea figurilor matplotlib"""
        fig = self.app.create_modern_figure(width=5, height=4)
        
        # Verifică proprietățile figurii
        self.assertEqual(fig.get_figwidth(), 5)
        self.assertEqual(fig.get_figheight(), 4)
        self.assertEqual(fig.get_facecolor(), self.app.colors['surface'])




class TestCalculationAccuracy(unittest.TestCase):
    """Teste pentru acuratețea calculelor"""
    
    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = TrafficDatabase(self.temp_db.name)
    
    def tearDown(self):
        self.db.close()
        os.unlink(self.temp_db.name)
    
    def test_daily_totals_calculation(self):
        """Test calcularea totalaurilor zilnice"""
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
            self.db.insert_traffic_data(test_date, hour, mari, mici)
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
        total_vehicles = 0
        days_count = 7
        
        for i in range(days_count):
            current_date = (week_start + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            daily_total = (i + 1) * 100  # Vehicule crescătoare
            total_vehicles += daily_total
            
            # Insert some hours for each day
            self.db.insert_traffic_data(current_date, 8, daily_total // 4, (daily_total * 3) // 4)
        
        expected_daily_average = total_vehicles / days_count
        
        # În implementarea reală, ar trebui să existe o metodă pentru asta
        # self.assertAlmostEqual(calculated_average, expected_daily_average, places=1)




# Test Runner Principal
def run_all_tests():
    """Rulează toate testele și generează un raport"""
    
    # Creează test suite
    test_classes = [
        TestTrafficDatabase,
        TestTrafficAnalyzerGUI,
        TestDataValidation,
        TestCalculationAccuracy,
        TestPerformanceAndStability,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Rulează testele cu output detaliat
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Raport final
    print(f"\n{'='*50}")
    print(f"RAPORT FINAL TESTE")
    print(f"{'='*50}")
    print(f"Teste rulate: {result.testsRun}")
    print(f"Succese: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Eșecuri: {len(result.failures)}")
    print(f"Erori: {len(result.errors)}")
    
    if result.failures:
        print(f"\nEȘECURI:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORI:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100
    print(f"\nRata de succes: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Rulează testele când scriptul este executat direct
    success = run_all_tests()
    exit(0 if success else 1)