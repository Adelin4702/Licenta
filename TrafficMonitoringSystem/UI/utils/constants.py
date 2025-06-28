"""
Constants and configuration for Traffic Analyzer App
"""

# Application configuration
APP_TITLE = "🚗 Analizator Trafic Inteligent"
APP_SUBTITLE = "Analiză avansată pentru vehicule mari și mici"
MIN_WINDOW_SIZE = (1200, 800)
DATABASE_NAME = '../traffic_binary.db'

# UI Configuration
HEADER_HEIGHT = 120
CALENDAR_PANEL_WIDTH = 320
STATS_PANEL_WIDTH = 250
CARD_HEADER_HEIGHT = 35

# Visualization options
VISUALIZATION_OPTIONS = [
    "📊 Trafic orar",
    "🥧 Distribuție procentuală", 
    "🔥 Comparație ore de vârf",
    "📈 Trafic săptămânal",
    "📅 Tendință lunară",
    "🗓️ Evoluție anuală"  # NEW YEARLY VIEW
]

# Peak hours definition
PEAK_HOURS = [7, 8, 9, 16, 17, 18]

# Date formats
DATE_FORMAT_INPUT = "%Y-%m-%d"
DATE_FORMAT_DISPLAY = "%d/%m/%Y"
DATE_FORMAT_SHORT = "%d/%m"

# Font configuration
FONT_FAMILY = 'Segoe UI'
FONT_SIZES = {
    'title': 28,
    'subtitle': 14,
    'header': 16,
    'card_header': 12,
    'normal': 10,
    'small': 9,
    'large': 14
}

# Chart configuration
CHART_CONFIG = {
    'bar_width': 0.35,
    'line_width': 4,
    'marker_size': 10,
    'marker_edge_width': 2,
    'alpha': 0.9,
    'alpha_fill': 0.2
}

# Messages
MESSAGES = {
    'no_data': "Nu există date pentru data selectată!",
    'no_weekly_data': "Nu există date pentru săptămâna",
    'no_monthly_data': "Nu există date pentru luna selectată!",
    'no_peak_data': "Nu există date pentru analiza orelor de vârf!",
    'test_data_success': "Date de test adăugate cu succes!",
    'test_data_error': "Eroare la adăugarea datelor de test!",
    'init_error': "Nu s-a putut inițializa aplicația",
    'check_dependencies': "Verificați dacă toate dependențele sunt instalate."
}