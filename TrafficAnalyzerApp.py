import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import datetime
from matplotlib.figure import Figure
import seaborn as sns
from tkcalendar import Calendar
import numpy as np
from PIL import Image, ImageTk
import matplotlib.dates as mdates

class TrafficAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator Trafic")
        self.root.geometry("1400x750")
        
        # Setare stil
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configurare culori
        self.bg_color = "#f0f2f5"
        self.accent_color = "#2563eb"
        self.secondary_color = "#1e40af"
        
        # Mod de clasificare (normal sau binary)
        self.classification_mode = tk.StringVar(value="normal")
        
        # Conexiuni la bazele de date
        self.conn_normal = sqlite3.connect('traffic_normal.db')
        self.conn_binary = sqlite3.connect('traffic_binary.db')
        self.create_tables()
        
        # Obține zilele cu date
        self.dates_with_data = self.get_dates_with_data()
        
        # Creare interfață
        self.create_gui()
        
    def create_tables(self):
        """Creează tabela dacă nu există pentru ambele moduri"""
        # Tabel normal
        cursor_normal = self.conn_normal.cursor()
        cursor_normal.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ora DATETIME,
                numar_masini INTEGER,
                numar_autoutilitare INTEGER,
                numar_camioane INTEGER,
                numar_autobuze INTEGER
            )
        ''')
        self.conn_normal.commit()
        
        # Tabel binary
        cursor_binary = self.conn_binary.cursor()
        cursor_binary.execute('''
            CREATE TABLE IF NOT EXISTS traffic_data_binary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ora DATETIME,
                numar_vehicule_mari INTEGER,
                numar_vehicule_mici INTEGER
            )
        ''')
        self.conn_binary.commit()
    
    def get_dates_with_data(self):
        """Obține lista de zile care au date înregistrate în ambele baze de date"""
        dates = set()
        
        # Obține date din baza normală
        cursor_normal = self.conn_normal.cursor()
        cursor_normal.execute('''
            SELECT DISTINCT date(ora) 
            FROM traffic_data 
            ORDER BY date(ora)
        ''')
        normal_dates = [row[0] for row in cursor_normal.fetchall()]
        dates.update(normal_dates)
        
        # Obține date din baza binară
        cursor_binary = self.conn_binary.cursor()
        cursor_binary.execute('''
            SELECT DISTINCT date(ora) 
            FROM traffic_data_binary 
            ORDER BY date(ora)
        ''')
        binary_dates = [row[0] for row in cursor_binary.fetchall()]
        dates.update(binary_dates)
        
        return sorted(list(dates))
        
    def create_gui(self):
        """Creează interfața grafică îmbunătățită"""
        self.root.configure(bg=self.bg_color)
        
        # Configurare stiluri personalizate
        self.style.configure('Custom.TFrame', background=self.bg_color)
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), background=self.bg_color)
        self.style.configure('Subtitle.TLabel', font=('Helvetica', 12), background=self.bg_color)
        self.style.configure('Custom.TButton', padding=10, font=('Helvetica', 10))
        self.style.configure('Custom.TLabelframe', background=self.bg_color)
        self.style.configure('Custom.TLabelframe.Label', font=('Helvetica', 12, 'bold'), background=self.bg_color)
        
        # Header
        header_frame = ttk.Frame(self.root, style='Custom.TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        
        ttk.Label(header_frame, text="🚗 Analizator Trafic", style='Title.TLabel').grid(row=0, column=0, sticky="w")
        ttk.Label(header_frame, text="Monitorizare și analiză trafic", style='Subtitle.TLabel').grid(row=1, column=0, sticky="w")
        
        # Selector pentru modul de clasificare
        mode_frame = ttk.Frame(header_frame)
        mode_frame.grid(row=0, column=1, padx=20, sticky="e")
        
        ttk.Label(mode_frame, text="Mod clasificare:", font=('Helvetica', 10)).pack(side="left", padx=5)
        normal_radio = ttk.Radiobutton(mode_frame, text="Normal (4 clase)", variable=self.classification_mode, value="normal", command=self.on_mode_change)
        normal_radio.pack(side="left", padx=5)
        binary_radio = ttk.Radiobutton(mode_frame, text="Binar (2 clase)", variable=self.classification_mode, value="binary", command=self.on_mode_change)
        binary_radio.pack(side="left", padx=5)
        
        # Main container
        main_container = ttk.Frame(self.root, style='Custom.TFrame')
        main_container.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        
        # Left panel (calendar și controale)
        left_panel = ttk.Frame(main_container, style='Custom.TFrame')
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Calendar frame
        calendar_frame = ttk.LabelFrame(left_panel, text="Selectează Data", padding="10", style='Custom.TLabelframe')
        calendar_frame.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        
        # Calendar widget
        self.calendar = Calendar(
            calendar_frame, 
            selectmode='day',
            year=datetime.datetime.now().year,
            month=datetime.datetime.now().month,
            day=datetime.datetime.now().day,
            date_pattern='yyyy-mm-dd',
            background=self.secondary_color,
            foreground='white',
            selectbackground=self.accent_color,
            normalbackground='white',
            normalforeground='black',
            weekendbackground='white',
            weekendforeground='black',
            othermonthforeground='gray',
            othermonthbackground='white',
            othermonthweforeground='gray',
            othermonthwebackground='white'
        )
        self.calendar.grid(row=0, column=0, pady=5, padx=5)
        
        # Marchează zilele cu date
        self.update_calendar_marks()
        
        # Bind calendar selection event
        self.calendar.bind("<<CalendarSelected>>", self.on_date_select)
        
        # Control panel
        control_frame = ttk.LabelFrame(left_panel, text="Controale", padding="10", style='Custom.TLabelframe')
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        
        # Tip vizualizare
        ttk.Label(control_frame, text="Tip vizualizare:", font=('Helvetica', 10)).grid(row=0, column=0, sticky="w", pady=5)
        self.view_type = ttk.Combobox(control_frame, values=[
            "📊 Vizualizare orară",
            "📈 Vizualizare zilnică", 
            "📅 Vizualizare lunară",
            "🥧 Distribuție procentuală",
            "📉 Tendință temporală",
            "🔥 Comparație categorii"
        ], state="readonly", width=25)
        self.view_type.set("📊 Vizualizare orară")
        self.view_type.grid(row=1, column=0, pady=5, sticky="ew")
        
        # Butoane
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        generate_btn = ttk.Button(button_frame, text="Generează Grafic", command=self.generate_visualization, style='Custom.TButton')
        generate_btn.grid(row=0, column=0, padx=5)
        
        test_data_btn = ttk.Button(button_frame, text="Adaugă Date Test", command=self.add_test_data, style='Custom.TButton')
        test_data_btn.grid(row=0, column=1, padx=5)
        
        # Right panel (grafice și statistici)
        right_panel = ttk.Frame(main_container, style='Custom.TFrame')
        right_panel.grid(row=0, column=1, sticky="nsew")
        
        # Graph frame
        self.graph_frame = ttk.LabelFrame(right_panel, text="Grafice", padding="10", style='Custom.TLabelframe')
        self.graph_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        
        # Stats frame with scrollbar
        stats_container = ttk.Frame(right_panel)
        stats_container.grid(row=1, column=0, sticky="nsew")
        
        # Create a canvas inside stats_container
        stats_canvas = tk.Canvas(stats_container, bg=self.bg_color)
        stats_canvas.pack(side="left", fill="both", expand=True)
        
        # Add scrollbar to stats_container
        stats_scrollbar = ttk.Scrollbar(stats_container, orient="vertical", command=stats_canvas.yview)
        stats_scrollbar.pack(side="right", fill="y")
        
        # Configure canvas
        stats_canvas.configure(yscrollcommand=stats_scrollbar.set)
        
        # Create stats frame inside canvas
        self.stats_frame = ttk.LabelFrame(stats_canvas, text="Statistici", padding="10", style='Custom.TLabelframe')
        stats_canvas_window = stats_canvas.create_window((0, 0), window=self.stats_frame, anchor="nw")
        
        # Bind configuration events to update scroll region
        def configure_stats_scroll(event):
            stats_canvas.configure(scrollregion=stats_canvas.bbox("all"))
            stats_canvas.itemconfig(stats_canvas_window, width=event.width)
        
        stats_canvas.bind('<Configure>', configure_stats_scroll)
        self.stats_frame.bind('<Configure>', lambda e: stats_canvas.configure(scrollregion=stats_canvas.bbox("all")))
        
        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            stats_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        stats_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(0, weight=1)
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=3)
        right_panel.rowconfigure(1, weight=1)
        
        # Data selectată
        self.selected_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    def on_mode_change(self):
        """Handler pentru schimbarea modului de clasificare"""
        self.dates_with_data = self.get_dates_with_data()
        self.update_calendar_marks()
        
    def update_calendar_marks(self):
        """Marchează zilele care au date în calendar"""
        for date_str in self.dates_with_data:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                self.calendar.calevent_create(date_obj, "Date disponibile", "highlight")
            except ValueError:
                continue
        
        # Configurează stilul pentru zilele cu date
        self.calendar.tag_config('highlight', background='#90EE90', foreground='black')
    
    def on_date_select(self, event):
        """Handler pentru selectarea datei din calendar"""
        self.selected_date = self.calendar.get_date()
        
    def add_test_data(self):
        """Adaugă date de test în baza de date"""
        mode = self.classification_mode.get()
        conn = self.conn_normal if mode == "normal" else self.conn_binary
        cursor = conn.cursor()
        
        # Generare date pentru ultimele 7 zile
        import random
        base_date = datetime.datetime.now() - datetime.timedelta(days=7)
        
        for day in range(7):
            current_date = base_date + datetime.timedelta(days=day)
            for hour in range(24):
                current_datetime = current_date.replace(hour=hour, minute=0, second=0)
                
                if mode == "normal":
                    # Generare date realiste (mai mult trafic în orele de vârf)
                    if 7 <= hour <= 9 or 16 <= hour <= 18:  # Ore de vârf
                        masini = random.randint(200, 400)
                        autoutilitare = random.randint(30, 60)
                        camioane = random.randint(10, 30)
                        autobuze = random.randint(5, 15)
                    elif 22 <= hour or hour <= 5:  # Noapte
                        masini = random.randint(20, 50)
                        autoutilitare = random.randint(5, 15)
                        camioane = random.randint(2, 8)
                        autobuze = random.randint(1, 5)
                    else:  # Ore normale
                        masini = random.randint(80, 200)
                        autoutilitare = random.randint(15, 40)
                        camioane = random.randint(5, 20)
                        autobuze = random.randint(3, 10)
                    
                    cursor.execute('''
                        INSERT INTO traffic_data (ora, numar_masini, numar_autoutilitare, numar_camioane, numar_autobuze)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (current_datetime, masini, autoutilitare, camioane, autobuze))
                else:  # binary mode
                    # Generare date pentru binary
                    if 7 <= hour <= 9 or 16 <= hour <= 18:  # Ore de vârf
                        vehicule_mari = random.randint(15, 45)
                        vehicule_mici = random.randint(230, 460)
                    elif 22 <= hour or hour <= 5:  # Noapte
                        vehicule_mari = random.randint(3, 13)
                        vehicule_mici = random.randint(25, 65)
                    else:  # Ore normale
                        vehicule_mari = random.randint(8, 28)
                        vehicule_mici = random.randint(95, 240)
                    
                    cursor.execute('''
                        INSERT INTO traffic_data_binary (ora, numar_vehicule_mari, numar_vehicule_mici)
                        VALUES (?, ?, ?)
                    ''', (current_datetime, vehicule_mari, vehicule_mici))
        
        conn.commit()
        self.dates_with_data = self.get_dates_with_data()
        self.update_calendar_marks()
        messagebox.showinfo("Succes", "Date de test adăugate cu succes!")
        
    def get_data(self, date=None, view_type="zilnic"):
        """Obține date din baza de date"""
        mode = self.classification_mode.get()
        conn = self.conn_normal if mode == "normal" else self.conn_binary
        cursor = conn.cursor()
        
        if view_type == "zilnic":
            if mode == "normal":
                query = '''
                    SELECT strftime('%H', ora) as hour, 
                           SUM(numar_masini) as masini,
                           SUM(numar_autoutilitare) as autoutilitare,
                           SUM(numar_camioane) as camioane,
                           SUM(numar_autobuze) as autobuze
                    FROM traffic_data
                    WHERE date(ora) = ?
                    GROUP BY strftime('%H', ora)
                    ORDER BY hour
                '''
            else:  # binary
                query = '''
                    SELECT strftime('%H', ora) as hour, 
                           SUM(numar_vehicule_mari) as vehicule_mari,
                           SUM(numar_vehicule_mici) as vehicule_mici
                    FROM traffic_data_binary
                    WHERE date(ora) = ?
                    GROUP BY strftime('%H', ora)
                    ORDER BY hour
                '''
            cursor.execute(query, (date,))
        
        elif view_type == "lunar":
            if mode == "normal":
                query = '''
                    SELECT strftime('%H', ora) as hour, 
                           AVG(numar_masini) as masini,
                           AVG(numar_autoutilitare) as autoutilitare,
                           AVG(numar_camioane) as camioane,
                           AVG(numar_autobuze) as autobuze
                    FROM traffic_data
                    WHERE strftime('%Y-%m', ora) = ?
                    GROUP BY strftime('%H', ora)
                    ORDER BY hour
                '''
            else:  # binary
                query = '''
                    SELECT strftime('%H', ora) as hour, 
                           AVG(numar_vehicule_mari) as vehicule_mari,
                           AVG(numar_vehicule_mici) as vehicule_mici
                    FROM traffic_data_binary
                    WHERE strftime('%Y-%m', ora) = ?
                    GROUP BY strftime('%H', ora)
                    ORDER BY hour
                '''
            cursor.execute(query, (date[:7],))
        
        else:  # toate datele
            if mode == "normal":
                query = '''
                    SELECT * FROM traffic_data
                    ORDER BY ora DESC
                    LIMIT 1000
                '''
            else:  # binary
                query = '''
                    SELECT * FROM traffic_data_binary
                    ORDER BY ora DESC
                    LIMIT 1000
                '''
            cursor.execute(query)
        
        return cursor.fetchall()
        
    def generate_visualization(self):
        """Generează vizualizarea selectată"""
        # Curățare frame-uri
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
            
        view_type = self.view_type.get()
        date = self.selected_date
        
        # Map view types without emoji for processing
        view_type_map = {
            "📊 Vizualizare orară": "Vizualizare orară",
            "📈 Vizualizare zilnică": "Vizualizare zilnică",
            "📅 Vizualizare lunară": "Vizualizare lunară",
            "🥧 Distribuție procentuală": "Distribuție procentuală",
            "📉 Tendință temporală": "Tendință temporală",
            "🔥 Comparație categorii": "Comparație categorii"
        }
        
        clean_view_type = view_type_map.get(view_type, view_type)
        
        if clean_view_type == "Vizualizare orară":
            self.generate_hourly_view(date)
        elif clean_view_type == "Vizualizare zilnică":
            self.generate_daily_view(date)
        elif clean_view_type == "Vizualizare lunară":
            self.generate_monthly_view(date)
        elif clean_view_type == "Distribuție procentuală":
            self.generate_percentage_distribution(date)
        elif clean_view_type == "Tendință temporală":
            self.generate_temporal_trend()
        elif clean_view_type == "Comparație categorii":
            self.generate_category_comparison(date)
            
    def generate_hourly_view(self, date):
        """Generează vizualizare orară"""
        data = self.get_data(date, "zilnic")
        mode = self.classification_mode.get()
        
        if not data:
            messagebox.showwarning("Avertisment", "Nu există date pentru data selectată!")
            return
        
        hours = [row[0] for row in data]
        
        if mode == "normal":
            masini = [row[1] for row in data]
            autoutilitare = [row[2] for row in data]
            camioane = [row[3] for row in data]
            autobuze = [row[4] for row in data]
            
            # Creare grafic cu stil îmbunătățit
            fig = Figure(figsize=(10, 6), facecolor=self.bg_color)
            ax = fig.add_subplot(111)
            ax.set_facecolor(self.bg_color)
            
            width = 0.2
            x = range(len(hours))
            
            # Culori moderne
            colors = ['#2563eb', '#10b981', '#f59e0b', '#ef4444']
            
            bars1 = ax.bar([i - width*1.5 for i in x], masini, width, label='Mașini', color=colors[0], alpha=0.8)
            bars2 = ax.bar([i - width*0.5 for i in x], autoutilitare, width, label='Autoutilitare', color=colors[1], alpha=0.8)
            bars3 = ax.bar([i + width*0.5 for i in x], camioane, width, label='Camioane', color=colors[2], alpha=0.8)
            bars4 = ax.bar([i + width*1.5 for i in x], autobuze, width, label='Autobuze', color=colors[3], alpha=0.8)
            
            # Adaugă valori deasupra barelor
            for bars in [bars1, bars2, bars3, bars4]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{int(height)}', ha='center', va='bottom', fontsize=8)
        else:  # binary mode
            vehicule_mari = [row[1] for row in data]
            vehicule_mici = [row[2] for row in data]
            
            # Creare grafic cu stil îmbunătățit
            fig = Figure(figsize=(10, 6), facecolor=self.bg_color)
            ax = fig.add_subplot(111)
            ax.set_facecolor(self.bg_color)
            
            width = 0.35
            x = range(len(hours))
            
            # Culori moderne
            colors = ['#ef4444', '#10b981']  # Roșu pentru mari, verde pentru mici
            
            bars1 = ax.bar([i - width/2 for i in x], vehicule_mari, width, label='Vehicule Mari', color=colors[0], alpha=0.8)
            bars2 = ax.bar([i + width/2 for i in x], vehicule_mici, width, label='Vehicule Mici', color=colors[1], alpha=0.8)
            
            # Adaugă valori deasupra barelor
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                               f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Ora', fontsize=12, fontweight='bold')
        ax.set_ylabel('Număr vehicule', fontsize=12, fontweight='bold')
        ax.set_title(f'Trafic orar - {date} ({mode.capitalize()})', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(hours)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Stil modern
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Afișare grafic
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistici cu stil îmbunătățit
        if mode == "normal":
            total = sum(masini) + sum(autoutilitare) + sum(camioane) + sum(autobuze)
            values = [sum(masini), sum(autoutilitare), sum(camioane), sum(autobuze)]
            labels = ['Mașini', 'Autoutilitare', 'Camioane', 'Autobuze']
            icons = ['🚘', '🚐', '🚛', '🚌']
        else:
            total = sum(vehicule_mari) + sum(vehicule_mici)
            values = [sum(vehicule_mari), sum(vehicule_mici)]
            labels = ['Vehicule Mari', 'Vehicule Mici']
            icons = ['🚛', '🚗']
        
        # Frame pentru statistici principale
        main_stats = ttk.Frame(self.stats_frame)
        main_stats.pack(fill='x', pady=10)
        
        # Statistici principale cu iconițe
        ttk.Label(main_stats, text=f"🚗 Total vehicule: {total}", font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=5)
        
        # Frame pentru statistici detaliate
        detail_stats = ttk.Frame(self.stats_frame)
        detail_stats.pack(fill='x', pady=5)
        
        for i, (icon, label, value) in enumerate(zip(icons, labels, values)):
            percent = (value / total) * 100 if total > 0 else 0
            ttk.Label(detail_stats, text=f"{icon} {label}: {value} ({percent:.1f}%)", 
                     font=('Helvetica', 10)).pack(anchor='w', pady=2)
        
        # Ora de vârf
        if mode == "normal":
            peak_hour = hours[masini.index(max(masini))]
        else:
            peak_hour = hours[vehicule_mici.index(max(vehicule_mici))]
        ttk.Label(self.stats_frame, text=f"⏰ Ora de vârf: {peak_hour}:00", 
                 font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=10)
        
    def generate_daily_view(self, date):
        """Generează vizualizare zilnică (ultimele 7 zile)"""
        mode = self.classification_mode.get()
        conn = self.conn_normal if mode == "normal" else self.conn_binary
        cursor = conn.cursor()
        
        end_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        start_date = end_date - datetime.timedelta(days=6)
        
        if mode == "normal":
            query = '''
                SELECT date(ora) as day,
                       SUM(numar_masini) as masini,
                       SUM(numar_autoutilitare) as autoutilitare,
                       SUM(numar_camioane) as camioane,
                       SUM(numar_autobuze) as autobuze
                FROM traffic_data
                WHERE date(ora) BETWEEN ? AND ?
                GROUP BY date(ora)
                ORDER BY day
            '''
        else:  # binary
            query = '''
                SELECT date(ora) as day,
                       SUM(numar_vehicule_mari) as vehicule_mari,
                       SUM(numar_vehicule_mici) as vehicule_mici
                FROM traffic_data_binary
                WHERE date(ora) BETWEEN ? AND ?
                GROUP BY date(ora)
                ORDER BY day
            '''
        
        cursor.execute(query, (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        data = cursor.fetchall()
        
        if not data:
            messagebox.showwarning("Avertisment", "Nu există date pentru perioada selectată!")
            return
        
        days = [row[0] for row in data]
        
        if mode == "normal":
            masini = [row[1] for row in data]
            autoutilitare = [row[2] for row in data]
            camioane = [row[3] for row in data]
            autobuze = [row[4] for row in data]
            
            # Creare grafic linie modern
            fig = Figure(figsize=(10, 6), facecolor=self.bg_color)
            ax = fig.add_subplot(111)
            ax.set_facecolor(self.bg_color)
            
            # Culori și stiluri
            colors = ['#2563eb', '#10b981', '#f59e0b', '#ef4444']
            markers = ['o', 's', '^', 'd']
            
            # Plotare linii cu umbre
            for i, (data_series, label, color, marker) in enumerate(zip(
                [masini, autoutilitare, camioane, autobuze],
                ['Mașini', 'Autoutilitare', 'Camioane', 'Autobuze'],
                colors, markers
            )):
                ax.plot(days, data_series, marker=marker, label=label, 
                       color=color, linewidth=2.5, markersize=8, 
                       alpha=0.9, markeredgecolor='white', markeredgewidth=1.5)
                
                # Adaugă umbre
                ax.fill_between(days, data_series, alpha=0.1, color=color)
        else:  # binary mode
            vehicule_mari = [row[1] for row in data]
            vehicule_mici = [row[2] for row in data]
            
            # Creare grafic linie modern
            fig = Figure(figsize=(10, 6), facecolor=self.bg_color)
            ax = fig.add_subplot(111)
            ax.set_facecolor(self.bg_color)
            
            # Culori și stiluri
            colors = ['#ef4444', '#10b981']  # Roșu pentru mari, verde pentru mici
            markers = ['o', 's']
            
            # Plotare linii cu umbre
            for i, (data_series, label, color, marker) in enumerate(zip(
                [vehicule_mari, vehicule_mici],
                ['Vehicule Mari', 'Vehicule Mici'],
                colors, markers
            )):
                ax.plot(days, data_series, marker=marker, label=label, 
                       color=color, linewidth=2.5, markersize=8, 
                       alpha=0.9, markeredgecolor='white', markeredgewidth=1.5)
                
                # Adaugă umbre
                ax.fill_between(days, data_series, alpha=0.1, color=color)
        
        ax.set_xlabel('Data', fontsize=12, fontweight='bold')
        ax.set_ylabel('Număr vehicule', fontsize=12, fontweight='bold')
        ax.set_title(f'Trafic zilnic (ultimele 7 zile) - {mode.capitalize()}', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Formatare date pe axa X
        ax.set_xticks(range(len(days)))
        ax.set_xticklabels([d.split('-')[2] + '/' + d.split('-')[1] for d in days], 
                          rotation=45, ha='right')
        
        # Stil modern
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Afișare grafic
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistici îmbunătățite
        if mode == "normal":
            total_vehicles = [sum(x) for x in zip(masini, autoutilitare, camioane, autobuze)]
        else:
            total_vehicles = [sum(x) for x in zip(vehicule_mari, vehicule_mici)]
            
        avg_daily = sum(total_vehicles) / len(total_vehicles)
        
        # Panel statistici principal
        ttk.Label(self.stats_frame, text="📊 Sumar săptămânal", 
                 font=('Helvetica', 14, 'bold')).pack(anchor='w', pady=10)
        
        ttk.Label(self.stats_frame, text=f"Medie zilnică: {avg_daily:.0f} vehicule", 
                 font=('Helvetica', 12)).pack(anchor='w', pady=5)
        
        # Ziua cu cel mai mult trafic
        max_idx = total_vehicles.index(max(total_vehicles))
        max_day = days[max_idx]
        max_day_formatted = datetime.datetime.strptime(max_day, "%Y-%m-%d").strftime("%d/%m/%Y")
        ttk.Label(self.stats_frame, text=f"📈 Zi maximă: {max_day_formatted} ({total_vehicles[max_idx]} vehicule)", 
                 font=('Helvetica', 12)).pack(anchor='w', pady=5)
        
        # Ziua cu cel mai puțin trafic
        min_idx = total_vehicles.index(min(total_vehicles))
        min_day = days[min_idx]
        min_day_formatted = datetime.datetime.strptime(min_day, "%Y-%m-%d").strftime("%d/%m/%Y")
        ttk.Label(self.stats_frame, text=f"📉 Zi minimă: {min_day_formatted} ({total_vehicles[min_idx]} vehicule)", 
                 font=('Helvetica', 12)).pack(anchor='w', pady=5)
        
    def generate_monthly_view(self, date):
        """Generează vizualizare lunară (medie orară)"""
        month = date[:7]  # YYYY-MM
        data = self.get_data(date, "lunar")
        mode = self.classification_mode.get()
        
        if not data:
            messagebox.showwarning("Avertisment", "Nu există date pentru luna selectată!")
            return
        
        hours = [row[0] for row in data]
        
        if mode == "normal":
            masini = [row[1] for row in data]
            autoutilitare = [row[2] for row in data]
            camioane = [row[3] for row in data]
            autobuze = [row[4] for row in data]
            
            # Creare grafic stivă
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            ax.stackplot(hours, masini, autoutilitare, camioane, autobuze,
                        labels=['Mașini', 'Autoutilitare', 'Camioane', 'Autobuze'],
                        alpha=0.8)
        else:  # binary mode
            vehicule_mari = [row[1] for row in data]
            vehicule_mici = [row[2] for row in data]
            
            # Creare grafic stivă
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            ax.stackplot(hours, vehicule_mari, vehicule_mici,
                        labels=['Vehicule Mari', 'Vehicule Mici'],
                        colors=['#ef4444', '#10b981'],
                        alpha=0.8)
        
        ax.set_xlabel('Ora')
        ax.set_ylabel('Număr mediu vehicule')
        ax.set_title(f'Medie orară pentru luna {month} - {mode.capitalize()}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistici lunare
        if mode == "normal":
            total_avg = sum(masini) + sum(autoutilitare) + sum(camioane) + sum(autobuze)
        else:
            total_avg = sum(vehicule_mari) + sum(vehicule_mici)
            
        ttk.Label(self.stats_frame, text=f"Total mediu lunar: {total_avg:.0f}").pack(anchor='w', pady=5)
        ttk.Label(self.stats_frame, text=f"Medie orară: {total_avg/24:.0f}").pack(anchor='w', pady=5)
        
    def generate_percentage_distribution(self, date):
        """Generează distribuție procentuală"""
        data = self.get_data(date, "zilnic")
        mode = self.classification_mode.get()
        
        if not data:
            messagebox.showwarning("Avertisment", "Nu există date pentru data selectată!")
            return
        
        if mode == "normal":
            total_masini = sum(row[1] for row in data)
            total_autoutilitare = sum(row[2] for row in data)
            total_camioane = sum(row[3] for row in data)
            total_autobuze = sum(row[4] for row in data)
            
            total = total_masini + total_autoutilitare + total_camioane + total_autobuze
            
            sizes = [total_masini, total_autoutilitare, total_camioane, total_autobuze]
            labels = ['Mașini', 'Autoutilitare', 'Camioane', 'Autobuze']
            colors = ['#2563eb', '#10b981', '#f59e0b', '#ef4444']
            explode = (0.05, 0, 0, 0)  # Evidențiază mașinile
            icons = ['🚘', '🚐', '🚛', '🚌']
        else:  # binary mode
            total_mari = sum(row[1] for row in data)
            total_mici = sum(row[2] for row in data)
            
            total = total_mari + total_mici
            
            sizes = [total_mari, total_mici]
            labels = ['Vehicule Mari', 'Vehicule Mici']
            colors = ['#ef4444', '#10b981']
            explode = (0.05, 0)  # Evidențiază vehiculele mari
            icons = ['🚛', '🚗']
        
        # Creare grafic plăcintă modern
        fig = Figure(figsize=(10, 6), facecolor=self.bg_color)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.bg_color)
        
        # Creare pie chart cu stil modern
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors, 
                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total)})',
                                         shadow=True, startangle=90, 
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # Stil pentru procente
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.axis('equal')
        ax.set_title(f'Distribuție procentuală - {date} ({mode.capitalize()})', fontsize=16, fontweight='bold', pad=20)
        
        # Adaugă legendă
        ax.legend(wedges, [f'{l} ({s:,})' for l, s in zip(labels, sizes)], 
                 title="Categorii", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # Afișare grafic
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistici detaliate cu stil îmbunătățit
        ttk.Label(self.stats_frame, text="📊 Analiză distribuție", 
                 font=('Helvetica', 14, 'bold')).pack(anchor='w', pady=10)
        
        ttk.Label(self.stats_frame, text=f"Total vehicule: {total:,}", 
                 font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=5)
        
        # Frame pentru fiecare categorie
        categories_frame = ttk.Frame(self.stats_frame)
        categories_frame.pack(fill='x', pady=10)
        
        for i, (icon, label, value, color) in enumerate(zip(icons, labels, sizes, colors)):
            percent = (value / total) * 100 if total > 0 else 0
            category_frame = ttk.Frame(categories_frame)
            category_frame.pack(fill='x', pady=5)
            
            # Creare bară de progres
            progress_frame = ttk.Frame(category_frame)
            progress_frame.pack(fill='x', pady=2)
            
            ttk.Label(category_frame, text=f"{icon} {label}: {value:,} ({percent:.1f}%)", 
                     font=('Helvetica', 11)).pack(anchor='w')
            
            # Bară de progres colorată
            style = ttk.Style()
            style_name = f"Custom{i}.Horizontal.TProgressbar"
            style.configure(style_name, foreground=color, background=color)
            
            progress = ttk.Progressbar(progress_frame, style=style_name, 
                                     length=200, mode='determinate', 
                                     maximum=100, value=percent)
            progress.pack(fill='x', pady=2)
        
        # Categoria dominantă
        dominant_idx = sizes.index(max(sizes))
        ttk.Label(self.stats_frame, 
                 text=f"📌 Categoria dominantă: {labels[dominant_idx]}", 
                 font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=10)
        
    def generate_temporal_trend(self):
        """Generează tendință temporală (toate datele)"""
        mode = self.classification_mode.get()
        conn = self.conn_normal if mode == "normal" else self.conn_binary
        cursor = conn.cursor()
        
        if mode == "normal":
            query = '''
                SELECT datetime(ora) as timestamp,
                       numar_masini + numar_autoutilitare + numar_camioane + numar_autobuze as total
                FROM traffic_data
                ORDER BY ora
            '''
        else:  # binary
            query = '''
                SELECT datetime(ora) as timestamp,
                       numar_vehicule_mari + numar_vehicule_mici as total
                FROM traffic_data_binary
                ORDER BY ora
            '''
        
        cursor.execute(query)
        data = cursor.fetchall()
        
        if not data:
            messagebox.showwarning("Avertisment", "Nu există date în baza de date!")
            return
        
        timestamps = [datetime.datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S") for row in data]
        totals = [row[1] for row in data]
        
        # Creare grafic cu tendință
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(timestamps, totals, alpha=0.5, color='blue')
        
        # Adăugare linie de tendință
        z = np.polyfit(range(len(timestamps)), totals, 1)
        p = np.poly1d(z)
        ax.plot(timestamps, p(range(len(timestamps))), "r--", linewidth=2, label='Tendință')
        
        ax.set_xlabel('Timp')
        ax.set_ylabel('Total vehicule')
        ax.set_title(f'Tendință temporală a traficului - {mode.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Statistici tendință
        if z[0] > 0:
            trend = "Creștere"
        elif z[0] < 0:
            trend = "Descreștere"
        else:
            trend = "Stabil"
            
        ttk.Label(self.stats_frame, text=f"Tendință generală: {trend}").pack(anchor='w', pady=5)
        ttk.Label(self.stats_frame, text=f"Rata de schimbare: {z[0]:.2f} vehicule/oră").pack(anchor='w', pady=5)
        ttk.Label(self.stats_frame, text=f"Total înregistrări: {len(data)}").pack(anchor='w', pady=5)
        
    def generate_category_comparison(self, date):
        """Generează comparație între categorii"""
        data = self.get_data(date, "zilnic")
        mode = self.classification_mode.get()
        
        if not data:
            messagebox.showwarning("Avertisment", "Nu există date pentru data selectată!")
            return
        
        hours = [row[0] for row in data]
        
        if mode == "normal":
            categories = {
                'Mașini': [row[1] for row in data],
                'Autoutilitare': [row[2] for row in data],
                'Camioane': [row[3] for row in data],
                'Autobuze': [row[4] for row in data]
            }
            icons = {'Mașini': '🚘', 'Autoutilitare': '🚐', 'Camioane': '🚛', 'Autobuze': '🚌'}
        else:  # binary
            categories = {
                'Vehicule Mari': [row[1] for row in data],
                'Vehicule Mici': [row[2] for row in data]
            }
            icons = {'Vehicule Mari': '🚛', 'Vehicule Mici': '🚗'}
        
        # Creare heatmap modern
        fig = Figure(figsize=(10, 6), facecolor=self.bg_color)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.bg_color)
        
        data_matrix = [categories[cat] for cat in categories]
        
        # Heatmap cu colormap modern
        im = ax.imshow(data_matrix, aspect='auto', cmap='RdYlBu_r', 
                      interpolation='nearest', alpha=0.9)
        
        # Adaugă valorile în celule
        for i in range(len(categories)):
            for j in range(len(hours)):
                text = ax.text(j, i, str(data_matrix[i][j]), 
                             ha="center", va="center", color="black", 
                             fontsize=9, fontweight='bold')
        
        ax.set_xticks(range(len(hours)))
        ax.set_xticklabels(hours, rotation=45, ha='right')
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(categories.keys())
        
        # Adăugare colorbar modern
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Număr vehicule', fontsize=12)
        
        ax.set_xlabel('Ora', fontsize=12, fontweight='bold')
        ax.set_title(f'Comparație categorii - {date} ({mode.capitalize()})', fontsize=16, fontweight='bold', pad=20)
        
        # Stil modern
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Afișare grafic
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistici comparative îmbunătățite
        ttk.Label(self.stats_frame, text="📊 Analiză comparativă", 
                 font=('Helvetica', 14, 'bold')).pack(anchor='w', pady=10)
        
        # Frame pentru fiecare categorie
        for category, values in categories.items():
            category_frame = ttk.Frame(self.stats_frame)
            category_frame.pack(fill='x', pady=8)
            
            max_val = max(values)
            max_hour = hours[values.index(max_val)]
            min_val = min(values)
            min_hour = hours[values.index(min_val)]
            avg_val = sum(values) / len(values)
            
            ttk.Label(category_frame, text=f"{icons[category]} {category}", 
                     font=('Helvetica', 12, 'bold')).pack(anchor='w')
            
            stats_text = f"Max: {max_val} la ora {max_hour} | Min: {min_val} la ora {min_hour} | Medie: {avg_val:.1f}"
            ttk.Label(category_frame, text=stats_text, 
                     font=('Helvetica', 10)).pack(anchor='w', padx=20)
            
            # Mini grafic pentru tendință
            mini_fig = Figure(figsize=(3, 1), facecolor=self.bg_color)
            mini_ax = mini_fig.add_subplot(111)
            mini_ax.plot(hours, values, color='#2563eb', linewidth=2)
            mini_ax.set_xticks([])
            mini_ax.set_yticks([])
            mini_ax.set_facecolor(self.bg_color)
            for spine in mini_ax.spines.values():
                spine.set_visible(False)
            
            mini_canvas = FigureCanvasTkAgg(mini_fig, category_frame)
            mini_canvas.draw()
            mini_canvas.get_tk_widget().pack(fill='x', padx=20, pady=2)
        
        # Analiza generală
        total_all = sum(sum(values) for values in categories.values())
        peak_category = max(categories.items(), key=lambda x: sum(x[1]))[0]
        
        ttk.Label(self.stats_frame, text=f"📌 Total general: {total_all:,} vehicule", 
                 font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=10)
        ttk.Label(self.stats_frame, text=f"🏆 Categoria dominantă: {peak_category}", 
                 font=('Helvetica', 12)).pack(anchor='w', pady=5)
        
    def __del__(self):
        """Închide conexiunea la baza de date la închiderea aplicației"""
        if hasattr(self, 'conn_normal'):
            self.conn_normal.close()
        if hasattr(self, 'conn_binary'):
            self.conn_binary.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficAnalyzerApp(root)
    root.mainloop()