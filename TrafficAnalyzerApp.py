import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
from matplotlib.figure import Figure
from tkcalendar import Calendar
import numpy as np
from db_functions import TrafficDatabase

class BinaryTrafficAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizator Trafic - Vehicule Mari/Mici")
        # self.root.geometry("1300x700")
        self.root.state('zoomed')
        # Setare stil
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configurare culori
        self.bg_color = "#f0f2f5"
        self.accent_color = "#2563eb"
        
        # Conexiune la baza de date
        self.db = TrafficDatabase('traffic_binary.db')
        
        # Obține zilele cu date
        self.dates_with_data = self.db.get_dates_with_data()
        
        # Creare interfață
        self.create_gui()
        
    def create_gui(self):
        """Creează interfața grafică simplificată"""
        self.root.configure(bg=self.bg_color)
        
        # Configurare stiluri
        self.style.configure('Custom.TFrame', background=self.bg_color)
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), background=self.bg_color)
        self.style.configure('Custom.TButton', padding=10, font=('Helvetica', 10))
        self.style.configure('Custom.TLabelframe', background=self.bg_color)
        self.style.configure('Custom.TLabelframe.Label', font=('Helvetica', 12, 'bold'), background=self.bg_color)
        
        # Header
        header_frame = ttk.Frame(self.root, style='Custom.TFrame')
        header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        
        ttk.Label(header_frame, text="🚗 Analizator Trafic", style='Title.TLabel').grid(row=0, column=0, sticky="w")
        
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
            background=self.accent_color,
            foreground='white',
            selectbackground='#10b981',
        )
        self.calendar.grid(row=0, column=0, pady=5, padx=5)

        # Marchează zilele cu date
        self.update_calendar_marks()
        self.calendar.bind("<<CalendarSelected>>", self.on_date_select)
        
        # Control panel
        control_frame = ttk.LabelFrame(left_panel, text="Controale", padding="10", style='Custom.TLabelframe')
        control_frame.grid(row=1, column=0, sticky="ew", pady=(0, 20))
        
        # Tip vizualizare
        ttk.Label(control_frame, text="Tip vizualizare:", font=('Helvetica', 10)).grid(row=0, column=0, sticky="w", pady=5)
        self.view_type = ttk.Combobox(control_frame, values=[
            "📊 Trafic orar",
            "🥧 Distribuție procentuală",
            "🔥 Comparație ore de vârf",
            "📈 Trafic săptămânal", 
            "📅 Tendință lunară"
        ], state="readonly", width=25)
        self.view_type.set("📊 Trafic orar")
        self.view_type.grid(row=1, column=0, pady=5, sticky="ew")
        self.view_type.bind("<<ComboboxSelected>>", self.generate_visualization)
        
        # Butoane
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, pady=10)
        
        test_data_btn = ttk.Button(button_frame, text="🔧 Generare Date Test", 
                                  command=self.add_test_data, style='Custom.TButton')
        test_data_btn.grid(row=0, column=1, padx=5)
        
        # Right panel (grafice și statistici)
        right_panel = ttk.Frame(main_container, style='Custom.TFrame')
        right_panel.grid(row=0, column=1, sticky="nsew")
        
        # Graph frame
        self.graph_frame = ttk.LabelFrame(right_panel, text="Grafice", padding="10", style='Custom.TLabelframe')
        self.graph_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 0))
        
        # Stats frame with scrollbar
        stats_container = ttk.Frame(right_panel)
        stats_container.grid(row=1, column=0, sticky="nsew")

        # Create a canvas inside stats_container
        stats_canvas = tk.Canvas(stats_container, bg=self.bg_color)
        stats_scrollbar = ttk.Scrollbar(stats_container, orient="vertical", command=stats_canvas.yview)

        # Pack scrollbar and canvas correctly
        stats_scrollbar.pack(side="right", fill="y")
        stats_canvas.pack(side="left", fill="both", expand=True)

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
        
    def update_calendar_marks(self):
        """Marchează zilele care au date în calendar"""
        for date_str in self.dates_with_data:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                self.calendar.calevent_create(date_obj, "Date disponibile", "highlight")
            except ValueError:
                continue
        
        self.calendar.tag_config('highlight', background='#90EE90', foreground='black')
    
    def on_date_select(self, event):
        """Handler pentru selectarea datei din calendar"""
        self.selected_date = self.calendar.get_date()
        self.generate_visualization()
        
    def add_test_data(self):
        """Adaugă date de test în baza de date"""
        success = self.db.generate_test_data(days=7)
        if success:
            self.dates_with_data = self.db.get_dates_with_data()
            self.update_calendar_marks()
            messagebox.showinfo("Succes", "Date de test adăugate cu succes!")
        else:
            messagebox.showerror("Eroare", "Eroare la adăugarea datelor de test!")
            
    def generate_visualization(self, event=None):
        """Generează vizualizarea selectată"""
        # Curățare frame-uri
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
            
        view_type = self.view_type.get()
        date = self.selected_date
        
        if "Trafic orar" in view_type:
            self.generate_hourly_view(date)
        elif "Trafic săptămânal" in view_type:
            self.generate_weekly_view(date)
        elif "Tendință lunară" in view_type:
            self.generate_monthly_trend(date)
        elif "Distribuție procentuală" in view_type:
            self.generate_percentage_distribution(date)
        elif "Comparație ore de vârf" in view_type:
            self.generate_peak_hours_comparison(date)
            
    def generate_hourly_view(self, date):
        """Generează vizualizare orară pentru vehicule mari vs mici"""
        data = self.db.get_hourly_data(date)
        
        if not data:
            messagebox.showwarning("Avertisment", "Nu există date pentru data selectată!")
            return
        
        hours = [row[0] for row in data]
        vehicule_mari = [row[1] for row in data]
        vehicule_mici = [row[2] for row in data]
        
        # Creare grafic modern
        fig = self.create_figure(max_height=400) 
        fig.tight_layout(pad=3.0)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.bg_color)
        
        width = 0.35
        x = range(len(hours))
        
        # Culori distinctive
        color_mari = '#ef4444'    # Roșu pentru vehicule mari
        color_mici = '#10b981'    # Verde pentru vehicule mici
        
        bars1 = ax.bar([i - width/2 for i in x], vehicule_mari, width, 
                      label='Vehicule Mari', color=color_mari, alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], vehicule_mici, width, 
                      label='Vehicule Mici', color=color_mici, alpha=0.8)
        
        # Adaugă valori deasupra barelor
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(vehicule_mari + vehicule_mici) * 0.01,
                           f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Ora', fontsize=12, fontweight='bold')
        ax.set_ylabel('Număr vehicule', fontsize=12, fontweight='bold')
        ax.set_title(f'Trafic orar - {date[8:10]}/{date[5:7]}/{date[:4]}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}" for h in hours])
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
        
        # Statistici
        total_mari = sum(vehicule_mari)
        total_mici = sum(vehicule_mici)
        total = total_mari + total_mici
        
        ttk.Label(self.stats_frame, text=f"📊 Statistici pentru {date}", 
                 font=('Helvetica', 14, 'bold')).pack(anchor='w', pady=10)
        
        ttk.Label(self.stats_frame, text=f"🚛 Total vehicule mari: {total_mari:,}", 
                 font=('Helvetica', 12)).pack(anchor='w', pady=5)
        ttk.Label(self.stats_frame, text=f"🚗 Total vehicule mici: {total_mici:,}", 
                 font=('Helvetica', 12)).pack(anchor='w', pady=5)
        ttk.Label(self.stats_frame, text=f"🚦 Total general: {total:,}", 
                 font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=5)
        
        # Procentaje
        if total > 0:
            percent_mari = (total_mari / total) * 100
            percent_mici = (total_mici / total) * 100
            ttk.Label(self.stats_frame, text=f"📈 Vehicule mari: {percent_mari:.1f}%", 
                     font=('Helvetica', 11)).pack(anchor='w', pady=2)
            ttk.Label(self.stats_frame, text=f"📈 Vehicule mici: {percent_mici:.1f}%", 
                     font=('Helvetica', 11)).pack(anchor='w', pady=2)
        
        # Ora de vârf
        if vehicule_mici:
            peak_hour_idx = vehicule_mici.index(max(vehicule_mici))
            peak_hour = hours[peak_hour_idx]
            ttk.Label(self.stats_frame, text=f"⏰ Ora de vârf: {peak_hour}:00", 
                     font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=10)
        
    def generate_weekly_view(self, date):
        """Generează vizualizare pentru săptămâna în care se află data selectată"""
        # Calculează prima zi a săptămânii (luni)
        selected_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        days_since_monday = selected_date.weekday()  # 0=Luni, 6=Duminică
        week_start = selected_date - datetime.timedelta(days=days_since_monday)
        week_end = week_start + datetime.timedelta(days=6)
        
        data = self.db.get_week_data_by_range(week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d"))
        
        if not data:
            messagebox.showwarning("Avertisment", f"Nu există date pentru săptămâna {week_start.strftime('%d/%m')} - {week_end.strftime('%d/%m/%Y')}!")
            return
        
        # Creează o mapare pentru zilele săptămânii
        week_days = []
        week_data_map = {row[0]: (row[1], row[2]) for row in data}
        
        vehicule_mari = []
        vehicule_mici = []
        day_labels = []
        
        for i in range(7):
            current_day = week_start + datetime.timedelta(days=i)
            day_str = current_day.strftime("%Y-%m-%d")
            day_label = current_day.strftime("%a %d/%m")  # Ex: "Lun 15/01"
            
            if day_str in week_data_map:
                mari, mici = week_data_map[day_str]
                vehicule_mari.append(mari)
                vehicule_mici.append(mici)
            else:
                vehicule_mari.append(0)
                vehicule_mici.append(0)
            
            day_labels.append(day_label)
        
        # Creare grafic linie
        fig = self.create_figure(max_height=400) 
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.bg_color)
        
        ax.plot(day_labels, vehicule_mari, marker='o', label='Vehicule Mari', 
               color='#ef4444', linewidth=3, markersize=8, markeredgecolor='white', markeredgewidth=2)
        ax.plot(day_labels, vehicule_mici, marker='s', label='Vehicule Mici', 
               color='#10b981', linewidth=3, markersize=8, markeredgecolor='white', markeredgewidth=2)
        
        # Adaugă umbre
        ax.fill_between(day_labels, vehicule_mari, alpha=0.1, color='#ef4444')
        ax.fill_between(day_labels, vehicule_mici, alpha=0.1, color='#10b981')
        
        ax.set_xlabel('Ziua săptămânii', fontsize=12, fontweight='bold')
        ax.set_ylabel('Număr vehicule', fontsize=12, fontweight='bold')
        ax.set_title(f'Trafic săptămânal ({week_start.strftime("%d/%m")} - {week_end.strftime("%d/%m/%Y")})', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Formatare date pe axa X
        ax.set_xticks(range(len(day_labels)))
        ax.set_xticklabels(day_labels, rotation=45, ha='right')
        
        # Stil modern
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistici săptămânale îmbunătățite
        total_mari_week = sum(vehicule_mari)
        total_mici_week = sum(vehicule_mici)
        total_week = total_mari_week + total_mici_week
        
        # Găsește ziua cu cel mai mult și cel mai puțin trafic
        total_daily = [m + s for m, s in zip(vehicule_mari, vehicule_mici)]
        max_day_idx = total_daily.index(max(total_daily))
        min_day_idx = total_daily.index(min(total_daily))
        
        ttk.Label(self.stats_frame, text=f"📅 Săptămâna {week_start.strftime('%d/%m')} - {week_end.strftime('%d/%m/%Y')}", 
                 font=('Helvetica', 14, 'bold')).pack(anchor='w', pady=10)
        
        ttk.Label(self.stats_frame, text=f"📊 Total săptămânal: {total_week:,} vehicule", 
                 font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=5)
        
        ttk.Label(self.stats_frame, text=f"🚛 Vehicule mari: {total_mari_week:,} ({(total_mari_week/total_week*100) if total_week > 0 else 0:.1f}%)", 
                 font=('Helvetica', 11)).pack(anchor='w', pady=3)
        ttk.Label(self.stats_frame, text=f"🚗 Vehicule mici: {total_mici_week:,} ({(total_mici_week/total_week*100) if total_week > 0 else 0:.1f}%)", 
                 font=('Helvetica', 11)).pack(anchor='w', pady=3)
        
        if total_week > 0:
            avg_daily = total_week / 7
            ttk.Label(self.stats_frame, text=f"📈 Medie zilnică: {avg_daily:.0f} vehicule", 
                     font=('Helvetica', 12)).pack(anchor='w', pady=5)
            
            ttk.Label(self.stats_frame, text=f"🏆 Ziua cu cel mai mult trafic: {day_labels[max_day_idx]} ({total_daily[max_day_idx]:,})", 
                     font=('Helvetica', 11)).pack(anchor='w', pady=3)
            ttk.Label(self.stats_frame, text=f"📉 Ziua cu cel mai puțin trafic: {day_labels[min_day_idx]} ({total_daily[min_day_idx]:,})", 
                     font=('Helvetica', 11)).pack(anchor='w', pady=3)
            
            # Analiza zilelor lucrătoare vs weekend
            weekdays_total = sum(total_daily[0:5])  # Luni-Vineri
            weekend_total = sum(total_daily[5:7])   # Sâmbătă-Duminică
            
            if weekdays_total > 0 and weekend_total > 0:
                ttk.Label(self.stats_frame, text="", font=('Helvetica', 8)).pack(anchor='w', pady=2)  # Separator
                ttk.Label(self.stats_frame, text="📊 Analiză zile lucrătoare vs weekend:", 
                         font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=5)
                ttk.Label(self.stats_frame, text=f"💼 Luni-Vineri: {weekdays_total:,} vehicule ({weekdays_total/5:.0f}/zi)", 
                         font=('Helvetica', 11)).pack(anchor='w', pady=2)
                ttk.Label(self.stats_frame, text=f"🏖️ Sâmbătă-Duminică: {weekend_total:,} vehicule ({weekend_total/2:.0f}/zi)", 
                         font=('Helvetica', 11)).pack(anchor='w', pady=2)
        
    def generate_monthly_trend(self, date):
        """Generează tendința lunară"""
        data = self.db.get_monthly_trend(date[:7])  # YYYY-MM
        
        if not data:
            messagebox.showwarning("Avertisment", "Nu există date pentru luna selectată!")
            return
        
        hours = [row[0] for row in data]
        avg_mari = [row[1] for row in data]
        avg_mici = [row[2] for row in data]
        
        # Creare grafic stivă
        fig = self.create_figure(max_height=400) 
        fig.tight_layout(pad=3.0)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.bg_color)
        
        ax.stackplot(hours, avg_mari, avg_mici,
                    labels=['Vehicule Mari', 'Vehicule Mici'],
                    colors=['#ef4444', '#10b981'],
                    alpha=0.8)
        
        ax.set_xlabel('Ora', fontsize=12, fontweight='bold')
        ax.set_ylabel('Medie vehicule', fontsize=12, fontweight='bold')
        ax.set_title(f'Tendință orară pentru luna {date[5:7]}/{date[:4]}', fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def generate_percentage_distribution(self, date):
        """Generează distribuție procentuală"""
        data = self.db.get_daily_totals(date)
        
        if not data:
            messagebox.showwarning("Avertisment", "Nu există date pentru data selectată!")
            return
        
        total_mari, total_mici = data
        total = total_mari + total_mici
        
        if total == 0:
            messagebox.showwarning("Avertisment", "Nu există vehicule înregistrate pentru data selectată!")
            return
        
        # Creare pie chart
        fig = self.create_figure(max_height=400) 
        fig.tight_layout(pad=3.0)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.bg_color)
        
        sizes = [total_mari, total_mici]
        labels = ['Vehicule Mari', 'Vehicule Mici']
        colors = ['#ef4444', '#10b981']
        explode = (0.05, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total):,})',
                                         shadow=True, startangle=90,
                                         textprops={'fontsize': 12, 'fontweight': 'bold'})
        
        # Stil pentru procente
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.axis('equal')
        ax.set_title(f'Distribuție vehicule - {date[8:10]}/{date[5:7]}/{date[:4]}', fontsize=16, fontweight='bold', pad=20)
        
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistici distribuție
        percent_mari = (total_mari / total) * 100
        percent_mici = (total_mici / total) * 100
        
        ttk.Label(self.stats_frame, text="🥧 Analiză distribuție", 
                 font=('Helvetica', 14, 'bold')).pack(anchor='w', pady=10)
        ttk.Label(self.stats_frame, text=f"📊 Total vehicule: {total:,}", 
                 font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=5)
        ttk.Label(self.stats_frame, text=f"🚛 Vehicule mari: {total_mari:,} ({percent_mari:.1f}%)", 
                 font=('Helvetica', 11)).pack(anchor='w', pady=3)
        ttk.Label(self.stats_frame, text=f"🚗 Vehicule mici: {total_mici:,} ({percent_mici:.1f}%)", 
                 font=('Helvetica', 11)).pack(anchor='w', pady=3)
        
    def generate_peak_hours_comparison(self, date):
        """Generează comparație ore de vârf"""
        peak_data = self.db.get_peak_hours_data(date)
        
        if not peak_data:
            messagebox.showwarning("Avertisment", "Nu există date pentru analiza orelor de vârf!")
            return
        
        # Separează datele în ore de vârf vs ore normale
        peak_hours = [7, 8, 9, 16, 17, 18]  # Ore de vârf
        peak_mari = sum(row[1] for row in peak_data if int(row[0]) in peak_hours)
        peak_mici = sum(row[2] for row in peak_data if int(row[0]) in peak_hours)
        normal_mari = sum(row[1] for row in peak_data if int(row[0]) not in peak_hours)
        normal_mici = sum(row[2] for row in peak_data if int(row[0]) not in peak_hours)
        
        # Creare grafic comparativ
        fig = self.create_figure(max_height=400) 
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.bg_color)
        
        categories = ['Ore de vârf (7-9, 16-18)', 'Ore normale']
        mari_data = [peak_mari, normal_mari]
        mici_data = [peak_mici, normal_mici]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mari_data, width, label='Vehicule Mari', 
                      color='#ef4444', alpha=0.8)
        bars2 = ax.bar(x + width/2, mici_data, width, label='Vehicule Mici', 
                      color='#10b981', alpha=0.8)
        
        # Adaugă valori
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(mari_data + mici_data) * 0.01,
                       f'{int(height):,}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Perioada', fontsize=12, fontweight='bold')
        ax.set_ylabel('Număr vehicule', fontsize=12, fontweight='bold')
        ax.set_title(f'Comparație ore de vârf vs ore normale - {date[8:10]}/{date[5:7]}/{date[:4]}', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Stil modern
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Statistici comparație
        total_peak = peak_mari + peak_mici
        total_normal = normal_mari + normal_mici
        
        ttk.Label(self.stats_frame, text="🔥 Analiza orelor de vârf", 
                 font=('Helvetica', 14, 'bold')).pack(anchor='w', pady=10)
        ttk.Label(self.stats_frame, text=f"📈 Total ore de vârf: {total_peak:,}", 
                 font=('Helvetica', 12)).pack(anchor='w', pady=5)
        ttk.Label(self.stats_frame, text=f"📊 Total ore normale: {total_normal:,}", 
                 font=('Helvetica', 12)).pack(anchor='w', pady=5)
        
        if total_normal > 0:
            factor = total_peak / total_normal
            ttk.Label(self.stats_frame, text=f"⚡ Factor multiplicare: {factor:.1f}x", 
                     font=('Helvetica', 12, 'bold')).pack(anchor='w', pady=5)
        
    def __del__(self):
        """Închide conexiunea la baza de date"""
        if hasattr(self, 'db'):
            self.db.close()

    def create_figure(self, max_height=400):
        """Creează figure cu înălțime maximă și width automat"""
        # Calculează width-ul bazat pe containerul grafic
        container_width = self.graph_frame.winfo_width()
        if container_width <= 1:  # Dacă nu e încă desenat
            container_width = 800  # Default
        
        # Calculează aspect ratio
        width_inches = container_width / 100  # Convertește la inches (aprox)
        height_inches = min(max_height / 100, width_inches * 0.6)  # Max height, aspect 1.67:1
        
        return Figure(figsize=(width_inches, height_inches), facecolor=self.bg_color)

if __name__ == "__main__":
    root = tk.Tk()
    app = BinaryTrafficAnalyzerApp(root)
    root.mainloop()