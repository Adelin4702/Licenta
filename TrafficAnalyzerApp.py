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
        self.root.title("🚗 Analizator Trafic Inteligent")
        self.root.state('zoomed')
        
        # Modern color scheme
        self.colors = {
            'primary': '#1e40af',      # Modern blue
            'secondary': '#64748b',    # Slate gray
            'success': '#059669',      # Emerald green
            'danger': '#dc2626',       # Red
            'warning': '#d97706',      # Amber
            'info': '#0ea5e9',         # Sky blue
            'light': '#f8fafc',        # Very light gray
            'dark': '#1e293b',         # Dark slate
            'background': '#f1f5f9',   # Light blue-gray
            'surface': '#ffffff',      # White
            'accent': '#8b5cf6',       # Purple
            'muted': '#6b7280'         # Gray
        }
        
        # Configure matplotlib style
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'Segoe UI',
            'font.size': 10,
            'axes.labelweight': 'bold',
            'axes.titleweight': 'bold',
            'figure.facecolor': self.colors['surface'],
            'axes.facecolor': self.colors['light'],
            'axes.edgecolor': self.colors['muted'],
            'axes.linewidth': 0.8,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5
        })
        
        # Database connection
        self.db = TrafficDatabase('traffic_binary.db')
        self.dates_with_data = self.db.get_dates_with_data()
        
        # Setup modern styling
        self.setup_styles()
        
        # Create GUI
        self.create_modern_gui()
        
    def setup_styles(self):
        """Configure modern ttk styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure root background
        self.root.configure(bg=self.colors['background'])
        
        # Header styles
        self.style.configure('Header.TFrame', 
                           background=self.colors['surface'],
                           relief='flat',
                           borderwidth=0)
        
        self.style.configure('Title.TLabel', 
                           font=('Segoe UI', 28, 'bold'),
                           background=self.colors['surface'],
                           foreground=self.colors['primary'])
        
        self.style.configure('Subtitle.TLabel', 
                           font=('Segoe UI', 14),
                           background=self.colors['surface'],
                           foreground=self.colors['secondary'])
        
        # Card styles
        self.style.configure('Card.TFrame', 
                           background=self.colors['surface'],
                           relief='solid',
                           borderwidth=1)
        
        self.style.configure('CardHeader.TLabel', 
                           font=('Segoe UI', 16, 'bold'),
                           background=self.colors['surface'],
                           foreground=self.colors['dark'])
        
        # Control styles
        self.style.configure('Modern.TLabelframe', 
                           background=self.colors['surface'],
                           relief='flat',
                           borderwidth=2,
                           labeloutside=False)
        
        self.style.configure('Modern.TLabelframe.Label', 
                           font=('Segoe UI', 14, 'bold'),
                           background=self.colors['surface'],
                           foreground=self.colors['primary'])
        
        # Button styles
        self.style.configure('Primary.TButton',
                           font=('Segoe UI', 11, 'bold'),
                           padding=(20, 12),
                           background=self.colors['primary'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none')
        
        self.style.map('Primary.TButton',
                      background=[('active', '#1d4ed8'),
                                ('pressed', '#1e3a8a')])
        
        self.style.configure('Success.TButton',
                           font=('Segoe UI', 11, 'bold'),
                           padding=(20, 12),
                           background=self.colors['success'],
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none')
        
        # Combobox style
        self.style.configure('Modern.TCombobox',
                           font=('Segoe UI', 11),
                           padding=10,
                           fieldbackground=self.colors['light'],
                           borderwidth=1,
                           relief='solid')
        
        # Stats label styles
        self.style.configure('Stat.TLabel',
                           font=('Segoe UI', 12),
                           background=self.colors['surface'],
                           foreground=self.colors['dark'],
                           padding=(10, 5))
        
        self.style.configure('StatTitle.TLabel',
                           font=('Segoe UI', 14, 'bold'),
                           background=self.colors['surface'],
                           foreground=self.colors['primary'],
                           padding=(10, 8))
        
        self.style.configure('StatValue.TLabel',
                           font=('Segoe UI', 18, 'bold'),
                           background=self.colors['surface'],
                           foreground=self.colors['dark'])
        
    def create_modern_gui(self):
        """Create modern, professional GUI"""
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section
        self.create_header(main_frame)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg=self.colors['background'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left panel (controls)
        left_panel = tk.Frame(content_frame, bg=self.colors['background'], width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # Right panel (visualization)
        right_panel = tk.Frame(content_frame, bg=self.colors['background'])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create panels
        self.create_control_panel(left_panel)
        self.create_visualization_panel(right_panel)
        
        # Initialize
        self.selected_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    def create_header(self, parent):
        """Create modern header with gradient effect"""
        header_frame = tk.Frame(parent, bg=self.colors['surface'], height=120)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Add subtle shadow effect
        shadow_frame = tk.Frame(parent, bg='#e2e8f0', height=2)
        shadow_frame.pack(fill=tk.X)
        
        # Header content
        header_content = tk.Frame(header_frame, bg=self.colors['surface'])
        header_content.pack(expand=True, fill=tk.BOTH, padx=40, pady=20)
        
        # Title and subtitle
        title_label = tk.Label(header_content, 
                              text="🚗 Analizator Trafic Inteligent",
                              font=('Segoe UI', 28, 'bold'),
                              bg=self.colors['surface'],
                              fg=self.colors['primary'])
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(header_content,
                                 text="Analiză avansată pentru vehicule mari și mici",
                                 font=('Segoe UI', 14),
                                 bg=self.colors['surface'],
                                 fg=self.colors['secondary'])
        subtitle_label.pack(anchor='w', pady=(5, 0))
        
        # Status indicator
        status_frame = tk.Frame(header_content, bg=self.colors['surface'])
        status_frame.pack(anchor='w', pady=(10, 0))
        
        status_dot = tk.Label(status_frame, text="●", 
                             font=('Segoe UI', 12),
                             fg=self.colors['success'],
                             bg=self.colors['surface'])
        status_dot.pack(side=tk.LEFT)
        
        status_text = tk.Label(status_frame, text=f"Conectat | {len(self.dates_with_data)} zile cu date",
                              font=('Segoe UI', 11),
                              bg=self.colors['surface'],
                              fg=self.colors['muted'])
        status_text.pack(side=tk.LEFT, padx=(5, 0))
        
    def create_control_panel(self, parent):
        """Create modern control panel"""
        # Calendar card
        self.create_calendar_card(parent)
        
        # Controls card
        self.create_controls_card(parent)
        
        # Quick stats card
        self.create_quick_stats_card(parent)
        
    def create_calendar_card(self, parent):
        """Create calendar card with modern styling"""
        card_frame = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        card_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Card header
        header_frame = tk.Frame(card_frame, bg=self.colors['primary'], height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="📅 Selectează Data",
                               font=('Segoe UI', 14, 'bold'),
                               bg=self.colors['primary'],
                               fg='white')
        header_label.pack(expand=True)
        
        # Calendar content
        calendar_content = tk.Frame(card_frame, bg=self.colors['surface'])
        calendar_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
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
            font=('Segoe UI', 10)
        )
        self.calendar.pack(pady=10)
        
        self.update_calendar_marks()
        self.calendar.bind("<<CalendarSelected>>", self.on_date_select)
        
    def create_controls_card(self, parent):
        """Create controls card"""
        card_frame = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        card_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Card header
        header_frame = tk.Frame(card_frame, bg=self.colors['info'], height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="⚙️ Opțiuni Vizualizare",
                               font=('Segoe UI', 14, 'bold'),
                               bg=self.colors['info'],
                               fg='white')
        header_label.pack(expand=True)
        
        # Controls content
        controls_content = tk.Frame(card_frame, bg=self.colors['surface'])
        controls_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # View type selector with modern styling
        view_label = tk.Label(controls_content, text="Tip vizualizare:",
                             font=('Segoe UI', 12, 'bold'),
                             bg=self.colors['surface'],
                             fg=self.colors['dark'])
        view_label.pack(anchor='w', pady=(0, 8))
        
        self.view_type = ttk.Combobox(controls_content, 
                                     values=[
                                         "📊 Trafic orar",
                                         "🥧 Distribuție procentuală", 
                                         "🔥 Comparație ore de vârf",
                                         "📈 Trafic săptămânal",
                                         "📅 Tendință lunară"
                                     ],
                                     state="readonly",
                                     font=('Segoe UI', 11),
                                     style='Modern.TCombobox')
        self.view_type.set("📊 Trafic orar")
        self.view_type.pack(fill=tk.X, pady=(0, 20))
        self.view_type.bind("<<ComboboxSelected>>", self.generate_visualization)
        
        # Action buttons with modern styling
        button_frame = tk.Frame(controls_content, bg=self.colors['surface'])
        button_frame.pack(fill=tk.X)
        
        generate_btn = tk.Button(button_frame, text="🔄 Actualizează",
                               font=('Segoe UI', 11, 'bold'),
                               bg=self.colors['primary'],
                               fg='white',
                               relief='flat',
                               padx=20, pady=12,
                               cursor='hand2',
                               command=self.generate_visualization)
        generate_btn.pack(fill=tk.X, pady=(0, 10))
        
        test_data_btn = tk.Button(button_frame, text="🔧 Generare Date Test",
                                font=('Segoe UI', 11, 'bold'),
                                bg=self.colors['success'],
                                fg='white',
                                relief='flat',
                                padx=20, pady=12,
                                cursor='hand2',
                                command=self.add_test_data)
        test_data_btn.pack(fill=tk.X)
        
        # Hover effects
        def on_enter(e, btn, color):
            btn.configure(bg=color)
        def on_leave(e, btn, color):
            btn.configure(bg=color)
            
        generate_btn.bind("<Enter>", lambda e: on_enter(e, generate_btn, '#1d4ed8'))
        generate_btn.bind("<Leave>", lambda e: on_leave(e, generate_btn, self.colors['primary']))
        test_data_btn.bind("<Enter>", lambda e: on_enter(e, test_data_btn, '#047857'))
        test_data_btn.bind("<Leave>", lambda e: on_leave(e, test_data_btn, self.colors['success']))
        
    def create_quick_stats_card(self, parent):
        """Create quick stats card"""
        card_frame = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        card_frame.pack(fill=tk.X)
        
        # Card header
        header_frame = tk.Frame(card_frame, bg=self.colors['accent'], height=50)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(header_frame, text="📊 Statistici Rapide",
                               font=('Segoe UI', 14, 'bold'),
                               bg=self.colors['accent'],
                               fg='white')
        header_label.pack(expand=True)
        
        # Stats content
        self.quick_stats_content = tk.Frame(card_frame, bg=self.colors['surface'])
        self.quick_stats_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Default message
        default_label = tk.Label(self.quick_stats_content,
                               text="Selectează o dată pentru\na vedea statisticile",
                               font=('Segoe UI', 11),
                               bg=self.colors['surface'],
                               fg=self.colors['muted'],
                               justify=tk.CENTER)
        default_label.pack(expand=True)
        
    def create_visualization_panel(self, parent):
        """Create visualization panel"""
        # Main visualization card
        viz_card = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        viz_card.pack(fill=tk.BOTH, expand=True)
        
        # Card header
        header_frame = tk.Frame(viz_card, bg=self.colors['dark'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        self.viz_title = tk.Label(header_frame, text="📈 Vizualizare Trafic",
                                 font=('Segoe UI', 16, 'bold'),
                                 bg=self.colors['dark'],
                                 fg='white')
        self.viz_title.pack(expand=True)
        
        # Visualization content area
        viz_content = tk.Frame(viz_card, bg=self.colors['surface'])
        viz_content.pack(fill=tk.BOTH, expand=True)
        
        # Graph area
        self.graph_frame = tk.Frame(viz_content, bg=self.colors['light'])
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Initial placeholder
        placeholder_label = tk.Label(self.graph_frame,
                                    text="🎯\nSelectează o opțiune de vizualizare\npentru a începe analiza",
                                    font=('Segoe UI', 16),
                                    bg=self.colors['light'],
                                    fg=self.colors['muted'],
                                    justify=tk.CENTER)
        placeholder_label.pack(expand=True)
        
        # Detailed stats area (collapsible)
        self.create_stats_panel(viz_card)
        
    def create_stats_panel(self, parent):
        """Create expandable stats panel"""
        # Stats toggle button
        self.stats_visible = tk.BooleanVar(value=True)
        
        toggle_frame = tk.Frame(parent, bg=self.colors['surface'])
        toggle_frame.pack(fill=tk.X, padx=20)
        
        self.toggle_btn = tk.Button(toggle_frame, text="📊 Ascunde Statistici Detaliate",
                                   font=('Segoe UI', 11, 'bold'),
                                   bg=self.colors['secondary'],
                                   fg='white',
                                   relief='flat',
                                   pady=8,
                                   cursor='hand2',
                                   command=self.toggle_stats)
        self.toggle_btn.pack(fill=tk.X)
        
        # Stats content with scrollbar
        self.stats_container = tk.Frame(parent, bg=self.colors['surface'])
        self.stats_container.pack(fill=tk.BOTH, padx=20, pady=(0, 20))
        
        # Create scrollable frame
        canvas = tk.Canvas(self.stats_container, bg=self.colors['light'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.stats_container, orient="vertical", command=canvas.yview)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.stats_frame = tk.Frame(canvas, bg=self.colors['light'])
        canvas_window = canvas.create_window((0, 0), window=self.stats_frame, anchor="nw")
        
        def configure_scroll(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)
            
        canvas.bind('<Configure>', configure_scroll)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
    def toggle_stats(self):
        """Toggle stats panel visibility"""
        if self.stats_visible.get():
            self.stats_container.pack_forget()
            self.toggle_btn.configure(text="📊 Arată Statistici Detaliate")
            self.stats_visible.set(False)
        else:
            self.stats_container.pack(fill=tk.BOTH, padx=20, pady=(0, 20))
            self.toggle_btn.configure(text="📊 Ascunde Statistici Detaliate")
            self.stats_visible.set(True)
            
    def update_calendar_marks(self):
        """Mark days with data in calendar"""
        for date_str in self.dates_with_data:
            try:
                date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                self.calendar.calevent_create(date_obj, "Date disponibile", "highlight")
            except ValueError:
                continue
        
        self.calendar.tag_config('highlight', background='#10b981', foreground='white')
    
    def on_date_select(self, event):
        """Handle date selection"""
        self.selected_date = self.calendar.get_date()
        self.generate_visualization()
        
    def add_test_data(self):
        """Add test data to database"""
        success = self.db.generate_test_data(days=7)
        if success:
            self.dates_with_data = self.db.get_dates_with_data()
            self.update_calendar_marks()
            messagebox.showinfo("✅ Succes", "Date de test adăugate cu succes!")
            # Update status in header
            self.update_header_status()
        else:
            messagebox.showerror("❌ Eroare", "Eroare la adăugarea datelor de test!")
            
    def update_header_status(self):
        """Update header status"""
        # This would update the status indicator in the header
        pass
        
    def generate_visualization(self, event=None):
        """Generate selected visualization with modern styling"""
        # Clear existing content
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        for widget in self.stats_frame.winfo_children():
            widget.destroy()
        for widget in self.quick_stats_content.winfo_children():
            widget.destroy()
            
        view_type = self.view_type.get()
        date = self.selected_date
        
        # Update visualization title
        self.viz_title.configure(text=f"📈 {view_type} - {date}")
        
        if "Trafic orar" in view_type:
            self.generate_modern_hourly_view(date)
        elif "Trafic săptămânal" in view_type:
            self.generate_modern_weekly_view(date)
        elif "Tendință lunară" in view_type:
            self.generate_modern_monthly_trend(date)
        elif "Distribuție procentuală" in view_type:
            self.generate_modern_percentage_distribution(date)
        elif "Comparație ore de vârf" in view_type:
            self.generate_modern_peak_hours_comparison(date)
            
    def create_modern_figure(self, width=3, height=3):
        """Create modern styled figure"""
        fig = Figure(figsize=(width, height), 
                    facecolor=self.colors['surface'],
                    edgecolor='none')
        
        # Add subtle border
        fig.patch.set_linewidth(0)
        
        return fig
        
    def generate_modern_hourly_view(self, date):
        """Generate modern hourly view"""
        data = self.db.get_hourly_data(date)
        
        if not data:
            self.show_no_data_message()
            return
        
        hours = [row[0] for row in data]
        vehicule_mari = [row[1] for row in data]
        vehicule_mici = [row[2] for row in data]
        
        # Create modern figure
        fig = self.create_modern_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        # Modern bar chart with gradients
        width = 0.35
        x = np.arange(len(hours))
        
        # Create bars with modern colors
        bars1 = ax.bar([i - width/2 for i in x], vehicule_mari, width,
                      label='🚛 Vehicule Mari', 
                      color=self.colors['danger'],
                      alpha=0.9,
                      edgecolor='white',
                      linewidth=1)
        
        bars2 = ax.bar([i + width/2 for i in x], vehicule_mici, width,
                      label='🚗 Vehicule Mici',
                      color=self.colors['success'], 
                      alpha=0.9,
                      edgecolor='white',
                      linewidth=1)
        
        # Add value labels with modern styling
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(vehicule_mari + vehicule_mici) * 0.02,
                           f'{int(height)}',
                           ha='center', va='bottom',
                           fontsize=10, fontweight='bold',
                           color=self.colors['dark'])
        
        # Modern styling
        ax.set_xlabel('Ora zilei', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_ylabel('Numărul de vehicule', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_title(f'Analiza traficului orar\n{date}', fontsize=16, fontweight='bold', 
                    color=self.colors['primary'], pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}:00" for h in hours], rotation=45, ha='right')
        
        # Modern legend
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor=self.colors['muted'])
        legend.get_frame().set_facecolor(self.colors['surface'])
        
        # Modern grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['muted'])
        ax.set_axisbelow(True)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        
        # Adjust layout
        fig.tight_layout(pad=2.0)
        
        # Display chart
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate modern stats
        self.generate_modern_hourly_stats(vehicule_mari, vehicule_mici, hours, date)
        
    def generate_modern_hourly_stats(self, vehicule_mari, vehicule_mici, hours, date):
        """Generate modern statistics for hourly view"""
        total_mari = sum(vehicule_mari)
        total_mici = sum(vehicule_mici)
        total = total_mari + total_mici
        
        # Quick stats card
        self.create_quick_stat_cards(total, total_mari, total_mici)
        
        # Detailed stats
        stats_container = tk.Frame(self.stats_frame, bg=self.colors['light'])
        stats_container.pack(fill=tk.X, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(stats_container, 
                              text=f"📊 Analiză Detaliată - {date}",
                              font=('Segoe UI', 16, 'bold'),
                              bg=self.colors['light'],
                              fg=self.colors['primary'])
        title_label.pack(anchor='w', pady=(0, 15))
        
        # Create stat boxes
        stats_grid = tk.Frame(stats_container, bg=self.colors['light'])
        stats_grid.pack(fill=tk.X, pady=10)
        
        # Statistics data
        stats_data = [
            ("🚛 Total Vehicule Mari", f"{total_mari:,}", self.colors['danger']),
            ("🚗 Total Vehicule Mici", f"{total_mici:,}", self.colors['success']),
            ("📈 Procentaj Vehicule Mari", f"{(total_mari/total*100) if total > 0 else 0:.1f}%", self.colors['warning']),
            ("📈 Procentaj Vehicule Mici", f"{(total_mici/total*100) if total > 0 else 0:.1f}%", self.colors['info'])
        ]
        
        for i, (label, value, color) in enumerate(stats_data):
            self.create_stat_box(stats_grid, label, value, color, row=i//2, col=i%2)
        
        # Peak hour analysis
        if vehicule_mici and vehicule_mari:
            peak_hour_mici_idx = vehicule_mici.index(max(vehicule_mici))
            peak_hour_mari_idx = vehicule_mari.index(max(vehicule_mari))
            peak_hour_mici = hours[peak_hour_mici_idx]
            peak_hour_mari = hours[peak_hour_mari_idx]
            
            # Peak hours section
            peak_section = tk.Frame(stats_container, bg=self.colors['surface'], relief='solid', bd=1)
            peak_section.pack(fill=tk.X, pady=(15, 0))
            
            peak_header = tk.Frame(peak_section, bg=self.colors['primary'], height=40)
            peak_header.pack(fill=tk.X)
            peak_header.pack_propagate(False)
            
            peak_title = tk.Label(peak_header, text="⏰ Analiza Orelor de Vârf",
                                 font=('Segoe UI', 14, 'bold'),
                                 bg=self.colors['primary'],
                                 fg='white')
            peak_title.pack(expand=True)
            
            peak_content = tk.Frame(peak_section, bg=self.colors['surface'])
            peak_content.pack(fill=tk.X, padx=15, pady=15)
            
            peak_stats = [
                (f"🚗 Ora de vârf vehicule mici: {peak_hour_mici}:00", f"{max(vehicule_mici)} vehicule"),
                (f"🚛 Ora de vârf vehicule mari: {peak_hour_mari}:00", f"{max(vehicule_mari)} vehicule"),
                (f"📊 Medie orară total", f"{total/len(hours):.1f} vehicule"),
                (f"🎯 Eficiența traficului", f"{(max(vehicule_mici + vehicule_mari)/sum(vehicule_mici + vehicule_mari)*100):.1f}%")
            ]
            
            for stat_text, stat_value in peak_stats:
                stat_row = tk.Frame(peak_content, bg=self.colors['surface'])
                stat_row.pack(fill=tk.X, pady=5)
                
                tk.Label(stat_row, text=stat_text,
                        font=('Segoe UI', 11),
                        bg=self.colors['surface'],
                        fg=self.colors['dark']).pack(side=tk.LEFT)
                
                tk.Label(stat_row, text=stat_value,
                        font=('Segoe UI', 11, 'bold'),
                        bg=self.colors['surface'],
                        fg=self.colors['primary']).pack(side=tk.RIGHT)
    
    def create_quick_stat_cards(self, total, total_mari, total_mici):
        """Create quick stat cards for the sidebar"""
        # Clear existing content
        for widget in self.quick_stats_content.winfo_children():
            widget.destroy()
        
        # Total vehicles card
        total_card = tk.Frame(self.quick_stats_content, bg=self.colors['primary'], relief='solid', bd=1)
        total_card.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(total_card, text="🚦 TOTAL",
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['primary'],
                fg='white').pack(pady=(10, 0))
        
        tk.Label(total_card, text=f"{total:,}",
                font=('Segoe UI', 20, 'bold'),
                bg=self.colors['primary'],
                fg='white').pack()
        
        tk.Label(total_card, text="vehicule",
                font=('Segoe UI', 10),
                bg=self.colors['primary'],
                fg='white').pack(pady=(0, 10))
        
        # Large vehicles card
        mari_card = tk.Frame(self.quick_stats_content, bg=self.colors['danger'], relief='solid', bd=1)
        mari_card.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(mari_card, text="🚛 MARI",
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['danger'],
                fg='white').pack(pady=(10, 0))
        
        tk.Label(mari_card, text=f"{total_mari:,}",
                font=('Segoe UI', 18, 'bold'),
                bg=self.colors['danger'],
                fg='white').pack()
        
        percentage_mari = (total_mari/total*100) if total > 0 else 0
        tk.Label(mari_card, text=f"{percentage_mari:.1f}%",
                font=('Segoe UI', 10),
                bg=self.colors['danger'],
                fg='white').pack(pady=(0, 10))
        
        # Small vehicles card
        mici_card = tk.Frame(self.quick_stats_content, bg=self.colors['success'], relief='solid', bd=1)
        mici_card.pack(fill=tk.X)
        
        tk.Label(mici_card, text="🚗 MICI",
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['success'],
                fg='white').pack(pady=(10, 0))
        
        tk.Label(mici_card, text=f"{total_mici:,}",
                font=('Segoe UI', 18, 'bold'),
                bg=self.colors['success'],
                fg='white').pack()
        
        percentage_mici = (total_mici/total*100) if total > 0 else 0
        tk.Label(mici_card, text=f"{percentage_mici:.1f}%",
                font=('Segoe UI', 10),
                bg=self.colors['success'],
                fg='white').pack(pady=(0, 10))
    
    def create_stat_box(self, parent, label, value, color, row=0, col=0):
        """Create a modern stat box"""
        stat_frame = tk.Frame(parent, bg=color, relief='solid', bd=1, width=180, height=80)
        stat_frame.grid(row=row, column=col, padx=10, pady=5, sticky='ew')
        stat_frame.pack_propagate(False)
        
        tk.Label(stat_frame, text=label,
                font=('Segoe UI', 10, 'bold'),
                bg=color,
                fg='white',
                wraplength=160).pack(pady=(10, 0))
        
        tk.Label(stat_frame, text=value,
                font=('Segoe UI', 14, 'bold'),
                bg=color,
                fg='white').pack(pady=(5, 10))
        
        # Configure column weights
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
    
    def generate_modern_weekly_view(self, date):
        """Generate modern weekly view"""
        selected_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        days_since_monday = selected_date.weekday()
        week_start = selected_date - datetime.timedelta(days=days_since_monday)
        week_end = week_start + datetime.timedelta(days=6)
        
        data = self.db.get_week_data_by_range(week_start.strftime("%Y-%m-%d"), week_end.strftime("%Y-%m-%d"))
        
        if not data:
            self.show_no_data_message(f"Nu există date pentru săptămâna {week_start.strftime('%d/%m')} - {week_end.strftime('%d/%m/%Y')}")
            return
        
        # Prepare data
        week_data_map = {row[0]: (row[1], row[2]) for row in data}
        vehicule_mari = []
        vehicule_mici = []
        day_labels = []
        
        for i in range(7):
            current_day = week_start + datetime.timedelta(days=i)
            day_str = current_day.strftime("%Y-%m-%d")
            day_label = current_day.strftime("%a\n%d/%m")
            
            if day_str in week_data_map:
                mari, mici = week_data_map[day_str]
                vehicule_mari.append(mari)
                vehicule_mici.append(mici)
            else:
                vehicule_mari.append(0)
                vehicule_mici.append(0)
            
            day_labels.append(day_label)
        
        # Create modern line chart
        fig = self.create_modern_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        # Modern line plots with gradients
        x_pos = range(len(day_labels))
        
        line1 = ax.plot(x_pos, vehicule_mari, marker='o', label='🚛 Vehicule Mari',
                       color=self.colors['danger'], linewidth=4, markersize=10,
                       markeredgecolor='white', markeredgewidth=2,
                       markerfacecolor=self.colors['danger'])
        
        line2 = ax.plot(x_pos, vehicule_mici, marker='s', label='🚗 Vehicule Mici',
                       color=self.colors['success'], linewidth=4, markersize=10,
                       markeredgecolor='white', markeredgewidth=2,
                       markerfacecolor=self.colors['success'])
        
        # Add gradient fill
        ax.fill_between(x_pos, vehicule_mari, alpha=0.2, color=self.colors['danger'])
        ax.fill_between(x_pos, vehicule_mici, alpha=0.2, color=self.colors['success'])
        
        # Modern styling
        ax.set_xlabel('Ziua săptămânii', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_ylabel('Numărul de vehicule', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_title(f'Analiza săptămânală a traficului\n{week_start.strftime("%d/%m")} - {week_end.strftime("%d/%m/%Y")}',
                    fontsize=16, fontweight='bold', color=self.colors['primary'], pad=20)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(day_labels, ha='center')
        
        # Modern legend and grid
        legend = ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor=self.colors['muted'])
        legend.get_frame().set_facecolor(self.colors['surface'])
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['muted'])
        ax.set_axisbelow(True)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        
        fig.tight_layout(pad=2.0)
        
        # Display chart
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate weekly stats
        self.generate_modern_weekly_stats(vehicule_mari, vehicule_mici, day_labels, week_start, week_end)
    
    def generate_modern_weekly_stats(self, vehicule_mari, vehicule_mici, day_labels, week_start, week_end):
        """Generate modern weekly statistics"""
        total_mari_week = sum(vehicule_mari)
        total_mici_week = sum(vehicule_mici)
        total_week = total_mari_week + total_mici_week
        
        # Quick stats
        self.create_quick_stat_cards(total_week, total_mari_week, total_mici_week)
        
        # Detailed stats
        stats_container = tk.Frame(self.stats_frame, bg=self.colors['light'])
        stats_container.pack(fill=tk.X, padx=10, pady=10)
        
        # Weekly analysis
        total_daily = [m + s for m, s in zip(vehicule_mari, vehicule_mici)]
        max_day_idx = total_daily.index(max(total_daily)) if total_daily else 0
        min_day_idx = total_daily.index(min(total_daily)) if total_daily else 0
        
        # Weekday vs weekend analysis
        weekdays_total = sum(total_daily[0:5])
        weekend_total = sum(total_daily[5:7])
        
        # Create comprehensive weekly report
        self.create_weekly_report_card(stats_container, {
            'period': f"{week_start.strftime('%d/%m')} - {week_end.strftime('%d/%m/%Y')}",
            'total_week': total_week,
            'total_mari': total_mari_week,
            'total_mici': total_mici_week,
            'avg_daily': total_week / 7 if total_week > 0 else 0,
            'best_day': day_labels[max_day_idx],
            'best_day_count': total_daily[max_day_idx],
            'worst_day': day_labels[min_day_idx],
            'worst_day_count': total_daily[min_day_idx],
            'weekdays_total': weekdays_total,
            'weekend_total': weekend_total,
            'weekdays_avg': weekdays_total / 5 if weekdays_total > 0 else 0,
            'weekend_avg': weekend_total / 2 if weekend_total > 0 else 0
        })
    
    def create_weekly_report_card(self, parent, data):
        """Create comprehensive weekly report card"""
        report_card = tk.Frame(parent, bg=self.colors['surface'], relief='solid', bd=1)
        report_card.pack(fill=tk.X, pady=10)
        
        # Header
        header = tk.Frame(report_card, bg=self.colors['info'], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text=f"📅 Raport Săptămânal - {data['period']}",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['info'],
                fg='white').pack(expand=True)
        
        # Content
        content = tk.Frame(report_card, bg=self.colors['surface'])
        content.pack(fill=tk.X, padx=20, pady=20)
        
        # Summary stats
        summary_frame = tk.Frame(content, bg=self.colors['light'])
        summary_frame.pack(fill=tk.X, pady=(0, 15))
        
        summary_stats = [
            ("📊 Total Săptămânal", f"{data['total_week']:,} vehicule", self.colors['primary']),
            ("📈 Medie Zilnică", f"{data['avg_daily']:.0f} vehicule", self.colors['info']),
            ("🏆 Ziua cu Cel Mai Mult Trafic", f"{data['best_day']}: {data['best_day_count']:,}", self.colors['success']),
            ("📉 Ziua cu Cel Mai Puțin Trafic", f"{data['worst_day']}: {data['worst_day_count']:,}", self.colors['warning'])
        ]
        
        for i, (label, value, color) in enumerate(summary_stats):
            self.create_stat_box(summary_frame, label, value, color, row=i//2, col=i%2)
        
        # Weekday vs Weekend comparison
        if data['weekdays_total'] > 0 or data['weekend_total'] > 0:
            comparison_frame = tk.Frame(content, bg=self.colors['surface'])
            comparison_frame.pack(fill=tk.X, pady=15)
            
            tk.Label(comparison_frame, text="💼 Analiza: Zile Lucrătoare vs Weekend",
                    font=('Segoe UI', 13, 'bold'),
                    bg=self.colors['surface'],
                    fg=self.colors['primary']).pack(anchor='w', pady=(0, 10))
            
            comparison_stats = [
                ("💼 Total Luni-Vineri", f"{data['weekdays_total']:,} ({data['weekdays_avg']:.0f}/zi)"),
                ("🏖️ Total Sâmbătă-Duminică", f"{data['weekend_total']:,} ({data['weekend_avg']:.0f}/zi)"),
                ("📊 Diferența Medie", f"{abs(data['weekdays_avg'] - data['weekend_avg']):.0f} vehicule/zi"),
                ("🎯 Tipul Preferat", "Zile lucrătoare" if data['weekdays_avg'] > data['weekend_avg'] else "Weekend")
            ]
            
            for stat_label, stat_value in comparison_stats:
                stat_row = tk.Frame(comparison_frame, bg=self.colors['surface'])
                stat_row.pack(fill=tk.X, pady=3)
                
                tk.Label(stat_row, text=stat_label,
                        font=('Segoe UI', 11),
                        bg=self.colors['surface'],
                        fg=self.colors['dark']).pack(side=tk.LEFT)
                
                tk.Label(stat_row, text=stat_value,
                        font=('Segoe UI', 11, 'bold'),
                        bg=self.colors['surface'],
                        fg=self.colors['accent']).pack(side=tk.RIGHT)
    
    def generate_modern_monthly_trend(self, date):
        """Generate modern monthly trend view"""
        data = self.db.get_monthly_trend(date[:7])
        
        if not data:
            self.show_no_data_message("Nu există date pentru luna selectată!")
            return
        
        hours = [row[0] for row in data]
        avg_mari = [row[1] for row in data]
        avg_mici = [row[2] for row in data]
        
        # Create modern stacked area chart
        fig = self.create_modern_figure()
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        # Create gradient stacked area
        ax.stackplot(hours, avg_mari, avg_mici,
                    labels=['🚛 Vehicule Mari', '🚗 Vehicule Mici'],
                    colors=[self.colors['danger'], self.colors['success']],
                    alpha=0.8)
        
        # Add trend lines
        ax.plot(hours, avg_mari, color=self.colors['danger'], linewidth=2, alpha=0.9)
        ax.plot(hours, avg_mici, color=self.colors['success'], linewidth=2, alpha=0.9)
        
        # Modern styling
        ax.set_xlabel('Ora zilei', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_ylabel('Medie vehicule', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_title(f'Tendința lunară a traficului\nLuna {date[5:7]}/{date[:4]}',
                    fontsize=16, fontweight='bold', color=self.colors['primary'], pad=20)
        
        # Modern legend and grid
        legend = ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor=self.colors['muted'])
        legend.get_frame().set_facecolor(self.colors['surface'])
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['muted'])
        ax.set_axisbelow(True)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        
        fig.tight_layout(pad=2.0)
        
        # Display chart
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate monthly stats
        total_mari = sum(avg_mari)
        total_mici = sum(avg_mici)
        self.create_quick_stat_cards(total_mari + total_mici, total_mari, total_mici)
    
    def generate_modern_percentage_distribution(self, date):
        """Generate modern percentage distribution"""
        data = self.db.get_daily_totals(date)
        
        if not data:
            self.show_no_data_message()
            return
        
        total_mari, total_mici = data
        total = total_mari + total_mici
        
        if total == 0:
            self.show_no_data_message("Nu există vehicule înregistrate pentru data selectată!")
            return
        
        # Create modern donut chart
        fig = self.create_modern_figure(width=10, height=8)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        # Data for pie chart
        sizes = [total_mari, total_mici]
        labels = ['🚛 Vehicule Mari', '🚗 Vehicule Mici']
        colors = [self.colors['danger'], self.colors['success']]
        explode = (0.05, 0)
        
        # Create donut chart
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                         autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total):,})',
                                         shadow=True, startangle=90,
                                         textprops={'fontsize': 12, 'fontweight': 'bold'},
                                         pctdistance=0.85, labeldistance=1.1)
        
        # Create donut hole
        centre_circle = plt.Circle((0,0), 0.60, fc=self.colors['surface'], linewidth=2, edgecolor=self.colors['muted'])
        ax.add_artist(centre_circle)
        
        # Add center text
        ax.text(0, 0, f'TOTAL\n{total:,}\nvehicule', ha='center', va='center',
               fontsize=16, fontweight='bold', color=self.colors['primary'])
        
        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax.axis('equal')
        ax.set_title(f'Distribuția vehiculelor\n{date}',
                    fontsize=16, fontweight='bold', color=self.colors['primary'], pad=20)
        
        fig.tight_layout(pad=2.0)
        
        # Display chart
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate distribution stats
        self.create_quick_stat_cards(total, total_mari, total_mici)
        
        # Detailed distribution analysis
        self.create_distribution_analysis(total, total_mari, total_mici, date)
    
    def create_distribution_analysis(self, total, total_mari, total_mici, date):
        """Create detailed distribution analysis"""
        analysis_card = tk.Frame(self.stats_frame, bg=self.colors['surface'], relief='solid', bd=1)
        analysis_card.pack(fill=tk.X, padx=10, pady=10)
        
        # Header
        header = tk.Frame(analysis_card, bg=self.colors['accent'], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text=f"🥧 Analiza Distribuției - {date}",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['accent'],
                fg='white').pack(expand=True)
        
        # Content
        content = tk.Frame(analysis_card, bg=self.colors['surface'])
        content.pack(fill=tk.X, padx=20, pady=20)
        
        # Calculate percentages and insights
        percent_mari = (total_mari / total) * 100
        percent_mici = (total_mici / total) * 100
        
        # Distribution insights
        insights = []
        if percent_mari > 60:
            insights.append("🚛 Dominanță vehicule mari")
        elif percent_mici > 60:
            insights.append("🚗 Dominanță vehicule mici")
        else:
            insights.append("⚖️ Distribuție echilibrată")
        
        if abs(percent_mari - percent_mici) < 10:
            insights.append("📊 Distribuție foarte echilibrată")
        
        # Display insights
        for insight in insights:
            tk.Label(content, text=insight,
                    font=('Segoe UI', 12, 'bold'),
                    bg=self.colors['surface'],
                    fg=self.colors['success']).pack(anchor='w', pady=5)
    
    def generate_modern_peak_hours_comparison(self, date):
        """Generate modern peak hours comparison"""
        peak_data = self.db.get_peak_hours_data(date)
        
        if not peak_data:
            self.show_no_data_message("Nu există date pentru analiza orelor de vârf!")
            return
        
        # Define peak hours
        peak_hours = [7, 8, 9, 16, 17, 18]
        peak_mari = sum(row[1] for row in peak_data if int(row[0]) in peak_hours)
        peak_mici = sum(row[2] for row in peak_data if int(row[0]) in peak_hours)
        normal_mari = sum(row[1] for row in peak_data if int(row[0]) not in peak_hours)
        normal_mici = sum(row[2] for row in peak_data if int(row[0]) not in peak_hours)
        
        # Create modern comparison chart
        fig = self.create_modern_figure()
        
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.colors['light'])
        
        categories = ['Ore de vârf\n(7-9, 16-18)', 'Ore normale']
        mari_data = [peak_mari, normal_mari]
        mici_data = [peak_mici, normal_mici]
        
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars with modern styling
        bars1 = ax.bar(x - width/2, mari_data, width, label='🚛 Vehicule Mari',
                      color=self.colors['danger'], alpha=0.9,
                      edgecolor='white', linewidth=2)
        bars2 = ax.bar(x + width/2, mici_data, width, label='🚗 Vehicule Mici',
                      color=self.colors['success'], alpha=0.9,
                      edgecolor='white', linewidth=2)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max(mari_data + mici_data) * 0.02,
                           f'{int(height):,}',
                           ha='center', va='bottom',
                           fontsize=12, fontweight='bold',
                           color=self.colors['dark'])
        
        # Modern styling
        ax.set_xlabel('Perioada zilei', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_ylabel('Numărul de vehicule', fontsize=12, fontweight='bold', color=self.colors['dark'])
        ax.set_title(f'Comparația orelor de vârf cu orele normale\n{date}',
                    fontsize=16, fontweight='bold', color=self.colors['primary'], pad=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories, ha='center')
        
        # Modern legend and grid
        legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                          framealpha=0.95, edgecolor=self.colors['muted'])
        legend.get_frame().set_facecolor(self.colors['surface'])
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color=self.colors['muted'])
        ax.set_axisbelow(True)
        
        # Remove spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(self.colors['muted'])
        ax.spines['bottom'].set_color(self.colors['muted'])
        
        fig.tight_layout(pad=2.0)
        
        # Display chart
        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Generate peak hours analysis
        total_peak = peak_mari + peak_mici
        total_normal = normal_mari + normal_mici
        self.create_quick_stat_cards(total_peak + total_normal, peak_mari + normal_mari, peak_mici + normal_mici)
        
        # Detailed peak analysis
        self.create_peak_analysis(total_peak, total_normal, peak_mari, peak_mici, normal_mari, normal_mici, date)
    
    def create_peak_analysis(self, total_peak, total_normal, peak_mari, peak_mici, normal_mari, normal_mici, date):
        """Create detailed peak hours analysis"""
        analysis_card = tk.Frame(self.stats_frame, bg=self.colors['surface'], relief='solid', bd=1)
        analysis_card.pack(fill=tk.X, padx=10, pady=10)
        
        # Header
        header = tk.Frame(analysis_card, bg=self.colors['warning'], height=50)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        tk.Label(header, text=f"🔥 Analiza Orelor de Vârf - {date}",
                font=('Segoe UI', 14, 'bold'),
                bg=self.colors['warning'],
                fg='white').pack(expand=True)
        
        # Content
        content = tk.Frame(analysis_card, bg=self.colors['surface'])
        content.pack(fill=tk.X, padx=20, pady=20)
        
        # Calculate statistics
        factor = total_peak / total_normal if total_normal > 0 else 0
        peak_percentage = (total_peak / (total_peak + total_normal)) * 100 if (total_peak + total_normal) > 0 else 0
        
        # Peak vs Normal comparison
        comparison_stats = [
            ("📈 Total Ore de Vârf", f"{total_peak:,} vehicule"),
            ("📊 Total Ore Normale", f"{total_normal:,} vehicule"),
            ("⚡ Factor de Intensificare", f"{factor:.1f}x mai intens"),
            ("🎯 Concentrația Traficului", f"{peak_percentage:.1f}% în ore de vârf"),
            ("🚛 Vehicule Mari în Vârf", f"{peak_mari:,} ({(peak_mari/total_peak*100) if total_peak > 0 else 0:.1f}%)"),
            ("🚗 Vehicule Mici în Vârf", f"{peak_mici:,} ({(peak_mici/total_peak*100) if total_peak > 0 else 0:.1f}%)")
        ]
        
        for stat_label, stat_value in comparison_stats:
            stat_row = tk.Frame(content, bg=self.colors['surface'])
            stat_row.pack(fill=tk.X, pady=5)
            
            tk.Label(stat_row, text=stat_label,
                    font=('Segoe UI', 11),
                    bg=self.colors['surface'],
                    fg=self.colors['dark']).pack(side=tk.LEFT)
            
            tk.Label(stat_row, text=stat_value,
                    font=('Segoe UI', 11, 'bold'),
                    bg=self.colors['surface'],
                    fg=self.colors['primary']).pack(side=tk.RIGHT)
        
        # Add insights
        insights_frame = tk.Frame(content, bg=self.colors['light'])
        insights_frame.pack(fill=tk.X, pady=(15, 0))
        
        tk.Label(insights_frame, text="💡 Insights:",
                font=('Segoe UI', 12, 'bold'),
                bg=self.colors['light'],
                fg=self.colors['primary']).pack(anchor='w', pady=(10, 5))
        
        insights = []
        if factor > 2:
            insights.append("🚨 Trafic foarte intens în orele de vârf")
        elif factor > 1.5:
            insights.append("⚠️ Trafic moderat intensificat în orele de vârf")
        else:
            insights.append("✅ Trafic relativ constant pe parcursul zilei")
            
        if peak_percentage > 40:
            insights.append("📊 Mare parte din trafic se concentrează în orele de vârf")
        
        for insight in insights:
            tk.Label(insights_frame, text=insight,
                    font=('Segoe UI', 10),
                    bg=self.colors['light'],
                    fg=self.colors['dark']).pack(anchor='w', pady=2, padx=10)
    
    def show_no_data_message(self, custom_message=None):
        """Show modern no data message"""
        message = custom_message or "Nu există date pentru data selectată!"
        
        # Clear existing content
        for widget in self.graph_frame.winfo_children():
            widget.destroy()
        
        # Create modern no data card
        no_data_frame = tk.Frame(self.graph_frame, bg=self.colors['surface'], relief='solid', bd=1)
        no_data_frame.pack(expand=True, fill=tk.BOTH, padx=50, pady=50)
        
        # Icon and message
        icon_label = tk.Label(no_data_frame, text="📊",
                             font=('Segoe UI', 48),
                             bg=self.colors['surface'],
                             fg=self.colors['muted'])
        icon_label.pack(expand=True, pady=(20, 10))
        
        message_label = tk.Label(no_data_frame, text=message,
                               font=('Segoe UI', 14),
                               bg=self.colors['surface'],
                               fg=self.colors['muted'],
                               justify=tk.CENTER)
        message_label.pack(expand=True, pady=(0, 10))
        
        suggestion_label = tk.Label(no_data_frame, text="💡 Sugestie: Generează date de test pentru a începe analiza",
                                  font=('Segoe UI', 11),
                                  bg=self.colors['surface'],
                                  fg=self.colors['secondary'],
                                  justify=tk.CENTER)
        suggestion_label.pack(expand=True, pady=(0, 20))
        
        # Clear quick stats
        for widget in self.quick_stats_content.winfo_children():
            widget.destroy()
            
        no_stats_label = tk.Label(self.quick_stats_content,
                                text="Selectează o dată cu date\npentru a vedea statisticile",
                                font=('Segoe UI', 11),
                                bg=self.colors['surface'],
                                fg=self.colors['muted'],
                                justify=tk.CENTER)
        no_stats_label.pack(expand=True)
    
    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'db'):
            self.db.close()

# Application entry point with error handling
if __name__ == "__main__":
    try:
        root = tk.Tk()
        
        # Set window icon and properties
        root.iconname("Traffic Analyzer")
        root.minsize(1200, 800)
        
        # Initialize app
        app = BinaryTrafficAnalyzerApp(root)
        
        # Add error handling for main loop
        def on_closing():
            try:
                if hasattr(app, 'db'):
                    app.db.close()
            except:
                pass
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Start application
        root.mainloop()
        
    except Exception as e:
        import tkinter.messagebox as mb
        mb.showerror("Eroare de Inițializare", 
                    f"Nu s-a putut inițializa aplicația:\n{str(e)}\n\nVerificați dacă toate dependențele sunt instalate.")
    finally:
        try:
            root.destroy()
        except:
            pass