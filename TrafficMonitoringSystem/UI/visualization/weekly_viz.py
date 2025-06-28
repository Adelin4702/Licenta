"""
Weekly visualization for Traffic Analyzer App
"""
from .base_viz import BaseVisualization
from UI.utils.date_utils import DateUtils
from UI.utils.constants import CHART_CONFIG, MESSAGES

class WeeklyVisualization(BaseVisualization):
    """Weekly traffic visualization with line charts"""
    
    def generate_visualization(self, date, formatted_date):
        """Generate modern weekly view"""
        # Get week range
        week_start_str, week_end_str, week_start, week_end = DateUtils.get_week_range(date)
        
        if not week_start or not week_end:
            self.show_no_data_message("Eroare la calcularea intervalului săptămânal!")
            return
        
        data = self.db.get_week_data_by_range(week_start_str, week_end_str)
        
        if not data:
            week_display = f"{DateUtils.format_date_short(week_start_str)} - {DateUtils.format_date_for_display(week_end_str)}"
            self.show_no_data_message(f"{MESSAGES['no_weekly_data']} {week_display}")
            return
        
        # Prepare data
        week_data_map = {row[0]: (row[1], row[2]) for row in data}
        vehicule_mari, vehicule_mici, day_labels = self._prepare_week_data(week_start, week_data_map)
        
        # Create visualization
        fig = self.styles.create_modern_figure()
        ax = fig.add_subplot(111)
        
        # Create line chart
        self._create_weekly_lines(ax, vehicule_mari, vehicule_mici, day_labels)
        
        # Apply styling
        week_title = f"Analiza săptămânală a traficului\n{DateUtils.format_date_short(week_start_str)} - {DateUtils.format_date_for_display(week_end_str)}"
        self.style_axes(ax, 'Ziua săptămânii', 'Numărul de vehicule', week_title)
        
        # Configure x-axis
        x_pos = range(len(day_labels))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(day_labels, ha='center')
        
        # Add legend
        self.create_modern_legend(ax)
        
        # Adjust layout and display
        fig.tight_layout(pad=2.0)
        self.create_chart_canvas(fig)
        
        # Generate statistics
        self.generate_stats(vehicule_mari, vehicule_mici, day_labels, week_start, week_end)
    
    def _prepare_week_data(self, week_start, week_data_map):
        """Prepare weekly data for visualization"""
        vehicule_mari = []
        vehicule_mici = []
        day_labels = DateUtils.get_week_day_labels(week_start)
        
        for i in range(7):
            current_day = week_start + DateUtils.timedelta(days=i)
            day_str = current_day.strftime("%Y-%m-%d")
            
            if day_str in week_data_map:
                mari, mici = week_data_map[day_str]
                vehicule_mari.append(mari)
                vehicule_mici.append(mici)
            else:
                vehicule_mari.append(0)
                vehicule_mici.append(0)
        
        return vehicule_mari, vehicule_mici, day_labels
    
    def _create_weekly_lines(self, ax, vehicule_mari, vehicule_mici, day_labels):
        """Create styled line chart for weekly data"""
        x_pos = range(len(day_labels))
        
        # Modern line plots with gradients
        line1 = ax.plot(x_pos, vehicule_mari, marker='o', label='Vehicule Mari',
                       color=self.colors['danger'], 
                       linewidth=CHART_CONFIG['line_width'], 
                       markersize=CHART_CONFIG['marker_size'],
                       markeredgecolor='white', 
                       markeredgewidth=CHART_CONFIG['marker_edge_width'],
                       markerfacecolor=self.colors['danger'])
        
        line2 = ax.plot(x_pos, vehicule_mici, marker='s', label='Vehicule Mici',
                       color=self.colors['success'], 
                       linewidth=CHART_CONFIG['line_width'], 
                       markersize=CHART_CONFIG['marker_size'],
                       markeredgecolor='white', 
                       markeredgewidth=CHART_CONFIG['marker_edge_width'],
                       markerfacecolor=self.colors['success'])
        
        # Add gradient fill
        ax.fill_between(x_pos, vehicule_mari, alpha=CHART_CONFIG['alpha_fill'], color=self.colors['danger'])
        ax.fill_between(x_pos, vehicule_mici, alpha=CHART_CONFIG['alpha_fill'], color=self.colors['success'])
    
    def generate_stats(self, vehicule_mari, vehicule_mici, day_labels, week_start, week_end):
        """Generate comprehensive weekly statistics"""
        total_mari_week = sum(vehicule_mari)
        total_mici_week = sum(vehicule_mici)
        total_week = total_mari_week + total_mici_week
        
        if total_week == 0:
            self.stats_panel.display_stats("Nu există date pentru această săptămână.")
            return
        
        # Weekly analysis
        total_daily = [m + s for m, s in zip(vehicule_mari, vehicule_mici)]
        max_day_idx = total_daily.index(max(total_daily)) if total_daily else 0
        min_day_idx = total_daily.index(min(total_daily)) if total_daily else 0
        
        # Weekday vs weekend analysis
        weekdays_total = sum(total_daily[0:5])
        weekend_total = sum(total_daily[5:7])
        day_labels_clean = [label.replace('\n', ' ') for label in day_labels]
        
        stats_text = f"""📅 SĂPTĂMÂNA {DateUtils.format_date_short(week_start.strftime("%Y-%m-%d"))}-{DateUtils.format_date_short(week_end.strftime("%Y-%m-%d"))}

📊 TOTALURI:
    Total: {total_week:,} vehicule
    Mari: {total_mari_week:,} ({(total_mari_week/total_week*100):.1f}%)
    Mici: {total_mici_week:,} ({(total_mici_week/total_week*100):.1f}%)
    Medie zilnică: {total_week/7:.0f}

🏆 EXTREME:
    Cel mai mult:  {day_labels_clean[max_day_idx]} ({total_daily[max_day_idx]:,})
    Cel mai puțin: {day_labels_clean[min_day_idx]} ({total_daily[min_day_idx]:,})

💼 LUCRĂTOARE vs WEEKEND:
    Luni-Vineri: {weekdays_total:,} ({weekdays_total/5:.0f}/zi)
    Sâmbătă-Duminică: {weekend_total:,} ({weekend_total/2:.0f}/zi)

📈 ZILNIC:"""
        
        for i, day_label in enumerate(day_labels_clean):
            if i < len(total_daily) and total_daily[i] > 0:
                stats_text += f"""
    {day_label}: {total_daily[i]:,} (M:{vehicule_mari[i]}, m:{vehicule_mici[i]})"""
        
        self.stats_panel.display_stats(stats_text)