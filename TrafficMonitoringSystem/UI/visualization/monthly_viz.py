"""
Monthly visualization for Traffic Analyzer App
"""
from .base_viz import BaseVisualization
from UI.utils.date_utils import DateUtils
from UI.utils.constants import MESSAGES

class MonthlyVisualization(BaseVisualization):
    """Monthly trend visualization with stacked area charts"""
    
    def generate_visualization(self, date, formatted_date):
        """Generate modern monthly trend view"""
        month_year = date[:7]  # Extract YYYY-MM
        data = self.db.get_monthly_trend(month_year)
        
        if not data:
            self.show_no_data_message(MESSAGES['no_monthly_data'])
            return
        
        hours = [row[0] for row in data]
        avg_mari = [row[1] for row in data]
        avg_mici = [row[2] for row in data]
        
        # Create modern stacked area chart
        fig = self.styles.create_modern_figure()
        ax = fig.add_subplot(111)
        
        # Create stacked area chart
        self._create_monthly_area_chart(ax, hours, avg_mari, avg_mici)
        
        # Apply styling
        month_display = DateUtils.get_month_year(date)
        title = f'Tendința lunară a traficului\nLuna {month_display}'
        self.style_axes(ax, 'Ora zilei', 'Medie vehicule', title)
        
        # Add legend
        self.create_modern_legend(ax, loc='upper right')
        
        # Adjust layout and display
        fig.tight_layout(pad=2.0)
        self.create_chart_canvas(fig)
        
        # Generate statistics
        self.generate_stats(avg_mari, avg_mici, hours, month_display)
    
    def _create_monthly_area_chart(self, ax, hours, avg_mari, avg_mici):
        """Create styled stacked area chart for monthly data"""
        # Create gradient stacked area
        ax.stackplot(hours, avg_mari, avg_mici,
                    labels=['Vehicule Mari', 'Vehicule Mici'],
                    colors=[self.colors['danger'], self.colors['success']],
                    alpha=0.8)
        
        # Add trend lines on top
        ax.plot(hours, avg_mari, color=self.colors['danger'], linewidth=2, alpha=0.9)
        ax.plot(hours, avg_mici, color=self.colors['success'], linewidth=2, alpha=0.9)
    
    def generate_stats(self, avg_mari, avg_mici, hours, month_display):
        """Generate comprehensive monthly statistics"""
        if not avg_mari or not avg_mici:
            self.stats_panel.display_stats("Nu există date pentru această lună.")
            return
        
        # Calculate totals and averages
        total_mari = sum(avg_mari)
        total_mici = sum(avg_mici)
        total = total_mari + total_mici
        
        # Calculate average per hour across the month
        avg_mari_per_hour = total_mari / len(avg_mari)
        avg_mici_per_hour = total_mici / len(avg_mici)
        
        # Find peak hours
        peak_mari_idx = avg_mari.index(max(avg_mari)) if max(avg_mari) > 0 else 0
        peak_mici_idx = avg_mici.index(max(avg_mici)) if max(avg_mici) > 0 else 0
        
        stats_text = f"""📅 LUNA {month_display}

📊 MEDII LUNARE:
    Total: {(avg_mici_per_hour + avg_mari_per_hour):.1f}/oră
    Mari: {avg_mari_per_hour:.1f}/oră ({(total_mari/total*100) if total > 0 else 0:.1f}%)
    Mici: {avg_mici_per_hour:.1f}/oră ({(total_mici/total*100) if total > 0 else 0:.1f}%)

⏰ ORE DE VÂRF:
    🚛 Mari: {hours[peak_mari_idx] if peak_mari_idx < len(hours) else 0}:00 ({avg_mari[peak_mari_idx] if peak_mari_idx < len(avg_mari) else 0:.1f})
    🚗 Mici: {hours[peak_mici_idx] if peak_mici_idx < len(hours) else 0}:00 ({avg_mici[peak_mici_idx] if peak_mici_idx < len(avg_mici) else 0:.1f})

📈 DISTRIBUȚIA ORARĂ:"""
        
        # Show hourly distribution
        for i, hour in enumerate(hours):
            if i < len(avg_mari) and i < len(avg_mici):
                hour_total = avg_mari[i] + avg_mici[i]
                stats_text += f"""
    {hour}:00 → {avg_mari[i]:.1f} 🚛, {avg_mici[i]:.1f} 🚗 (total: {hour_total:.1f})"""
        
        self.stats_panel.display_stats(stats_text)