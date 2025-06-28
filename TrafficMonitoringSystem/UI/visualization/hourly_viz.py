"""
Hourly visualization for Traffic Analyzer App
"""
import numpy as np
from .base_viz import BaseVisualization
from UI.utils.date_utils import DateUtils
from UI.utils.constants import CHART_CONFIG

class HourlyVisualization(BaseVisualization):
    """Hourly traffic visualization with bar charts"""
    
    def generate_visualization(self, date, formatted_date):
        """Generate modern hourly view"""
        data = self.db.get_hourly_data(date)
        
        if not data:
            self.show_no_data_message()
            return
        
        hours = [row[0] for row in data]
        vehicule_mari = [row[1] for row in data]
        vehicule_mici = [row[2] for row in data]
        
        # Create modern figure
        fig = self.styles.create_modern_figure()
        ax = fig.add_subplot(111)
        
        # Create modern bar chart
        self._create_hourly_bars(ax, hours, vehicule_mari, vehicule_mici)
        
        # Apply styling
        self.style_axes(ax, 'Ora zilei', 'Numărul de vehicule', 
                       f'Analiza traficului orar\n{formatted_date}')
        
        # Configure x-axis
        x = np.arange(len(hours))
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}:00" for h in hours], rotation=45, ha='right')
        
        # Add legend
        self.create_modern_legend(ax)
        
        # Adjust layout and display
        fig.tight_layout(pad=1.0)
        self.create_chart_canvas(fig)
        
        # Generate statistics
        self.generate_stats(vehicule_mari, vehicule_mici, hours, formatted_date)
    
    def _create_hourly_bars(self, ax, hours, vehicule_mari, vehicule_mici):
        """Create styled bar chart for hourly data"""
        width = CHART_CONFIG['bar_width']
        x = np.arange(len(hours))
        
        # Create bars with modern colors
        bars1 = ax.bar([i - width/2 for i in x], vehicule_mari, width,
                      label='Vehicule Mari', 
                      color=self.colors['danger'],
                      alpha=CHART_CONFIG['alpha'],
                      edgecolor='white',
                      linewidth=1)
        
        bars2 = ax.bar([i + width/2 for i in x], vehicule_mici, width,
                      label='Vehicule Mici',
                      color=self.colors['success'], 
                      alpha=CHART_CONFIG['alpha'],
                      edgecolor='white',
                      linewidth=1)
        
        # Add value labels
        all_values = vehicule_mari + vehicule_mici
        self.add_value_labels_to_bars(ax, bars1, all_values)
        self.add_value_labels_to_bars(ax, bars2, all_values)
    
    def generate_stats(self, vehicule_mari, vehicule_mici, hours, formatted_date):
        """Generate comprehensive hourly statistics"""
        total_mari = sum(vehicule_mari)
        total_mici = sum(vehicule_mici)
        total = total_mari + total_mici
        
        if total == 0:
            self.stats_panel.display_stats("Nu există date pentru această zi.")
            return
        
        # Calculate statistics
        stats_text = f"""📊 STATISTICI {formatted_date}

🚦 TOTALURI:
    Total: {total:,} vehicule
    Mari: {total_mari:,} ({(total_mari/total*100):.1f}%)
    Mici: {total_mici:,} ({(total_mici/total*100):.1f}%)

⏰ ORE DE VÂRF:"""
        
        if vehicule_mici and vehicule_mari:
            peak_hour_mici_idx = vehicule_mici.index(max(vehicule_mici))
            peak_hour_mari_idx = vehicule_mari.index(max(vehicule_mari))
            peak_hour_mici = hours[peak_hour_mici_idx]
            peak_hour_mari = hours[peak_hour_mari_idx]
            
            stats_text += f"""
    🚗 Vehicule mici: {peak_hour_mici}:00 ({max(vehicule_mici)})
    🚛 Vehicule mari: {peak_hour_mari}:00 ({max(vehicule_mari)})
    📈 Medie: {total/len(hours):.1f}/oră

📋 DISTRIBUȚIA ORARĂ:"""
            
            # Show only non-zero hours
            for i, hour in enumerate(hours):
                if i < len(vehicule_mari) and i < len(vehicule_mici):
                    hour_total = vehicule_mari[i] + vehicule_mici[i]
                    if hour_total > 0:
                        stats_text += f"""
    {hour}:00 → {vehicule_mari[i]} 🚛, {vehicule_mici[i]} 🚗 (total: {hour_total})"""
        
        self.stats_panel.display_stats(stats_text)