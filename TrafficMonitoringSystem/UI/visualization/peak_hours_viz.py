"""
Peak hours visualization for Traffic Analyzer App
"""
import numpy as np
from .base_viz import BaseVisualization
from UI.utils.constants import PEAK_HOURS, CHART_CONFIG, MESSAGES

class PeakHoursVisualization(BaseVisualization):
    """Peak hours comparison visualization with bar charts"""
    
    def generate_visualization(self, date, formatted_date):
        """Generate modern peak hours comparison"""
        peak_data = self.db.get_peak_hours_data(date)
        
        if not peak_data:
            self.show_no_data_message(MESSAGES['no_peak_data'])
            return
        
        # Calculate peak vs normal hours data
        peak_mari, peak_mici, normal_mari, normal_mici = self._calculate_peak_data(peak_data)
        
        # Create modern comparison chart
        fig = self.styles.create_modern_figure()
        ax = fig.add_subplot(111)
        
        # Create comparison bars
        self._create_comparison_bars(ax, peak_mari, peak_mici, normal_mari, normal_mici)
        
        # Apply styling
        title = f'Comparația orelor de vârf cu orele normale\n{formatted_date}'
        self.style_axes(ax, 'Perioada zilei', 'Numărul de vehicule', title)
        
        # Configure categories
        categories = ['Ore de vârf\n(7-9, 16-18)', 'Ore normale']
        x = np.arange(len(categories))
        ax.set_xticks(x)
        ax.set_xticklabels(categories, ha='center')
        
        # Add legend
        self.create_modern_legend(ax)
        
        # Adjust layout and display
        fig.tight_layout(pad=2.0)
        self.create_chart_canvas(fig)
        
        # Generate statistics
        total_peak = peak_mari + peak_mici
        total_normal = normal_mari + normal_mici
        self.generate_stats(total_peak, total_normal, peak_mari, peak_mici, normal_mari, normal_mici, formatted_date)
    
    def _calculate_peak_data(self, peak_data):
        """Calculate peak vs normal hours data"""
        peak_mari = sum(row[1] for row in peak_data if int(row[0]) in PEAK_HOURS)
        peak_mici = sum(row[2] for row in peak_data if int(row[0]) in PEAK_HOURS)
        normal_mari = sum(row[1] for row in peak_data if int(row[0]) not in PEAK_HOURS)
        normal_mici = sum(row[2] for row in peak_data if int(row[0]) not in PEAK_HOURS)
        
        return peak_mari, peak_mici, normal_mari, normal_mici
    
    def _create_comparison_bars(self, ax, peak_mari, peak_mici, normal_mari, normal_mici):
        """Create styled comparison bar chart"""
        categories = ['Ore de vârf\n(7-9, 16-18)', 'Ore normale']
        mari_data = [peak_mari, normal_mari]
        mici_data = [peak_mici, normal_mici]
        
        x = np.arange(len(categories))
        width = CHART_CONFIG['bar_width']
        
        # Create bars with modern styling
        bars1 = ax.bar(x - width/2, mari_data, width, label='Vehicule Mari',
                      color=self.colors['danger'], alpha=CHART_CONFIG['alpha'],
                      edgecolor='white', linewidth=2)
        bars2 = ax.bar(x + width/2, mici_data, width, label='Vehicule Mici',
                      color=self.colors['success'], alpha=CHART_CONFIG['alpha'],
                      edgecolor='white', linewidth=2)
        
        # Add value labels
        all_values = mari_data + mici_data
        self.add_value_labels_to_bars(ax, bars1, all_values)
        self.add_value_labels_to_bars(ax, bars2, all_values)
    
    def generate_stats(self, total_peak, total_normal, peak_mari, peak_mici, normal_mari, normal_mici, formatted_date):
        """Generate peak hours analysis statistics with colored sections"""
        # Calculate factor
        factor = total_peak / total_normal if total_normal > 0 else 0
        
        # Clear previous stats and create colored sections
        self.stats_panel.clear_stats()
        
        # Comparison section
        self.stats_panel.add_stats_section(
            title="📊 Comparație generale", 
            content=f"Ore de vârf: {total_peak:,}\nOre normale: {total_normal:,}\nFactor: {factor:.1f}x mai intens la vârf",
            title_color=self.colors['primary']
        )
        
        self.stats_panel.add_divider()
        
        # Large vehicles section
        self.stats_panel.add_stats_section(
            title="🚛 Vehicule mari", 
            content=f"Ore de vârf: {peak_mari:,} ({(peak_mari/total_peak*100) if total_peak > 0 else 0:.1f}%)\nOre normale: {normal_mari:,} ({(normal_mari/total_normal*100) if total_normal > 0 else 0:.1f}%)",
            title_color=self.colors['danger']
        )
        
        self.stats_panel.add_divider()
        
        # Small vehicles section
        self.stats_panel.add_stats_section(
            title="🚗 Vehicule mici", 
            content=f"Ore de vârf: {peak_mici:,} ({(peak_mici/total_peak*100) if total_peak > 0 else 0:.1f}%)\nOre normale: {normal_mici:,} ({(normal_mici/total_normal*100) if total_normal > 0 else 0:.1f}%)",
            title_color=self.colors['success']
        )