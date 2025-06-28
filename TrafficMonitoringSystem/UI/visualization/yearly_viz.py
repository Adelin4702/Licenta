"""
Yearly visualization for Traffic Analyzer App
"""
import numpy as np
import datetime
from .base_viz import BaseVisualization
from UI.utils.date_utils import DateUtils
from UI.utils.constants import CHART_CONFIG, MESSAGES

class YearlyVisualization(BaseVisualization):
    """Yearly traffic visualization with monthly progression"""
    
    def generate_visualization(self, date, formatted_date):
        """Generate modern yearly view showing monthly progression"""
        # Clear existing visualization first
        self.clear_visualization()
        
        # Extract year from date
        year = date[:4]  # Extract YYYY
        data = self.get_yearly_data(year)
        
        if not data:
            self.show_no_data_message(f"Nu există date pentru anul {year}!")
            return
        
        # Prepare data
        months, vehicule_mari, vehicule_mici = self._prepare_yearly_data(data, year)
        
        # Create visualization
        fig = self.styles.create_modern_figure(width=10, height=7)
        ax = fig.add_subplot(111)
        
        # Create combined chart (bars + line)
        self._create_yearly_chart(ax, months, vehicule_mari, vehicule_mici)
        
        # Apply styling
        title = f'Evoluția anuală a traficului\nAnul {year}'
        self.style_axes(ax, 'Luna', 'Numărul de vehicule', title)
        
        # Configure x-axis
        x_pos = range(len(months))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(months, rotation=45, ha='right')
        
        # Add legend
        self.create_modern_legend(ax, loc='upper left')
        
        # Adjust layout and display
        fig.tight_layout(pad=2.0)
        self.create_chart_canvas(fig)
        
        # Generate statistics
        self.generate_stats(vehicule_mari, vehicule_mici, months, year)
    
    def get_yearly_data(self, year):
        """Get yearly data - will be replaced by filtered version"""
        try:
            # This method will be replaced by the filtered version in main app
            # Base query for yearly data
            query = '''
                SELECT strftime('%m', r.timestamp) as month,
                       SUM(r.nr_of_large_vehicles) as large_vehicles,
                       SUM(r.nr_of_small_vehicles) as small_vehicles
                FROM record r
                WHERE strftime('%Y', r.timestamp) = ?
                GROUP BY strftime('%m', r.timestamp)
                ORDER BY month
            '''
            
            if hasattr(self.db, 'cursor'):
                self.db.cursor.execute(query, [year])
                return self.db.cursor.fetchall()
            else:
                # Fallback to method call
                return self.db.get_yearly_data(year)
        except Exception as e:
            print(f"Error getting yearly data: {e}")
            return []
    
    def _prepare_yearly_data(self, data, year):
        """Prepare yearly data for visualization"""
        # Create map of existing data
        data_map = {int(row[0]): (row[1], row[2]) for row in data}
        
        vehicule_mari = []
        vehicule_mici = []
        months = []
        
        # Romanian month names
        month_names = [
            '', 'Ian', 'Feb', 'Mar', 'Apr', 'Mai', 'Iun',
            'Iul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]
        
        # Fill data for all 12 months
        for month_num in range(1, 13):
            if month_num in data_map:
                mari, mici = data_map[month_num]
                vehicule_mari.append(mari or 0)
                vehicule_mici.append(mici or 0)
            else:
                vehicule_mari.append(0)
                vehicule_mici.append(0)
            
            months.append(month_names[month_num])
        
        return months, vehicule_mari, vehicule_mici
    
    def _create_yearly_chart(self, ax, months, vehicule_mari, vehicule_mici):
        """Create styled yearly chart with area lines"""
        x_pos = np.arange(len(months))
        
        # Create area plots with lines
        ax.fill_between(x_pos, vehicule_mari, 
                       color=self.colors['danger'], alpha=0.3, 
                       label='Vehicule Mari')
        
        ax.fill_between(x_pos, vehicule_mici, 
                       color=self.colors['success'], alpha=0.3, 
                       label='Vehicule Mici')
        
        # Add line plots on top
        ax.plot(x_pos, vehicule_mari, 
               color=self.colors['danger'], 
               linewidth=3, 
               marker='o', 
               markersize=8,
               markeredgecolor='white',
               markeredgewidth=2,
               label='_nolegend_')  # Don't show in legend
        
        ax.plot(x_pos, vehicule_mici, 
               color=self.colors['success'], 
               linewidth=3, 
               marker='s', 
               markersize=8,
               markeredgecolor='white',
               markeredgewidth=2,
               label='_nolegend_')  # Don't show in legend
        
        # Calculate and add total line
        totals = [mari + mici for mari, mici in zip(vehicule_mari, vehicule_mici)]
        ax.plot(x_pos, totals, 
               color=self.colors['primary'], 
               linewidth=4, 
               marker='D', 
               markersize=10,
               markeredgecolor='white',
               markeredgewidth=2,
               label='Total',
               linestyle='--',
               alpha=0.9)
        
        # Add value annotations for totals only
        self._add_total_annotations(ax, x_pos, totals)
    
    def _add_total_annotations(self, ax, x_pos, totals):
        """Add annotations only for total values"""
        max_value = max(totals) if totals else 1
        
        for i, total in enumerate(totals):
            if total > max_value * 0.1:  # Only show if > 10% of max
                ax.annotate(f'{total}',
                           (x_pos[i], total),
                           textcoords="offset points",
                           xytext=(0,15),
                           ha='center',
                           fontsize=10,
                           fontweight='bold',
                           color=self.colors['primary'],
                           bbox=dict(boxstyle="round,pad=0.4", 
                                   facecolor='white', 
                                   edgecolor=self.colors['primary'],
                                   alpha=0.9))
    
    def generate_stats(self, vehicule_mari, vehicule_mici, months, year):
        """Generate comprehensive yearly statistics with colored sections"""
        # Calculate totals and statistics
        total_mari_year = sum(vehicule_mari)
        total_mici_year = sum(vehicule_mici)
        total_year = total_mari_year + total_mici_year
        
        if total_year == 0:
            self.stats_panel.display_stats("Nu există date pentru acest an.")
            return
        
        # Monthly totals
        monthly_totals = [mari + mici for mari, mici in zip(vehicule_mari, vehicule_mici)]
        
        # Find peak and low months (only non-zero)
        non_zero_totals = [t for t in monthly_totals if t > 0]
        if non_zero_totals:
            max_month_idx = monthly_totals.index(max(monthly_totals))
            min_month_idx = monthly_totals.index(min(non_zero_totals))
        else:
            max_month_idx = min_month_idx = 0
        
        # Growth analysis
        growth_analysis = self._analyze_growth_trend(monthly_totals, months)
        
        # Clear previous stats and create colored sections
        self.stats_panel.clear_stats()
        
        # Annual totals section
        self.stats_panel.add_stats_section(
            title=f"📊 Totaluri anuale {year}", 
            content=f"Total: {total_year} vehicule\nMari: {total_mari_year} ({(total_mari_year/total_year*100):.1f}%)\nMici: {total_mici_year} ({(total_mici_year/total_year*100):.1f}%)\nMedie lunară: {total_year/12:.0f} vehicule",
            title_color=self.colors['primary']
        )
        
        self.stats_panel.add_divider()
        
        # Monthly extremes section
        self.stats_panel.add_stats_section(
            title="🎯 Extreme lunare", 
            content=f"🏆 Vârf: {months[max_month_idx]} ({monthly_totals[max_month_idx]})\n📉 Minim: {months[min_month_idx]} ({monthly_totals[min_month_idx]})\n📈 Diferența: {monthly_totals[max_month_idx] - monthly_totals[min_month_idx]}",
            title_color=self.colors['warning']
        )
        
        self.stats_panel.add_divider()
        
        # Growth trend section
        self.stats_panel.add_stats_section(
            title="📈 Tendință creștere", 
            content=growth_analysis,
            title_color=self.colors['success']
        )
        
        self.stats_panel.add_divider()
        
        # Monthly breakdown
        monthly_content = ""
        for i, month in enumerate(months):
            if monthly_totals[i] > 0:
                monthly_content += f"{month}: {monthly_totals[i]} (M:{vehicule_mari[i]}, m:{vehicule_mici[i]})\n"
        
        self.stats_panel.add_stats_section(
            title="📋 Detalii lunare", 
            content=monthly_content.strip(),
            title_color=self.colors['info']
        )
        
        self.stats_panel.text_stats_frame.update_idletasks()
        self.stats_panel.canvas.configure(scrollregion=self.stats_panel.canvas.bbox("all"))
    
    def _analyze_growth_trend(self, monthly_totals, months):
        """Analyze growth trend throughout the year"""
        # Filter out zero months for trend analysis
        non_zero_data = [(i, total) for i, total in enumerate(monthly_totals) if total > 0]
        
        if len(non_zero_data) < 2:
            return "Insuficiente date pentru analiza tendinței"
        
        insights = []
        
        # Calculate quarter-over-quarter growth
        quarters = {
            'Q1': [0, 1, 2],    # Ian, Feb, Mar
            'Q2': [3, 4, 5],    # Apr, Mai, Iun  
            'Q3': [6, 7, 8],    # Iul, Aug, Sep
            'Q4': [9, 10, 11]   # Oct, Nov, Dec
        }
        
        quarter_totals = {}
        for quarter, month_indices in quarters.items():
            total = sum(monthly_totals[i] for i in month_indices)
            if total > 0:
                quarter_totals[quarter] = total
        
        # Analyze quarterly growth
        if len(quarter_totals) >= 2:
            quarter_list = list(quarter_totals.items())
            for i in range(1, len(quarter_list)):
                prev_quarter, prev_total = quarter_list[i-1]
                curr_quarter, curr_total = quarter_list[i]
                
                if prev_total > 0:
                    growth = ((curr_total - prev_total) / prev_total) * 100
                    if growth > 10:
                        insights.append(f"{curr_quarter}: Creștere {growth:+.1f}% față de {prev_quarter}")
                    elif growth < -10:
                        insights.append(f"{curr_quarter}: Scădere {growth:+.1f}% față de {prev_quarter}")
                    else:
                        insights.append(f"{curr_quarter}: Stabil ({growth:+.1f}%)")
        
        # Overall trend
        if len(non_zero_data) >= 3:
            first_month_total = non_zero_data[0][1]
            last_month_total = non_zero_data[-1][1]
            
            overall_growth = ((last_month_total - first_month_total) / first_month_total) * 100
            
            if overall_growth > 20:
                insights.append(f"Tendință generală: Creștere puternică ({overall_growth:+.1f}%)")
            elif overall_growth > 5:
                insights.append(f"Tendință generală: Creștere moderată ({overall_growth:+.1f}%)")
            elif overall_growth < -20:
                insights.append(f"Tendință generală: Scădere puternică ({overall_growth:+.1f}%)")
            elif overall_growth < -5:
                insights.append(f"Tendință generală: Scădere moderată ({overall_growth:+.1f}%)")
            else:
                insights.append(f"Tendință generală: Stabilă ({overall_growth:+.1f}%)")
        
        return "\n".join(insights) if insights else "Tendință în curs de dezvoltare"