"""
Distribution visualization for Traffic Analyzer App
"""
import matplotlib.pyplot as plt
import numpy as np
from .base_viz import BaseVisualization

class DistributionVisualization(BaseVisualization):
    """Percentage distribution visualization with enhanced donut charts"""
    
    def generate_visualization(self, date, formatted_date):
        """Generate modern percentage distribution with enhanced styling"""
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
        fig = self.styles.create_modern_figure(width=8, height=7)
        ax = fig.add_subplot(111)
        
        # Create enhanced donut chart
        self._create_enhanced_donut_chart(ax, total_mari, total_mici, total, formatted_date)
        
        # Display chart
        fig.tight_layout(pad=2.0)
        self.create_chart_canvas(fig)
        
        # Generate enhanced statistics
        self.generate_enhanced_stats(total, total_mari, total_mici, formatted_date)
    
    def _create_enhanced_donut_chart(self, ax, total_mari, total_mici, total, formatted_date):
        """Create enhanced donut chart with modern styling"""
        ax.set_facecolor(self.colors['light'])
        
        # Data for pie chart
        sizes = [total_mari, total_mici]
        labels = ['Vehicule Mari', 'Vehicule Mici']
        colors = [self.colors['danger'], self.colors['success']]
        explode = (0.08, 0.02)  # Slightly more separation
        
        # Create outer donut chart
        wedges, texts, autotexts = ax.pie(
            sizes, 
            explode=explode, 
            labels=labels, 
            colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total):,})',
            shadow=True, 
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            pctdistance=0.85, 
            labeldistance=1.15,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
        )
        
        # Create inner circle (donut hole) with gradient effect
        inner_circle = plt.Circle((0, 0), 0.6, fc=self.colors['surface'], 
                                 linewidth=3, edgecolor=self.colors['primary'], alpha=0.9)
        ax.add_artist(inner_circle)
        
        # Add center text with better formatting
        center_text = f'TOTAL\n{total:,}\nvehicule'
        ax.text(0, 0, center_text, ha='center', va='center',
               fontsize=15, fontweight='bold', color=self.colors['primary'])
        
        # Style the percentage text
        for i, autotext in enumerate(autotexts):
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
            # Add subtle shadow effect
            autotext.set_path_effects([plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black', alpha=0.3)])
        
        # Enhanced label styling
        for i, text in enumerate(texts):
            text.set_fontsize(12)
            text.set_fontweight('bold')
            text.set_color(self.colors['dark'])
        
        # Add decorative elements
        self._add_chart_decorations(ax, total_mari, total_mici, total)
        
        ax.axis('equal')
        ax.set_title(f'Distribuția vehiculelor\n{formatted_date}',
                    fontsize=17, fontweight='bold', color=self.colors['primary'], pad=25)
    
    def _add_chart_decorations(self, ax, total_mari, total_mici, total):
        """Add decorative elements to enhance the chart"""
        # Calculate percentages
        percent_mari = (total_mari / total) * 100
        percent_mici = (total_mici / total) * 100
        
        # Add small info boxes around the chart
        info_positions = [
            (1.3, 0.5, f"🚛 {total_mari:,}\n{percent_mari:.1f}%", self.colors['danger']),
            (1.3, -0.5, f"🚗 {total_mici:,}\n{percent_mici:.1f}%", self.colors['success'])
        ]
        
        for x, y, text, color in info_positions:
            # Create info box
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.1, edgecolor=color)
            ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold',
                   color=color, bbox=bbox_props)
    
    def generate_enhanced_stats(self, total, total_mari, total_mici, formatted_date):
        """Generate comprehensive distribution statistics with visual enhancements"""
        percent_mari = (total_mari / total) * 100
        percent_mici = (total_mici / total) * 100
        
        # Clear existing stats
        self.stats_panel.clear_stats()
        
        # Add main metrics as cards
        self.stats_panel.add_metric_card(
            "Total Vehicule", 
            f"{total:,}", 
            icon="🚦", 
            color=self.colors['primary']
        )
        
        self.stats_panel.add_metric_card(
            "Vehicule Mari", 
            f"{total_mari:,}", 
            unit=f"({percent_mari:.1f}%)",
            icon="🚛", 
            color=self.colors['danger']
        )
        
        self.stats_panel.add_metric_card(
            "Vehicule Mici", 
            f"{total_mici:,}", 
            unit=f"({percent_mici:.1f}%)",
            icon="🚗", 
            color=self.colors['success']
        )
        
        self.stats_panel.add_divider()
        
        # Generate traffic analysis insights
        insights = self._generate_traffic_insights(percent_mari, percent_mici, total)
        
        self.stats_panel.add_stats_section(
            "💡 ANALIZĂ TRAFIC",
            insights,
            self.colors['info']
        )
        
        # Add recommendations
        recommendations = self._generate_recommendations(percent_mari, percent_mici, total)
        
        self.stats_panel.add_stats_section(
            "📋 RECOMANDĂRI",
            recommendations,
            self.colors['accent']
        )
        
        # Scroll to top for better visibility
        self.stats_panel.scroll_to_top()
    
    def _generate_traffic_insights(self, percent_mari, percent_mici, total):
        """Generate intelligent traffic analysis insights"""
        insights = []
        
        # Determine traffic type
        if percent_mari > 60:
            insights.append("🚛 Dominanță vehicule mari")
            insights.append("Tipul: Transport comercial")
            insights.append("Zona: Probabil industrială/logistică")
        elif percent_mici > 70:
            insights.append("🚗 Dominanță vehicule mici")
            insights.append("Tipul: Trafic personal")
            insights.append("Zona: Probabil rezidențială/urbană")
        else:
            insights.append("⚖️ Distribuție echilibrată")
            insights.append("Tipul: Trafic mixt")
            insights.append("Zona: Arteră principală")
        
        # Traffic intensity analysis
        if total > 1000:
            insights.append("📈 Trafic intens")
        elif total > 500:
            insights.append("📊 Trafic moderat")
        else:
            insights.append("📉 Trafic redus")
        
        return "\n".join(f"• {insight}" for insight in insights)
    
    def _generate_recommendations(self, percent_mari, percent_mici, total):
        """Generate actionable recommendations based on traffic distribution"""
        recommendations = []
        
        # Traffic management recommendations
        if percent_mari > 60:
            recommendations.append("Considerați benzi dedicate pentru camioane")
            recommendations.append("Implementați restricții de oră pentru vehicule mari")
            recommendations.append("Monitorizați uzura drumului")
        elif percent_mici > 70:
            recommendations.append("Optimizați semaforele pentru fluiditate")
            recommendations.append("Considerați piste pentru biciclete")
            recommendations.append("Implementați zone cu viteză redusă")
        
        # Volume-based recommendations
        if total > 1000:
            recommendations.append("Evaluați necesitatea unor benzi suplimentare")
            recommendations.append("Implementați sisteme de management trafic")
        elif total < 200:
            recommendations.append("Verificați necesitatea semaforului")
            recommendations.append("Considerați sens giratoriu")
        
        if not recommendations:
            recommendations.append("Monitorizați în continuare traficul")
            recommendations.append("Colectați date pe perioade mai lungi")
        
        return "\n".join(f"• {rec}" for rec in recommendations)