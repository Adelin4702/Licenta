"""
Distribution visualization for Traffic Analyzer App
"""
import matplotlib.pyplot as plt
import numpy as np

from .base_viz import BaseVisualization

class DistributionVisualization(BaseVisualization):
    """Percentage distribution visualization with clean donut charts and simple stats"""

    def generate_visualization(self, date, formatted_date):
        """Generate clean percentage distribution visualization"""
        self.clear_visualization()

        data = self.db.get_daily_totals(date)

        if not data:
            self.show_no_data_message()
            return

        total_mari, total_mici = data
        total = total_mari + total_mici

        if total == 0:
            self.show_no_data_message("Nu există vehicule înregistrate pentru data selectată!")
            return

        fig = self.styles.create_modern_figure(width=8, height=7)
        ax = fig.add_subplot(111)

        self._create_donut_chart(ax, total_mari, total_mici, total, formatted_date)

        fig.tight_layout(pad=2.0)
        self.create_chart_canvas(fig)

        self.generate_simple_stats(total, total_mari, total_mici, formatted_date)

    def _create_donut_chart(self, ax, total_mari, total_mici, total, formatted_date):
        """Create a clean donut chart without visual effects"""
        ax.set_facecolor(self.colors['light'])

        sizes = [total_mari, total_mici]
        labels = ['Vehicule Mari', 'Vehicule Mici']
        colors = [self.colors['danger'], self.colors['success']]
        explode = (0.08, 0.02)

        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total):,})',
            shadow=False,
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
            pctdistance=0.85,
            labeldistance=1.15,
            wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2)
        )

        inner_circle = plt.Circle((0, 0), 0.6, fc=self.colors['surface'],
                                  linewidth=3, edgecolor=self.colors['primary'], alpha=0.9)
        ax.add_artist(inner_circle)

        ax.text(0, 0, f'TOTAL\n{total:,}\nvehicule', ha='center', va='center',
                fontsize=15, fontweight='bold', color=self.colors['primary'])

        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)

        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
            text.set_color(self.colors['dark'])

        ax.axis('equal')
        ax.set_title(f'Distribuția vehiculelor\n{formatted_date}',
                     fontsize=17, fontweight='bold', color=self.colors['primary'], pad=25)

    def generate_simple_stats(self, total, total_mari, total_mici, formatted_date):
        """Display simple text-based statistics instead of cards"""
        percent_mari = (total_mari / total) * 100
        percent_mici = (total_mici / total) * 100

        self.stats_panel.clear_stats()

        stats = [
            f"🚦 Total vehicule: {total:,}",
            f"🚛 Vehicule mari: {total_mari:,} ({percent_mari:.1f}%)",
            f"🚗 Vehicule mici: {total_mici:,} ({percent_mici:.1f}%)"
        ]

        self.stats_panel.add_stats_section(
            "📊 STATISTICI ZILNICE",
            "\n".join(stats),
            self.colors['primary']
        )

        self.stats_panel.add_divider()
        
        insights = self._generate_traffic_insights(percent_mari, percent_mici, total)
        self.stats_panel.add_stats_section(
            "💡 ANALIZĂ TRAFIC",
            insights,
            self.colors['info']
        )
        
        self.stats_panel.add_divider()

        recommendations = self._generate_recommendations(percent_mari, percent_mici, total)
        self.stats_panel.add_stats_section(
            "📋 RECOMANDĂRI",
            recommendations,
            self.colors['accent']
        )

        self.stats_panel.scroll_to_top()

    def _generate_traffic_insights(self, percent_mari, percent_mici, total):
        """Generate intelligent traffic analysis insights"""
        insights = []

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

        if percent_mari > 60:
            recommendations.append("Considerați benzi dedicate pentru camioane")
            recommendations.append("Implementați restricții de oră pentru vehicule mari")
            recommendations.append("Monitorizați uzura drumului")
        elif percent_mici > 70:
            recommendations.append("Optimizați semaforele pentru fluiditate")
            recommendations.append("Considerați piste pentru biciclete")
            recommendations.append("Implementați zone cu viteză redusă")

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
