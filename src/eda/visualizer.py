"""
Sales Data Visualizer - EDA Visualizations.

Creates comprehensive visualizations for exploratory data analysis:
- Time series plots
- Distribution plots
- Seasonality plots
- Correlation heatmaps
- Trend analysis plots
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class SalesVisualizer:
    """
    Comprehensive sales data visualizer.
    
    Creates publication-ready visualizations for EDA reports.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str = 'date',
        target_col: str = 'daily_sales',
        output_dir: str = 'reports/eda'
    ):
        """
        Initialize visualizer.
        
        Args:
            df: Sales data DataFrame
            date_col: Name of date column
            target_col: Name of target column
            output_dir: Directory to save plots
        """
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure date is datetime
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.sort_values(date_col).reset_index(drop=True)
        
        # Color scheme
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#3B3B3B'
        }
    
    def create_all_plots(self) -> dict:
        """Create all EDA plots and return paths."""
        paths = {}
        
        paths['time_series'] = self.plot_time_series()
        paths['distribution'] = self.plot_distribution()
        paths['seasonality'] = self.plot_seasonality()
        paths['trend'] = self.plot_trend_analysis()
        paths['correlation'] = self.plot_correlation_heatmap()
        paths['boxplots'] = self.plot_boxplots()
        paths['summary'] = self.plot_summary_dashboard()
        
        return paths
    
    def plot_time_series(self, figsize: Tuple[int, int] = (14, 6)) -> str:
        """Plot time series of daily sales."""
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(
            self.df[self.date_col],
            self.df[self.target_col],
            color=self.colors['primary'],
            linewidth=1,
            alpha=0.8
        )
        
        # Add 7-day moving average
        ma7 = self.df[self.target_col].rolling(window=7).mean()
        ax.plot(
            self.df[self.date_col],
            ma7,
            color=self.colors['accent'],
            linewidth=2,
            label='7-Day Moving Average'
        )
        
        # Add 30-day moving average
        ma30 = self.df[self.target_col].rolling(window=30).mean()
        ax.plot(
            self.df[self.date_col],
            ma30,
            color=self.colors['secondary'],
            linewidth=2,
            label='30-Day Moving Average'
        )
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Daily Sales ($)', fontsize=12)
        ax.set_title('Daily Sales Over Time', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        plt.tight_layout()
        
        path = self.output_dir / 'time_series.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def plot_distribution(self, figsize: Tuple[int, int] = (14, 5)) -> str:
        """Plot distribution of sales."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        target = self.df[self.target_col]
        
        # Histogram
        axes[0].hist(
            target,
            bins=30,
            color=self.colors['primary'],
            edgecolor='white',
            alpha=0.7
        )
        axes[0].axvline(target.mean(), color=self.colors['accent'], linestyle='--', linewidth=2, label=f'Mean: ${target.mean():,.0f}')
        axes[0].axvline(target.median(), color=self.colors['secondary'], linestyle='--', linewidth=2, label=f'Median: ${target.median():,.0f}')
        axes[0].set_xlabel('Daily Sales ($)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Sales Distribution', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Box plot
        bp = axes[1].boxplot(target, patch_artist=True)
        bp['boxes'][0].set_facecolor(self.colors['primary'])
        bp['boxes'][0].set_alpha(0.7)
        axes[1].set_ylabel('Daily Sales ($)', fontsize=11)
        axes[1].set_title('Sales Box Plot', fontsize=12, fontweight='bold')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        axes[1].set_xticklabels(['Daily Sales'])
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(target, dist="norm", plot=axes[2])
        axes[2].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        axes[2].get_lines()[0].set_color(self.colors['primary'])
        axes[2].get_lines()[1].set_color(self.colors['accent'])
        
        plt.tight_layout()
        
        path = self.output_dir / 'distribution.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def plot_seasonality(self, figsize: Tuple[int, int] = (14, 10)) -> str:
        """Plot seasonality patterns."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Day of week
        self.df['_dow'] = self.df[self.date_col].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_means = self.df.groupby('_dow')[self.target_col].mean().reindex(day_order)
        
        colors_dow = [self.colors['secondary'] if d in ['Saturday', 'Sunday'] else self.colors['primary'] for d in day_order]
        axes[0, 0].bar(range(7), dow_means.values, color=colors_dow, edgecolor='white')
        axes[0, 0].set_xticks(range(7))
        axes[0, 0].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        axes[0, 0].set_xlabel('Day of Week', fontsize=11)
        axes[0, 0].set_ylabel('Average Sales ($)', fontsize=11)
        axes[0, 0].set_title('Sales by Day of Week', fontsize=12, fontweight='bold')
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Month
        self.df['_month'] = self.df[self.date_col].dt.month
        month_means = self.df.groupby('_month')[self.target_col].mean()
        
        axes[0, 1].bar(month_means.index, month_means.values, color=self.colors['primary'], edgecolor='white')
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[0, 1].set_xlabel('Month', fontsize=11)
        axes[0, 1].set_ylabel('Average Sales ($)', fontsize=11)
        axes[0, 1].set_title('Sales by Month', fontsize=12, fontweight='bold')
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Quarter
        self.df['_quarter'] = self.df[self.date_col].dt.quarter
        quarter_means = self.df.groupby('_quarter')[self.target_col].mean()
        
        axes[1, 0].bar(quarter_means.index, quarter_means.values, color=self.colors['accent'], edgecolor='white')
        axes[1, 0].set_xticks([1, 2, 3, 4])
        axes[1, 0].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        axes[1, 0].set_xlabel('Quarter', fontsize=11)
        axes[1, 0].set_ylabel('Average Sales ($)', fontsize=11)
        axes[1, 0].set_title('Sales by Quarter', fontsize=12, fontweight='bold')
        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Autocorrelation
        from pandas.plotting import autocorrelation_plot
        autocorrelation_plot(self.df[self.target_col], ax=axes[1, 1], color=self.colors['primary'])
        axes[1, 1].set_title('Autocorrelation', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Lag (Days)', fontsize=11)
        axes[1, 1].set_xlim(0, 60)
        
        # Cleanup
        self.df.drop(['_dow', '_month', '_quarter'], axis=1, inplace=True, errors='ignore')
        
        plt.tight_layout()
        
        path = self.output_dir / 'seasonality.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def plot_trend_analysis(self, figsize: Tuple[int, int] = (14, 10)) -> str:
        """Plot trend analysis."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        target = self.df[self.target_col].values
        dates = self.df[self.date_col]
        x = np.arange(len(target))
        
        # Linear trend
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, target)
        trend_line = slope * x + intercept
        
        axes[0, 0].scatter(dates, target, alpha=0.5, s=10, color=self.colors['primary'], label='Actual')
        axes[0, 0].plot(dates, trend_line, color=self.colors['accent'], linewidth=2, label=f'Trend (R²={r_value**2:.3f})')
        axes[0, 0].set_xlabel('Date', fontsize=11)
        axes[0, 0].set_ylabel('Daily Sales ($)', fontsize=11)
        axes[0, 0].set_title('Linear Trend Analysis', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Monthly trend
        self.df['_month'] = self.df[self.date_col].dt.to_period('M')
        monthly = self.df.groupby('_month')[self.target_col].mean()
        
        axes[0, 1].plot(range(len(monthly)), monthly.values, marker='o', color=self.colors['primary'], linewidth=2)
        axes[0, 1].set_xlabel('Month', fontsize=11)
        axes[0, 1].set_ylabel('Average Sales ($)', fontsize=11)
        axes[0, 1].set_title('Monthly Average Trend', fontsize=12, fontweight='bold')
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Set x-ticks for monthly trend
        tick_positions = list(range(0, len(monthly), 3))
        tick_labels = [str(monthly.index[i]) for i in tick_positions]
        axes[0, 1].set_xticks(tick_positions)
        axes[0, 1].set_xticklabels(tick_labels, rotation=45, ha='right')
        
        self.df.drop('_month', axis=1, inplace=True, errors='ignore')
        
        # Residuals
        residuals = target - trend_line
        axes[1, 0].scatter(dates, residuals, alpha=0.5, s=10, color=self.colors['primary'])
        axes[1, 0].axhline(y=0, color=self.colors['accent'], linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Date', fontsize=11)
        axes[1, 0].set_ylabel('Residual ($)', fontsize=11)
        axes[1, 0].set_title('Trend Residuals', fontsize=12, fontweight='bold')
        axes[1, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Residual distribution
        axes[1, 1].hist(residuals, bins=30, color=self.colors['primary'], edgecolor='white', alpha=0.7)
        axes[1, 1].axvline(x=0, color=self.colors['accent'], linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Residual ($)', fontsize=11)
        axes[1, 1].set_ylabel('Frequency', fontsize=11)
        axes[1, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        path = self.output_dir / 'trend_analysis.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (10, 8)) -> str:
        """Plot correlation heatmap."""
        # Select numeric columns
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            # Not enough numeric columns for correlation
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'Not enough numeric columns\nfor correlation analysis',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        else:
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                ax=ax,
                cbar_kws={'shrink': 0.8}
            )
            
            ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        path = self.output_dir / 'correlation_heatmap.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def plot_boxplots(self, figsize: Tuple[int, int] = (14, 5)) -> str:
        """Plot boxplots by different categories."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # By day of week
        self.df['_dow'] = self.df[self.date_col].dt.dayofweek
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        bp1 = axes[0].boxplot(
            [self.df[self.df['_dow'] == i][self.target_col].values for i in range(7)],
            patch_artist=True
        )
        for i, box in enumerate(bp1['boxes']):
            box.set_facecolor(self.colors['secondary'] if i >= 5 else self.colors['primary'])
            box.set_alpha(0.7)
        axes[0].set_xticklabels(day_labels)
        axes[0].set_xlabel('Day of Week', fontsize=11)
        axes[0].set_ylabel('Daily Sales ($)', fontsize=11)
        axes[0].set_title('Sales Distribution by Day', fontsize=12, fontweight='bold')
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # By quarter
        self.df['_quarter'] = self.df[self.date_col].dt.quarter
        
        bp2 = axes[1].boxplot(
            [self.df[self.df['_quarter'] == q][self.target_col].values for q in [1, 2, 3, 4]],
            patch_artist=True
        )
        for box in bp2['boxes']:
            box.set_facecolor(self.colors['accent'])
            box.set_alpha(0.7)
        axes[1].set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
        axes[1].set_xlabel('Quarter', fontsize=11)
        axes[1].set_ylabel('Daily Sales ($)', fontsize=11)
        axes[1].set_title('Sales Distribution by Quarter', fontsize=12, fontweight='bold')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # By holiday
        if 'is_holiday' in self.df.columns:
            bp3 = axes[2].boxplot(
                [self.df[self.df['is_holiday'] == 0][self.target_col].values,
                 self.df[self.df['is_holiday'] == 1][self.target_col].values],
                patch_artist=True
            )
            bp3['boxes'][0].set_facecolor(self.colors['primary'])
            bp3['boxes'][1].set_facecolor(self.colors['secondary'])
            for box in bp3['boxes']:
                box.set_alpha(0.7)
            axes[2].set_xticklabels(['Non-Holiday', 'Holiday'])
            axes[2].set_xlabel('Holiday Status', fontsize=11)
            axes[2].set_ylabel('Daily Sales ($)', fontsize=11)
            axes[2].set_title('Sales: Holiday vs Non-Holiday', fontsize=12, fontweight='bold')
            axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        else:
            axes[2].text(0.5, 0.5, 'No holiday data', ha='center', va='center')
            axes[2].axis('off')
        
        # Cleanup
        self.df.drop(['_dow', '_quarter'], axis=1, inplace=True, errors='ignore')
        
        plt.tight_layout()
        
        path = self.output_dir / 'boxplots.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)
    
    def plot_summary_dashboard(self, figsize: Tuple[int, int] = (16, 12)) -> str:
        """Create a summary dashboard with key visualizations."""
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Time series (top, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.df[self.date_col], self.df[self.target_col], 
                color=self.colors['primary'], linewidth=0.8, alpha=0.7)
        ma30 = self.df[self.target_col].rolling(window=30).mean()
        ax1.plot(self.df[self.date_col], ma30, color=self.colors['accent'], linewidth=2)
        ax1.set_title('Daily Sales Time Series', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sales ($)')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(self.df[self.target_col], bins=25, color=self.colors['primary'], 
                edgecolor='white', alpha=0.7)
        ax2.axvline(self.df[self.target_col].mean(), color=self.colors['accent'], 
                   linestyle='--', linewidth=2)
        ax2.set_title('Sales Distribution', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Daily Sales ($)')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # Day of week (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        self.df['_dow'] = self.df[self.date_col].dt.day_name()
        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_means = self.df.groupby('_dow')[self.target_col].mean()
        dow_means = dow_means.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        colors_dow = [self.colors['secondary'] if i >= 5 else self.colors['primary'] for i in range(7)]
        ax3.bar(day_order, dow_means.values, color=colors_dow)
        ax3.set_title('Sales by Day of Week', fontsize=12, fontweight='bold')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        self.df.drop('_dow', axis=1, inplace=True, errors='ignore')
        
        # Monthly trend (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        self.df['_month'] = self.df[self.date_col].dt.to_period('M')
        monthly = self.df.groupby('_month')[self.target_col].mean()
        ax4.plot(range(len(monthly)), monthly.values, marker='o', 
                color=self.colors['primary'], linewidth=2, markersize=4)
        ax4.set_title('Monthly Average Trend', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Month')
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        self.df.drop('_month', axis=1, inplace=True, errors='ignore')
        
        # Key stats (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        stats_text = [
            f"Total Records: {len(self.df):,}",
            f"Date Range: {self.df[self.date_col].min().strftime('%Y-%m-%d')}",
            f"         to {self.df[self.date_col].max().strftime('%Y-%m-%d')}",
            "",
            f"Mean Sales: ${self.df[self.target_col].mean():,.0f}",
            f"Std Dev: ${self.df[self.target_col].std():,.0f}",
            f"Min: ${self.df[self.target_col].min():,.0f}",
            f"Max: ${self.df[self.target_col].max():,.0f}",
        ]
        ax5.text(0.1, 0.9, '\n'.join(stats_text), transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax5.set_title('Key Statistics', fontsize=12, fontweight='bold')
        
        # Correlation with marketing spend (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        if 'marketing_spend' in self.df.columns:
            ax6.scatter(self.df['marketing_spend'], self.df[self.target_col],
                       alpha=0.5, s=15, color=self.colors['primary'])
            corr = self.df['marketing_spend'].corr(self.df[self.target_col])
            ax6.set_title(f'Sales vs Marketing (r={corr:.3f})', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Marketing Spend ($)')
            ax6.set_ylabel('Daily Sales ($)')
            ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        else:
            ax6.text(0.5, 0.5, 'No marketing data', ha='center', va='center')
            ax6.axis('off')
        
        # Autocorrelation (bottom center)
        ax7 = fig.add_subplot(gs[2, 1])
        lags = range(1, 31)
        autocorrs = [self.df[self.target_col].autocorr(lag=l) for l in lags]
        ax7.bar(lags, autocorrs, color=self.colors['primary'], alpha=0.7)
        ax7.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax7.set_title('Autocorrelation (Lags 1-30)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Lag (Days)')
        ax7.set_ylabel('Correlation')
        
        # Trend info (bottom right)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        # Calculate trend
        from scipy import stats
        x = np.arange(len(self.df))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, self.df[self.target_col])
        
        mid = len(self.df) // 2
        first_half = self.df[self.target_col].iloc[:mid].mean()
        second_half = self.df[self.target_col].iloc[mid:].mean()
        growth = ((second_half - first_half) / first_half) * 100
        
        trend_text = [
            "TREND ANALYSIS",
            "=" * 25,
            f"Direction: {'UPWARD' if slope > 0 else 'DOWNWARD'}",
            f"Daily Change: ${slope:,.2f}",
            f"R-squared: {r_value**2:.4f}",
            "",
            f"First Half Avg: ${first_half:,.0f}",
            f"Second Half Avg: ${second_half:,.0f}",
            f"Growth Rate: {growth:.1f}%"
        ]
        ax8.text(0.1, 0.9, '\n'.join(trend_text), transform=ax8.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax8.set_title('Trend Summary', fontsize=12, fontweight='bold')
        
        plt.suptitle('Sales Data - EDA Summary Dashboard', fontsize=16, fontweight='bold', y=1.02)
        
        path = self.output_dir / 'summary_dashboard.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(path)

