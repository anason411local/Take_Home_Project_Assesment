"""
Sales Data Analyzer - Statistical analysis for EDA.

Provides comprehensive statistical analysis of sales data including:
- Basic statistics
- Trend analysis
- Seasonality detection
- Correlation analysis
- Outlier detection
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    basic_stats: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    seasonality: Dict[str, Any]
    correlations: Dict[str, float]
    outliers: Dict[str, Any]
    data_quality: Dict[str, Any]


class SalesDataAnalyzer:
    """
    Comprehensive sales data analyzer.
    
    Performs statistical analysis to understand data patterns
    before model training.
    """
    
    def __init__(self, df: pd.DataFrame, date_col: str = 'date', target_col: str = 'daily_sales'):
        """
        Initialize analyzer.
        
        Args:
            df: Sales data DataFrame
            date_col: Name of date column
            target_col: Name of target column (sales)
        """
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        
        # Ensure date is datetime
        if date_col in self.df.columns:
            self.df[date_col] = pd.to_datetime(self.df[date_col])
            self.df = self.df.sort_values(date_col).reset_index(drop=True)
    
    def analyze_all(self) -> AnalysisResult:
        """Run all analyses and return results."""
        return AnalysisResult(
            basic_stats=self.get_basic_stats(),
            trend_analysis=self.analyze_trend(),
            seasonality=self.analyze_seasonality(),
            correlations=self.analyze_correlations(),
            outliers=self.detect_outliers(),
            data_quality=self.check_data_quality()
        )
    
    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics of the data."""
        target = self.df[self.target_col]
        
        stats_dict = {
            'count': len(self.df),
            'mean': target.mean(),
            'std': target.std(),
            'min': target.min(),
            'max': target.max(),
            'median': target.median(),
            'q1': target.quantile(0.25),
            'q3': target.quantile(0.75),
            'iqr': target.quantile(0.75) - target.quantile(0.25),
            'skewness': target.skew(),
            'kurtosis': target.kurtosis(),
            'cv': (target.std() / target.mean()) * 100  # Coefficient of variation
        }
        
        # Date range
        if self.date_col in self.df.columns:
            stats_dict['date_start'] = self.df[self.date_col].min()
            stats_dict['date_end'] = self.df[self.date_col].max()
            stats_dict['date_range_days'] = (self.df[self.date_col].max() - self.df[self.date_col].min()).days + 1
        
        return stats_dict
    
    def analyze_trend(self) -> Dict[str, Any]:
        """Analyze overall trend in the data."""
        target = self.df[self.target_col].values
        x = np.arange(len(target))
        
        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, target)
        
        # Calculate trend strength
        trend_line = slope * x + intercept
        trend_strength = r_value ** 2  # R-squared
        
        # Monthly averages for trend visualization
        monthly_avg = None
        if self.date_col in self.df.columns:
            self.df['_month'] = self.df[self.date_col].dt.to_period('M')
            monthly_avg = self.df.groupby('_month')[self.target_col].mean().to_dict()
            monthly_avg = {str(k): v for k, v in monthly_avg.items()}
            self.df.drop('_month', axis=1, inplace=True)
        
        # First half vs second half comparison
        mid_point = len(target) // 2
        first_half_avg = target[:mid_point].mean()
        second_half_avg = target[mid_point:].mean()
        growth_rate = ((second_half_avg - first_half_avg) / first_half_avg) * 100
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': trend_strength,
            'p_value': p_value,
            'trend_direction': 'upward' if slope > 0 else 'downward',
            'daily_change': slope,
            'first_half_avg': first_half_avg,
            'second_half_avg': second_half_avg,
            'growth_rate_percent': growth_rate,
            'monthly_averages': monthly_avg
        }
    
    def analyze_seasonality(self) -> Dict[str, Any]:
        """Analyze seasonality patterns."""
        result = {}
        
        if self.date_col not in self.df.columns:
            return result
        
        # Day of week analysis
        self.df['_dow'] = self.df[self.date_col].dt.day_name()
        dow_stats = self.df.groupby('_dow')[self.target_col].agg(['mean', 'std']).to_dict()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_means = {day: dow_stats['mean'].get(day, 0) for day in day_order}
        
        result['day_of_week'] = {
            'means': dow_means,
            'best_day': max(dow_means, key=dow_means.get),
            'worst_day': min(dow_means, key=dow_means.get),
            'weekend_avg': (dow_means.get('Saturday', 0) + dow_means.get('Sunday', 0)) / 2,
            'weekday_avg': sum(dow_means.get(d, 0) for d in day_order[:5]) / 5
        }
        
        # Month analysis
        self.df['_month'] = self.df[self.date_col].dt.month
        month_stats = self.df.groupby('_month')[self.target_col].mean().to_dict()
        result['monthly'] = {
            'means': month_stats,
            'best_month': max(month_stats, key=month_stats.get),
            'worst_month': min(month_stats, key=month_stats.get)
        }
        
        # Quarter analysis
        self.df['_quarter'] = self.df[self.date_col].dt.quarter
        quarter_stats = self.df.groupby('_quarter')[self.target_col].mean().to_dict()
        result['quarterly'] = {
            'means': quarter_stats,
            'best_quarter': max(quarter_stats, key=quarter_stats.get)
        }
        
        # Holiday effect
        if 'is_holiday' in self.df.columns:
            holiday_avg = self.df[self.df['is_holiday'] == 1][self.target_col].mean()
            non_holiday_avg = self.df[self.df['is_holiday'] == 0][self.target_col].mean()
            result['holiday_effect'] = {
                'holiday_avg': holiday_avg,
                'non_holiday_avg': non_holiday_avg,
                'holiday_lift_percent': ((holiday_avg - non_holiday_avg) / non_holiday_avg) * 100 if non_holiday_avg > 0 else 0
            }
        
        # Cleanup
        self.df.drop(['_dow', '_month', '_quarter'], axis=1, inplace=True, errors='ignore')
        
        # Autocorrelation
        target = self.df[self.target_col]
        result['autocorrelation'] = {
            'lag_1': target.autocorr(lag=1),
            'lag_7': target.autocorr(lag=7),
            'lag_14': target.autocorr(lag=14),
            'lag_30': target.autocorr(lag=30) if len(target) > 30 else None
        }
        
        return result
    
    def analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between features and target."""
        correlations = {}
        target = self.df[self.target_col]
        
        # Numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != self.target_col]
        
        for col in numeric_cols:
            if col in self.df.columns:
                corr = target.corr(self.df[col])
                if not np.isnan(corr):
                    correlations[col] = corr
        
        # Sort by absolute correlation
        correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return correlations
    
    def detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        target = self.df[self.target_col]
        
        # IQR method
        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        iqr_outliers = self.df[(target < lower_bound) | (target > upper_bound)]
        
        # Z-score method
        z_scores = np.abs(stats.zscore(target))
        zscore_outliers = self.df[z_scores > 3]
        
        return {
            'iqr_method': {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'n_outliers': len(iqr_outliers),
                'outlier_percent': (len(iqr_outliers) / len(self.df)) * 100
            },
            'zscore_method': {
                'threshold': 3,
                'n_outliers': len(zscore_outliers),
                'outlier_percent': (len(zscore_outliers) / len(self.df)) * 100
            }
        }
    
    def check_data_quality(self) -> Dict[str, Any]:
        """Check data quality issues."""
        quality = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values': {},
            'duplicates': 0,
            'date_gaps': []
        }
        
        # Missing values
        for col in self.df.columns:
            missing = self.df[col].isnull().sum()
            if missing > 0:
                quality['missing_values'][col] = {
                    'count': missing,
                    'percent': (missing / len(self.df)) * 100
                }
        
        # Duplicate rows
        quality['duplicates'] = self.df.duplicated().sum()
        
        # Date gaps
        if self.date_col in self.df.columns:
            date_diff = self.df[self.date_col].diff().dt.days
            gaps = date_diff[date_diff > 1]
            if len(gaps) > 0:
                quality['date_gaps'] = [
                    {
                        'date': str(self.df.loc[idx, self.date_col].date()),
                        'gap_days': int(gap)
                    }
                    for idx, gap in gaps.items()
                ]
        
        # Data types
        quality['column_types'] = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        
        return quality
    
    def get_summary_report(self) -> str:
        """Generate a text summary report."""
        result = self.analyze_all()
        
        report = []
        report.append("=" * 70)
        report.append("EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 70)
        
        # Basic Stats
        report.append("\n1. BASIC STATISTICS")
        report.append("-" * 40)
        stats = result.basic_stats
        report.append(f"   Total Records: {stats['count']:,}")
        report.append(f"   Date Range: {stats.get('date_start', 'N/A')} to {stats.get('date_end', 'N/A')}")
        report.append(f"   Mean Sales: ${stats['mean']:,.2f}")
        report.append(f"   Std Dev: ${stats['std']:,.2f}")
        report.append(f"   Min: ${stats['min']:,.2f}")
        report.append(f"   Max: ${stats['max']:,.2f}")
        report.append(f"   Coefficient of Variation: {stats['cv']:.2f}%")
        
        # Trend
        report.append("\n2. TREND ANALYSIS")
        report.append("-" * 40)
        trend = result.trend_analysis
        report.append(f"   Trend Direction: {trend['trend_direction'].upper()}")
        report.append(f"   Daily Change: ${trend['daily_change']:,.2f}")
        report.append(f"   R-squared: {trend['r_squared']:.4f}")
        report.append(f"   Growth Rate: {trend['growth_rate_percent']:.1f}%")
        
        # Seasonality
        report.append("\n3. SEASONALITY")
        report.append("-" * 40)
        season = result.seasonality
        if 'day_of_week' in season:
            report.append(f"   Best Day: {season['day_of_week']['best_day']}")
            report.append(f"   Worst Day: {season['day_of_week']['worst_day']}")
            report.append(f"   Weekend vs Weekday: ${season['day_of_week']['weekend_avg']:,.0f} vs ${season['day_of_week']['weekday_avg']:,.0f}")
        if 'autocorrelation' in season:
            report.append(f"   Lag-1 Autocorrelation: {season['autocorrelation']['lag_1']:.4f}")
            report.append(f"   Lag-7 Autocorrelation: {season['autocorrelation']['lag_7']:.4f}")
        
        # Correlations
        report.append("\n4. TOP CORRELATIONS")
        report.append("-" * 40)
        for col, corr in list(result.correlations.items())[:5]:
            report.append(f"   {col}: {corr:.4f}")
        
        # Data Quality
        report.append("\n5. DATA QUALITY")
        report.append("-" * 40)
        quality = result.data_quality
        report.append(f"   Missing Values: {len(quality['missing_values'])} columns with missing data")
        report.append(f"   Duplicates: {quality['duplicates']}")
        report.append(f"   Date Gaps: {len(quality['date_gaps'])}")
        
        # Outliers
        report.append("\n6. OUTLIERS")
        report.append("-" * 40)
        outliers = result.outliers
        report.append(f"   IQR Method: {outliers['iqr_method']['n_outliers']} outliers ({outliers['iqr_method']['outlier_percent']:.1f}%)")
        report.append(f"   Z-Score Method: {outliers['zscore_method']['n_outliers']} outliers ({outliers['zscore_method']['outlier_percent']:.1f}%)")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)

