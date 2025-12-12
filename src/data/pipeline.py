"""
Main data processing pipeline for sales forecasting system.

Orchestrates the complete data processing workflow:
1. Load raw data
2. Validate data quality
3. Clean and preprocess
4. Engineer features
5. Export processed data
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import json
import logging

from .loader import DataLoader
from .validator import DataValidator, ValidationResult
from .cleaner import DataCleaner
from ..features.engineer import FeatureEngineer
from ..utils.config import Config
from ..utils.logger import setup_logger


@dataclass
class PipelineResult:
    """Container for pipeline execution results."""
    success: bool
    data: Optional[pd.DataFrame]
    validation_result: Optional[ValidationResult]
    feature_summary: Optional[Dict]
    rows_processed: int
    rows_dropped: int
    error_message: Optional[str] = None


class DataProcessingPipeline:
    """
    End-to-end data processing pipeline.
    
    Combines data loading, validation, cleaning, and feature engineering
    into a single, configurable pipeline.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the pipeline.
        
        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config or Config()
        self.logger = setup_logger(f"{__name__}.DataProcessingPipeline")
        
        # Initialize components
        self.loader = DataLoader(self.config)
        self.validator = DataValidator(self.config)
        self.cleaner = DataCleaner(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
    
    def run(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        validate: bool = True,
        create_features: bool = True,
        feature_mode: str = 'minimal',
        drop_na: bool = True,
        save_report: bool = True
    ) -> PipelineResult:
        """
        Execute the complete data processing pipeline.
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to save processed data (optional)
            validate: Whether to run validation checks
            create_features: Whether to create engineered features
            feature_mode: 'minimal' (~15 features) or 'full' (~68 features)
            drop_na: Whether to drop rows with NaN values
            save_report: Whether to save processing report
            
        Returns:
            PipelineResult with processed data and metadata
        """
        self.logger.info("="*60)
        self.logger.info("Starting Data Processing Pipeline")
        self.logger.info("="*60)
        
        try:
            # Step 1: Load data
            self.logger.info("Step 1: Loading data...")
            df = self.loader.load_csv(input_path)
            initial_rows = len(df)
            self.logger.info(f"Loaded {initial_rows} rows")
            
            # Step 2: Validate data
            validation_result = None
            if validate:
                self.logger.info("Step 2: Validating data...")
                validation_result = self.validator.validate(df)
                self.logger.info(f"Validation: {'PASSED' if validation_result.is_valid else 'FAILED'}")
                
                if not validation_result.is_valid:
                    self.logger.warning("Data validation failed, but continuing with cleaning...")
            
            # Step 3: Clean data
            self.logger.info("Step 3: Cleaning data...")
            df = self.cleaner.clean(df)
            self.logger.info(f"Cleaning complete. {len(df)} rows remaining")
            
            # Step 4: Feature engineering
            feature_summary = None
            if create_features:
                self.logger.info(f"Step 4: Engineering features (mode={feature_mode})...")
                df = self.feature_engineer.create_features(df, mode=feature_mode)
                feature_summary = self.feature_engineer.get_feature_summary(df)
                self.logger.info(f"Created {feature_summary['total_features']} features")
            
            # Step 5: Handle NaN values from feature engineering
            rows_dropped = 0
            if drop_na and create_features:
                self.logger.info("Step 5: Handling NaN values...")
                df, rows_dropped = self.feature_engineer.drop_na_rows(df)
                self.logger.info(f"Dropped {rows_dropped} rows with NaN values")
            
            # Step 6: Save processed data
            if output_path:
                self.logger.info(f"Step 6: Saving processed data to {output_path}...")
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_path, index=False)
                self.logger.info(f"Saved {len(df)} rows to {output_path}")
            
            # Save processing report
            if save_report and output_path:
                report_path = output_path.with_suffix('.report.json')
                self._save_report(report_path, validation_result, feature_summary, initial_rows, len(df))
            
            self.logger.info("="*60)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
            self.logger.info("="*60)
            
            return PipelineResult(
                success=True,
                data=df,
                validation_result=validation_result,
                feature_summary=feature_summary,
                rows_processed=initial_rows,
                rows_dropped=rows_dropped
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return PipelineResult(
                success=False,
                data=None,
                validation_result=None,
                feature_summary=None,
                rows_processed=0,
                rows_dropped=0,
                error_message=str(e)
            )
    
    def _save_report(
        self,
        report_path: Path,
        validation_result: Optional[ValidationResult],
        feature_summary: Optional[Dict],
        initial_rows: int,
        final_rows: int
    ):
        """Save processing report to JSON file."""
        report = {
            'pipeline_summary': {
                'initial_rows': initial_rows,
                'final_rows': final_rows,
                'rows_removed': initial_rows - final_rows
            }
        }
        
        if validation_result:
            report['validation'] = {
                'is_valid': validation_result.is_valid,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings,
                'statistics': validation_result.statistics
            }
        
        if feature_summary:
            report['features'] = feature_summary
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Saved processing report to {report_path}")
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        report = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'missing_values': {
                'total': df.isnull().sum().sum(),
                'by_column': df.isnull().sum().to_dict()
            },
            'duplicates': {
                'total_duplicate_rows': df.duplicated().sum()
            },
            'column_types': df.dtypes.astype(str).to_dict()
        }
        
        # Add numeric column statistics
        numeric_cols = df.select_dtypes(include=['number']).columns
        report['numeric_statistics'] = {}
        for col in numeric_cols:
            report['numeric_statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'skew': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        return report

