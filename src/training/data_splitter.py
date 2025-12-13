"""
Time Series Data Splitting utilities with Walk-Forward Validation.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Generator, Optional
from dataclasses import dataclass
import logging

from ..utils.logger import setup_logger


@dataclass
class SplitInfo:
    """Information about a train/test split."""
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int


class TimeSeriesSplitter:
    """
    Time Series Cross-Validation with Walk-Forward approach.
    
    Ensures temporal ordering is preserved - no data leakage from future.
    
    Methods:
    - simple_split: Single train/test split
    - walk_forward_split: Multiple expanding window splits
    - sliding_window_split: Multiple fixed-size window splits
    """
    
    def __init__(self, date_column: str = 'date'):
        """
        Initialize TimeSeriesSplitter.
        
        Args:
            date_column: Name of the date column
        """
        self.date_column = date_column
        self.logger = setup_logger(f"{__name__}.TimeSeriesSplitter")
    
    def simple_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame, SplitInfo]:
        """
        Simple temporal train/test split.
        
        Args:
            df: DataFrame sorted by date
            train_ratio: Proportion for training (default: 0.8)
            
        Returns:
            Tuple of (train_df, test_df, split_info)
        """
        df = df.sort_values(self.date_column).reset_index(drop=True)
        
        split_idx = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        split_info = SplitInfo(
            fold=0,
            train_start=str(train_df[self.date_column].min().date()),
            train_end=str(train_df[self.date_column].max().date()),
            test_start=str(test_df[self.date_column].min().date()),
            test_end=str(test_df[self.date_column].max().date()),
            train_size=len(train_df),
            test_size=len(test_df)
        )
        
        self.logger.info(f"Simple split: Train {split_info.train_size} rows, Test {split_info.test_size} rows")
        
        return train_df, test_df, split_info
    
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        min_train_size: Optional[int] = None
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, SplitInfo], None, None]:
        """
        Walk-Forward (Expanding Window) Cross-Validation.
        
        Training window expands with each fold while test window moves forward.
        
        Args:
            df: DataFrame sorted by date
            n_splits: Number of splits/folds
            test_size: Fixed test size (default: auto-calculated)
            min_train_size: Minimum training size (default: 50% of data)
            
        Yields:
            Tuple of (train_df, test_df, split_info) for each fold
        """
        df = df.sort_values(self.date_column).reset_index(drop=True)
        n_samples = len(df)
        
        if min_train_size is None:
            min_train_size = n_samples // 2
        
        if test_size is None:
            test_size = (n_samples - min_train_size) // n_splits
        
        self.logger.info(f"Walk-forward CV: {n_splits} folds, test_size={test_size}, min_train={min_train_size}")
        
        for fold in range(n_splits):
            # Expanding training window
            train_end_idx = min_train_size + fold * test_size
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_size, n_samples)
            
            if test_end_idx <= test_start_idx:
                break
            
            train_df = df.iloc[:train_end_idx].copy()
            test_df = df.iloc[test_start_idx:test_end_idx].copy()
            
            split_info = SplitInfo(
                fold=fold,
                train_start=str(train_df[self.date_column].min().date()),
                train_end=str(train_df[self.date_column].max().date()),
                test_start=str(test_df[self.date_column].min().date()),
                test_end=str(test_df[self.date_column].max().date()),
                train_size=len(train_df),
                test_size=len(test_df)
            )
            
            self.logger.debug(f"Fold {fold}: Train {split_info.train_size}, Test {split_info.test_size}")
            
            yield train_df, test_df, split_info
    
    def sliding_window_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, SplitInfo], None, None]:
        """
        Sliding Window Cross-Validation.
        
        Fixed-size training window slides forward with each fold.
        
        Args:
            df: DataFrame sorted by date
            n_splits: Number of splits/folds
            train_size: Fixed training window size
            test_size: Fixed test size
            
        Yields:
            Tuple of (train_df, test_df, split_info) for each fold
        """
        df = df.sort_values(self.date_column).reset_index(drop=True)
        n_samples = len(df)
        
        if train_size is None:
            train_size = n_samples // 2
        
        if test_size is None:
            test_size = (n_samples - train_size) // n_splits
        
        step_size = test_size
        
        self.logger.info(f"Sliding window CV: {n_splits} folds, train_size={train_size}, test_size={test_size}")
        
        for fold in range(n_splits):
            train_start_idx = fold * step_size
            train_end_idx = train_start_idx + train_size
            test_start_idx = train_end_idx
            test_end_idx = min(test_start_idx + test_size, n_samples)
            
            if test_end_idx <= test_start_idx or train_end_idx > n_samples:
                break
            
            train_df = df.iloc[train_start_idx:train_end_idx].copy()
            test_df = df.iloc[test_start_idx:test_end_idx].copy()
            
            split_info = SplitInfo(
                fold=fold,
                train_start=str(train_df[self.date_column].min().date()),
                train_end=str(train_df[self.date_column].max().date()),
                test_start=str(test_df[self.date_column].min().date()),
                test_end=str(test_df[self.date_column].max().date()),
                train_size=len(train_df),
                test_size=len(test_df)
            )
            
            self.logger.debug(f"Fold {fold}: Train {split_info.train_size}, Test {split_info.test_size}")
            
            yield train_df, test_df, split_info
    
    def get_feature_target_split(
        self,
        df: pd.DataFrame,
        target_column: str = 'daily_sales',
        feature_columns: Optional[List[str]] = None,
        exclude_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and target (y).
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            feature_columns: Specific columns to use as features (optional)
            exclude_columns: Columns to exclude from features (optional)
            
        Returns:
            Tuple of (X, y)
        """
        if exclude_columns is None:
            exclude_columns = [self.date_column, target_column, 'product_category', 'day_of_week']
        else:
            exclude_columns = list(exclude_columns) + [self.date_column, target_column]
        
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c not in exclude_columns]
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        return X, y

