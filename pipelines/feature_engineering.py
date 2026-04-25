import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class RULFeatureEngineer:
    """
    Feature Engineering for RUL Prediction.
    
    Creates features from raw sensor data including:
    - Rolling statistics (mean, std, min, max)
    - Trend features
    - Normalized RUL labels
    """
    
    def __init__(
        self,
        max_rul: int = 125,
        sequence_length: int = 30,
        sensor_columns: List[str] = None
    ):
        self.max_rul = max_rul
        self.sequence_length = sequence_length
        self.sensor_columns = sensor_columns
        self.scaler_params = {}
    
    def compute_rul(self, df: pd.DataFrame, unit_col: str = 'unit_number') -> pd.DataFrame:
        """
        Compute RUL for each engine unit.
        
        RUL = remaining cycles until failure
        Capped at max_rul (typical: 125 cycles)
        """
        df = df.copy()
        df['max_cycle'] = df.groupby(unit_col)['time_cycles'].transform('max')
        df['RUL'] = df['max_cycle'] - df['time_cycles']
        df['RUL'] = df['RUL'].clip(upper=self.max_rul)
        
        df = df.drop('max_cycle', axis=1)
        return df
    
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        unit_col: str = 'unit_number',
        windows: List[int] = [5, 10, 20]
    ) -> pd.DataFrame:
        """Add rolling window statistics."""
        df = df.copy()
        
        if self.sensor_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.sensor_columns = [c for c in numeric_cols 
                                   if c not in [unit_col, 'time_cycles', 'RUL', 'op_setting_1', 'op_setting_2', 'op_setting_3']]
        
        for window in windows:
            for sensor in self.sensor_columns[:5]:
                if sensor in df.columns:
                    df[f'{sensor}_roll_mean_{window}'] = df.groupby(unit_col)[sensor].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    df[f'{sensor}_roll_std_{window}'] = df.groupby(unit_col)[sensor].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    ).fillna(0)
        
        return df
    
    def add_trend_features(
        self,
        df: pd.DataFrame,
        unit_col: str = 'unit_number'
    ) -> pd.DataFrame:
        """Add trend/difference features."""
        df = df.copy()
        
        if self.sensor_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.sensor_columns = [c for c in numeric_cols 
                                   if c not in [unit_col, 'time_cycles', 'RUL']]
        
        for sensor in self.sensor_columns[:5]:
            if sensor in df.columns:
                df[f'{sensor}_diff'] = df.groupby(unit_col)[sensor].diff().fillna(0)
        
        return df
    
    def normalize_by_unit(
        self,
        df: pd.DataFrame,
        unit_col: str = 'unit_number'
    ) -> pd.DataFrame:
        """Normalize sensor values per unit (min-max per unit)."""
        df = df.copy()
        
        if self.sensor_columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.sensor_columns = [c for c in numeric_cols 
                                   if c not in [unit_col, 'time_cycles', 'RUL']]
        
        for sensor in self.sensor_columns:
            if sensor in df.columns:
                min_val = df.groupby(unit_col)[sensor].transform('min')
                max_val = df.groupby(unit_col)[sensor].transform('max')
                range_val = max_val - min_val
                range_val = range_val.replace(0, 1)
                df[f'{sensor}_norm'] = (df[sensor] - min_val) / range_val
        
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        unit_col: str = 'unit_number',
        target_col: str = 'RUL'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding window sequences for LSTM."""
        X, y = [], []
        
        for unit in df[unit_col].unique():
            unit_data = df[df[unit_col] == unit]
            
            unit_features = unit_data[feature_cols].values
            unit_targets = unit_data[target_col].values
            
            for i in range(len(unit_data) - self.sequence_length + 1):
                X.append(unit_features[i:i + self.sequence_length])
                y.append(unit_targets[i + self.sequence_length - 1])
        
        return np.array(X), np.array(y)
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        unit_col: str = 'unit_number'
    ) -> pd.DataFrame:
        """Fit on data and transform."""
        df = self.compute_rul(df, unit_col)
        df = self.add_rolling_features(df, unit_col)
        df = self.add_trend_features(df, unit_col)
        return df
    
    def transform(
        self,
        df: pd.DataFrame,
        unit_col: str = 'unit_number'
    ) -> pd.DataFrame:
        """Transform new data using fitted parameters."""
        df = self.compute_rul(df, unit_col)
        df = self.add_rolling_features(df, unit_col)
        df = self.add_trend_features(df, unit_col)
        return df


def load_cmapss_data(
    data_dir: str = 'data/raw',
    subset: str = 'FD001'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load C-MAPSS dataset.
    
    Returns train, test, and RUL (for test) dataframes.
    """
    data_path = Path(data_dir)
    
    cols = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    cols.extend(sensor_cols)
    
    train_file = data_path / f'train_{subset}.txt'
    test_file = data_path / f'test_{subset}.txt'
    rul_file = data_path / f'RUL_{subset}.txt'
    
    if not train_file.exists():
        raise FileNotFoundError(f"Data not found at {data_path}")
    
    train_df = pd.read_csv(train_file, sep=r'\s+', header=None, names=cols)
    test_df = pd.read_csv(test_file, sep=r'\s+', header=None, names=cols)
    
    rul_df = pd.read_csv(rul_file, sep=r'\s+', header=None, names=['RUL'])
    rul_df['unit_number'] = range(1, len(rul_df) + 1)
    
    return train_df, test_df, rul_df


def prepare_data(
    data_dir: str = 'data/raw',
    subset: str = 'FD001',
    max_rul: int = 125,
    sequence_length: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare C-MAPSS data for training.
    
    Returns: X_train, y_train, X_test, y_test, feature_cols
    """
    train_df, test_df, rul_df = load_cmapss_data(data_dir, subset)
    
    engineer = RULFeatureEngineer(max_rul=max_rul, sequence_length=sequence_length)
    
    train_df = engineer.fit_transform(train_df)
    
    test_df = engineer.transform(test_df)
    
    if 'RUL' not in test_df.columns:
        test_df = test_df.merge(rul_df, on='unit_number', how='left')
        test_df['RUL'] = test_df['RUL'].fillna(0)
    
    feature_cols = [c for c in train_df.columns 
                    if c not in ['unit_number', 'time_cycles', 'RUL']]
    
    X_train, y_train = engineer.create_sequences(
        train_df, feature_cols, 'unit_number', 'RUL'
    )
    
    test_engineer = RULFeatureEngineer(max_rul=max_rul, sequence_length=sequence_length)
    X_test, y_test = test_engineer.create_sequences(
        test_df, feature_cols, 'unit_number', 'RUL'
    )
    
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    
    return X_train, y_train, X_test, y_test, feature_cols


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, features = prepare_data()
    print(f"Train shape: {X_train.shape}, {y_train.shape}")
    print(f"Test shape: {X_test.shape}, {y_test.shape}")
    print(f"Features: {len(features)}")