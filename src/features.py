"""
Feature Engineering for Volatility Prediction
Creates technical indicators, volatility features, and ML-ready dataset
"""

import pandas as pd
import numpy as np
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VolatilityFeatures:
    def __init__(self, df):
        self.df = df.copy()
        if 'Date' in df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
    
    def add_technical_indicators(self):
        """Add technical analysis indicators"""
        df = self.df
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['Price'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['Price'].ewm(span=window, adjust=False).mean()
        
        # Price distance from moving averages
        df['price_dist_sma20'] = (df['Price'] - df['sma_20']) / df['sma_20']
        df['price_dist_sma50'] = (df['Price'] - df['sma_50']) / df['sma_50']
        
        # MACD
        ema_12 = df['Price'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Price'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['sma_20']
        bb_std = df['Price'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Price'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Average True Range (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Price'].shift())
        low_close = np.abs(df['Low'] - df['Price'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr_14'] = true_range.rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['Price']
        
        # Volume indicators
        if 'Vol' in df.columns:
            df['volume_sma_20'] = df['Vol'].rolling(20).mean()
            df['volume_ratio'] = df['Vol'] / df['volume_sma_20']
        
        return df
    
    def add_volatility_features(self):
        """Add volatility-specific features"""
        df = self.df
        returns = df['log_return']
        
        # Historical volatility (different windows, annualized)
        for window in [5, 10, 21, 63]:
            df[f'hist_vol_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Parkinson volatility (using high-low, more efficient)
        hl_ratio = df['High'] / df['Low']
        # Handle potential zeros or negatives in High/Low
        hl_ratio = hl_ratio.replace([np.inf, -np.inf], np.nan)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(hl_ratio) ** 2).rolling(5).mean()
        ) * np.sqrt(252)
        
        # Garman-Klass volatility (most efficient, uses OHLC)
        log_hl = np.log(df['High'] / df['Low']) ** 2
        log_co = np.log(df['Price'] / df['Open']) ** 2
        df['garman_klass_vol'] = np.sqrt(
            0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        ).rolling(5).mean() * np.sqrt(252)
        
        # Volatility of volatility
        df['vol_of_vol'] = df['hist_vol_21d'].rolling(21).std()
        
        # Volatility regime (high/low based on median)
        vol_median = df['hist_vol_21d'].median()
        df['high_vol_regime'] = (df['hist_vol_21d'] > vol_median).astype(int)
        
        # Volatility trend (increasing/decreasing)
        df['vol_trend'] = np.where(
            df['hist_vol_21d'] > df['hist_vol_21d'].shift(5), 1, -1
        )
        
        return df
    
    def add_lag_features(self, lags=[1, 2, 3, 5, 10]):
        """Add lagged features for ML models"""
        df = self.df
        
        # Key features to lag
        features_to_lag = ['log_return', 'squared_return', 'hist_vol_21d', 
                          'rsi', 'macd', 'atr_14', 'bb_width']
        
        for feature in features_to_lag:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        # Lagged returns for volatility clustering capture
        for lag in [1, 2, 3, 5]:
            df[f'abs_return_lag_{lag}'] = df['abs_return'].shift(lag)
            df[f'squared_return_lag_{lag}'] = df['squared_return'].shift(lag)
        
        return df
    
    def add_calendar_features(self):
        """Add calendar/time features"""
        df = self.df
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        df['year'] = df['Date'].dt.year
        df['is_month_start'] = df['Date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['Date'].dt.is_month_end.astype(int)
        
        # Day of month effect
        df['day_of_month'] = df['Date'].dt.day
        
        # Create dummy variables for day of week (Monday effect, etc.)
        for day in range(5):
            df[f'dow_{day}'] = (df['day_of_week'] == day).astype(int)
        
        return df
    
    def add_market_regime_features(self):
        """Add market regime indicators"""
        df = self.df
        
        # Trend direction based on moving averages
        df['trend'] = np.where(df['Price'] > df['sma_50'], 1, 
                              np.where(df['Price'] < df['sma_50'], -1, 0))
        
        # Volatility clustering indicator (ARCH effect)
        df['high_prev_vol'] = (df['squared_return'].shift(1) > 
                               df['squared_return'].shift(1).quantile(0.8)).astype(int)
        
        # Jump detection (extreme returns)
        df['return_zscore'] = np.abs(stats.zscore(df['log_return'].fillna(0)))
        df['is_jump'] = (df['return_zscore'] > 3).astype(int)
        
        # Drawdown features
        rolling_max = df['Price'].expanding().max()
        df['drawdown'] = (df['Price'] - rolling_max) / rolling_max
        df['max_drawdown_21d'] = df['drawdown'].rolling(21).min()
        
        return df
    
    def create_all_features(self):
        """Create complete feature set"""
        logger.info("Adding technical indicators...")
        self.df = self.add_technical_indicators()
        
        logger.info("Adding volatility features...")
        self.df = self.add_volatility_features()
        
        logger.info("Adding lag features...")
        self.df = self.add_lag_features()
        
        logger.info("Adding calendar features...")
        self.df = self.add_calendar_features()
        
        logger.info("Adding market regime features...")
        self.df = self.add_market_regime_features()
        
        # Define target variables (what we want to predict)
        # Primary target: next-day realized volatility
        self.df['target_vol_1d'] = self.df['hist_vol_21d'].shift(-1)
        
        # Alternative targets
        self.df['target_squared_ret'] = self.df['squared_return'].shift(-1)
        self.df['target_abs_ret'] = self.df['abs_return'].shift(-1)
        
        # Multi-horizon targets
        self.df['target_vol_5d'] = self.df['log_return'].rolling(5).std().shift(-5) * np.sqrt(252)
        
        logger.info(f"Feature engineering complete. Total features: {self.df.shape[1]}")
        
        return self.df.dropna()


def prepare_ml_dataset(df, target_col='target_vol_1d', test_size=0.2, 
                       min_train_size=500):
    """
    Prepare train/test splits for ML models
    Time-series aware split (no random shuffling)
    """
    # Select feature columns (exclude non-feature columns)
    exclude_cols = ['Date', 'stock', 'target_vol_1d', 'target_vol_5d',
                   'target_squared_ret', 'target_abs_ret',
                   'Price', 'Open', 'High', 'Low', 'Vol', 'Change_%',
                   'simple_return', 'log_return', 'squared_return', 'abs_return']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Remove any remaining non-numeric columns
    feature_cols = [col for col in feature_cols 
                   if pd.api.types.is_numeric_dtype(df[col])]
    
    X = df[feature_cols].ffill().fillna(0)
    y = df[target_col]
    
    n_samples = len(df)
    
    # Calculate split index based on test_size
    split_idx = int(n_samples * (1 - test_size))
    
    # Ensure minimum training size only if we have enough data
    if split_idx < min_train_size:
        if n_samples > min_train_size:
            # If we have enough total samples, enforce min_train_size
            split_idx = min_train_size
        else:
            # If dataset is smaller than min_train_size, use 80/20 split
            # or at least keep 10 samples for test if possible
            split_idx = max(int(n_samples * 0.8), n_samples - 10)
            logger.warning(f"Dataset size ({n_samples}) < min_train_size ({min_train_size}). "
                          f"Using adjusted split: {split_idx} train, {n_samples - split_idx} test")
    
    # Ensure we don't exceed dataset bounds
    split_idx = min(split_idx, n_samples - 1)
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    dates_train = df['Date'].iloc[:split_idx] if 'Date' in df.columns else None
    dates_test = df['Date'].iloc[split_idx:] if 'Date' in df.columns else None
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Features: {len(feature_cols)}")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'feature_cols': feature_cols,
        'dates_train': dates_train, 'dates_test': dates_test
    }


def create_panel_features(panel_df):
    """Create features for panel data (multiple stocks)"""
    all_features = []
    
    for stock in panel_df['stock'].unique():
        stock_df = panel_df[panel_df['stock'] == stock].copy()
        
        logger.info(f"Processing {stock}...")
        feature_eng = VolatilityFeatures(stock_df)
        stock_features = feature_eng.create_all_features()
        all_features.append(stock_features)
    
    combined = pd.concat(all_features, ignore_index=True)
    return combined


# Example usage
if __name__ == "__main__":
    import os
    import glob
    
    # Process all stocks in the processed folder
    processed_files = glob.glob("data/processed/*_processed.csv")
    
    for file_path in processed_files:
        # Extract stock name from filename
        stock_name = os.path.basename(file_path).replace('_processed.csv', '')
        
        print(f"\nProcessing {stock_name}...")
        df = pd.read_csv(file_path, parse_dates=['Date'])
        
        feature_eng = VolatilityFeatures(df)
        df_features = feature_eng.create_all_features()
        
        print(f"Shape after features: {df_features.shape}")
        print(f"Columns: {list(df_features.columns)}")
        
        # Save to features folder
        output_path = f"data/features/{stock_name}_features.csv"
        os.makedirs("data/features", exist_ok=True)
        df_features.to_csv(output_path, index=False)
        print(f"Saved to: {output_path}")
        
        # Prepare ML dataset
        dataset = prepare_ml_dataset(df_features)
        print(f"Train: {len(dataset['X_train'])}, Test: {len(dataset['X_test'])}")