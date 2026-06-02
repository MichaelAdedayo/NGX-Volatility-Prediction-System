"""
FIXED Data Loading and Preprocessing Module for NSE Data
Handles CSV files manually downloaded from investing.com
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_volume(vol_str):
    """Convert volume strings like '0.42K', '100.32K', '1.5M' to numbers"""
    if pd.isna(vol_str) or vol_str == '' or str(vol_str).strip() == '':
        return np.nan
    
    vol_str = str(vol_str).strip().upper()
    vol_str = vol_str.replace(',', '')
    
    if 'K' in vol_str:
        return float(vol_str.replace('K', '')) * 1000
    elif 'M' in vol_str:
        return float(vol_str.replace('M', '')) * 1000000
    elif 'B' in vol_str:
        return float(vol_str.replace('B', '')) * 1000000000
    else:
        try:
            return float(vol_str)
        except:
            return np.nan


class NSEDataLoader:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        
    def load_investing_data(self, filename):
        """
        Load CSV data from investing.com
        Auto-detects separator (tab or comma)
        """
        filepath = os.path.join(self.data_path, "raw", filename)
        
        try:
            # Try reading with different separators
            df = None
            for sep in ['\t', ',', ';']:
                try:
                    df = pd.read_csv(filepath, sep=sep)
                    if df.shape[1] > 1:  # Successfully parsed multiple columns
                        logger.info(f"Loaded {filename} with separator '{sep}': {df.shape}")
                        break
                except:
                    continue
            
            if df is None or df.shape[1] <= 1:
                raise ValueError(f"Could not parse {filename} with any separator")
            
            # Clean column names
            df.columns = [col.strip().replace(' ', '_').replace('.', '') 
                         for col in df.columns]
            
            logger.info(f"Columns found: {list(df.columns)}")
            
            # Parse date (try different formats)
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
            except:
                try:
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                except:
                    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
            
            # Convert numeric columns (handle commas)
            numeric_cols = ['Price', 'Open', 'High', 'Low']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle volume with K/M suffixes
            if 'Vol' in df.columns:
                df['Vol'] = df['Vol'].apply(parse_volume)
            
            # Handle percentage change
            if 'Change_%' in df.columns:
                df['Change_%'] = df['Change_%'].astype(str).str.replace('%', '')
                df['Change_%'] = pd.to_numeric(df['Change_%'], errors='coerce')
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None
    
    def calculate_returns(self, df, price_col='Price'):
        """Calculate various return types"""
        df = df.copy()
        
        # Simple returns
        df['simple_return'] = df[price_col].pct_change()
        
        # Log returns (preferred for volatility analysis)
        df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
        
        # Squared returns (proxy for volatility)
        df['squared_return'] = df['log_return'] ** 2
        
        # Absolute returns
        df['abs_return'] = df['log_return'].abs()
        
        # Realized volatility (rolling window) - annualized
        df['realized_vol_5d'] = df['log_return'].rolling(window=5).std() * np.sqrt(252)
        df['realized_vol_21d'] = df['log_return'].rolling(window=21).std() * np.sqrt(252)
        
        return df
    
    def prepare_multiple_stocks(self, stock_files):
        """Load and combine multiple stocks"""
        combined_data = {}
        
        for stock_name, filename in stock_files.items():
            df = self.load_investing_data(filename)
            if df is not None:
                df = self.calculate_returns(df)
                df['stock'] = stock_name
                combined_data[stock_name] = df
        
        return combined_data
    
    def create_panel_data(self, combined_data):
        """Create panel dataset for multiple stocks"""
        if not combined_data:
            logger.error("No data to concatenate!")
            return pd.DataFrame()
        
        panel_df = pd.concat(combined_data.values(), ignore_index=True)
        panel_df = panel_df.sort_values(['stock', 'Date'])
        return panel_df


def save_processed_data(df, filepath):
    """Save processed data"""
    if df.empty:
        logger.warning(f"Skipping save - empty dataframe for {filepath}")
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved processed data to {filepath}")


# Example usage
if __name__ == "__main__":
    loader = NSEDataLoader("data")
    
    # Define your stock files
    stocks = {
        'DANGSUG': 'DANGSUG.csv',
        'DANGCEM': 'DANGCEM.csv',
        'MTNN': 'MTNN.csv',
        'GTCO': 'GTCO.csv',
        'SEPLAT': 'SEPLAT.csv',
        'AIRTEL': 'AIRTEL.csv',
        'INTERBREW': 'INTERBREW.csv',
        'FIRSTHOLDCO': 'FIRSTHOLDCO.csv',
        'ETI': 'ETI.csv',
        'ZENITH': 'ZENITH.csv',
        'ACCESS': 'ACCESS.csv',
        'NESTLE': 'NESTLE.csv',
        'NB': 'NB.csv',
        'CWG': 'CWG.CSV',
        'WAPCO': 'WAPCO.csv'
    }
    
    # Load all stocks
    data = loader.prepare_multiple_stocks(stocks)
    
    # Save individual processed files
    for name, df in data.items():
        save_processed_data(df, f"data/processed/{name}_processed.csv")
    
    # Create and save panel data
    panel = loader.create_panel_data(data)
    save_processed_data(panel, "data/processed/panel_data.csv")