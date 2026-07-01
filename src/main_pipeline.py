#!/usr/bin/env python3
"""
NSE Stock Volatility Prediction - Main Pipeline
DASHBOARD-COMPATIBLE VERSION
"""

import sys
import json
import logging
import warnings
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from .news_features import add_news_features_to_dataframe
except ImportError:  # pragma: no cover - fallback for direct execution
    from news_features import add_news_features_to_dataframe

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DIRS = {
    'data_raw': BASE_DIR / 'data' / 'raw',
    'data_processed': BASE_DIR / 'data' / 'processed',
    'figures': BASE_DIR / 'results' / 'figures',
    'tables': BASE_DIR / 'results' / 'tables',
    'forecasts': BASE_DIR / 'results' / 'forecasts',
    'models': BASE_DIR / 'models'
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

class WindowsSafeHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            # Replace Unicode characters with ASCII equivalents
            msg = msg.replace('\u2713', '[OK]').replace('\u2717', '[FAIL]')
            msg = msg.replace('\U0001f4ca', '[DATA]').replace('\U0001f4c8', '[CHART]')
            msg = msg.replace('\U0001f4be', '[SAVE]').replace('\U0001f52e', '[FORECAST]')
            msg = msg.replace('\u26a0', '[WARN]').replace('\u23f3', '[WAIT]')
            self.stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging():
    logger = logging.getLogger('NSE_Pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    ch = WindowsSafeHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                                       datefmt='%H:%M:%S'))
    logger.addHandler(ch)

    fh = logging.FileHandler(DIRS['tables'] / 'pipeline.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

    return logger

logger = setup_logging()


@dataclass
class StockResult:
    stock_name: str
    success: bool
    error_message: Optional[str] = None
    data_points: int = 0
    test_rmse: Optional[float] = None
    test_mape: Optional[float] = None
    directional_accuracy: Optional[float] = None
    best_model: Optional[str] = None
    processing_time: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class DataLoader:
    """Robust CSV loader with flexible column detection"""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.stock_name = filepath.stem

    def load(self) -> Optional[pd.DataFrame]:
        if not self.filepath.exists():
            logger.error(f"[FAIL] File not found: {self.filepath}")
            return None

        logger.info(f"[WAIT] Loading {self.stock_name} from {self.filepath}")

        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 'windows-1252']

        for encoding in encodings:
            try:
                logger.info(f"   Trying encoding: {encoding}")
                df = pd.read_csv(self.filepath, encoding=encoding)
                logger.info(f"   Read {len(df)} rows with {len(df.columns)} columns using {encoding}")
                df = self._process(df)
                if df is not None:
                    logger.info(f"[OK] Loaded {len(df)} rows")
                    return df
                else:
                    logger.warning(f"   _process returned None for {encoding}")
            except Exception as e:
                logger.warning(f"   Failed with {encoding}: {str(e)[:100]}")
                continue

        # Try manual parse as last resort
        logger.info("   Attempting manual parse...")
        try:
            result = self._manual_parse()
            if result is not None:
                logger.info(f"[OK] Manual parse succeeded: {len(result)} rows")
                return result
        except Exception as e:
            logger.warning(f"   Manual parse failed: {str(e)[:100]}")

        logger.error(f"[FAIL] Could not load {self.stock_name}")
        return None
    def _manual_parse(self) -> Optional[pd.DataFrame]:
        with open(self.filepath, 'rb') as f:
            raw = f.read()

        text = None
        for enc in ['utf-8', 'latin-1', 'cp1252']:
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue

        if not text:
            return None

        lines = text.strip().split('\n')
        if len(lines) < 2:
            return None

        first = lines[0]
        seps = [',', ';', '\t', '|']
        sep_counts = {s: first.count(s) for s in seps}
        best = max(sep_counts, key=sep_counts.get)

        if sep_counts[best] == 0:
            header = first.split()
            data = [line.split() for line in lines[1:] if line.strip()]
        else:
            header = first.split(best)
            data = [line.split(best) for line in lines[1:] if line.strip()]

        df = pd.DataFrame(data, columns=header)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        return self._process(df)

    def _process(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or len(df) < 2:
            logger.error("   [FAIL] DataFrame is None or has < 2 rows")
            return None

        df = df.copy()
        original_cols = df.columns.tolist()
        df.columns = [str(c).strip() for c in df.columns]
        logger.info(f"   Original columns: {original_cols}")
        logger.info(f"   Cleaned columns: {df.columns.tolist()}")
        logger.info(f"   First few rows:\n{df.head(2).to_string()}")

        # Case-insensitive column lookup
        col_lower = {c.lower().replace(' ', '_'): c for c in df.columns}
        logger.info(f"   Column map: {col_lower}")

        # Find date column
        date_col = None
        for key in ['date', 'time', 'day', 'datetime', 'timestamp', 'trading_date']:
            if key in col_lower:
                date_col = col_lower[key]
                break

        # Map to standard column names (Title Case for dashboard compatibility)
        price_mappings = {
            'Price': ['price', 'close', 'closing', 'last', 'adj_close', 'settle', 'ending'],
            'Open': ['open', 'opening', 'first'],
            'High': ['high', 'highest', 'max', 'maximum'],
            'Low': ['low', 'lowest', 'min', 'minimum'],
            'Volume': ['volume', 'vol', 'shares', 'qty', 'quantity', 'turnover']
        }

        result_cols = {}
        for std_name, variants in price_mappings.items():
            for v in variants:
                if v in col_lower:
                    result_cols[std_name] = col_lower[v]
                    break

        logger.info(f"   Matched columns: {result_cols}")

        # Must have Price
        if 'Price' not in result_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            logger.info(f"   No Price column matched. Numeric cols: {numeric_cols}")
            if numeric_cols:
                result_cols['Price'] = numeric_cols[0]
            else:
                # Try to convert all columns to numeric
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                logger.info(f"   After forced conversion, numeric cols: {numeric_cols}")
                if numeric_cols:
                    result_cols['Price'] = numeric_cols[0]
                else:
                    logger.error("   [FAIL] No numeric columns found after conversion")
                    return None

        # Rename to standard names
        rename_map = {v: k for k, v in result_cols.items()}
        df = df.rename(columns=rename_map)

        # Ensure all required columns exist
        if 'Open' not in df.columns:
            df['Open'] = df['Price']
        if 'High' not in df.columns:
            df['High'] = df['Price'] * 1.001
        if 'Low' not in df.columns:
            df['Low'] = df['Price'] * 0.999
        if 'Volume' not in df.columns:
            df['Volume'] = 0

        # Handle Date
        if date_col:
            df = df.rename(columns={date_col: 'Date'})
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Date'].isna().all():
                df['Date'] = pd.date_range(end=datetime.now(), periods=len(df), freq='B')
        else:
            df['Date'] = pd.date_range(end=datetime.now(), periods=len(df), freq='B')

        # Select and order columns for dashboard compatibility
        final_cols = ['Date', 'Price', 'Open', 'High', 'Low', 'Volume']
        available = [c for c in final_cols if c in df.columns]
        df = df[available].copy()

        # Convert to numeric
        for c in ['Price', 'Open', 'High', 'Low', 'Volume']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        df = df.dropna(subset=['Price'])
        df = df.sort_values('Date').reset_index(drop=True)
        logger.info(f"   Final DataFrame: {len(df)} rows, columns: {df.columns.tolist()}")

        return df if len(df) >= 10 else None


class VolatilityFeatures:
    """Feature engineering for volatility models"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def calculate_returns(self):
        """Calculate returns"""
        self.df['log_return'] = np.log(self.df['Price'] / self.df['Price'].shift(1))
        self.df['simple_return'] = self.df['Price'].pct_change()
        return self.df

    def create_all_features(self, stock_name: Optional[str] = None):
        """Create comprehensive feature set and optionally enrich it with news/policy features."""
        df = self.df.copy()

        # Ensure returns exist
        if 'log_return' not in df.columns:
            df = self.calculate_returns()

        if stock_name is None and 'stock' in df.columns and not df['stock'].empty:
            stock_name = str(df['stock'].iloc[0])

        if stock_name is not None:
            logger.info(f"   Adding news and policy features for {stock_name}...")
            df = add_news_features_to_dataframe(df, stock_name)
        else:
            logger.info("   No stock name provided; skipping news feature enrichment")

        # Historical volatility
        for window in [5, 10, 21, 63, 126, 252]:
            df[f'hist_vol_{window}d'] = df['log_return'].rolling(window).std() * np.sqrt(252)

        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'ma_{window}d'] = df['Price'].rolling(window).mean()
            df[f'ma_ratio_{window}d'] = df['Price'] / df[f'ma_{window}d']

        # RSI
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14d'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['Price'].ewm(span=12, adjust=False).mean()
        ema26 = df['Price'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        df['bb_middle'] = df['Price'].rolling(20).mean()
        bb_std = df['Price'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Price'].shift())
        low_close = np.abs(df['Low'] - df['Price'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14d'] = tr.rolling(14).mean()

        # Volume features
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['volume_ma_20'] = df['Volume'].rolling(20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma_20']

        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'price_lag_{lag}'] = df['Price'].shift(lag)
            df[f'return_lag_{lag}'] = df['log_return'].shift(lag)
            df[f'vol_lag_{lag}'] = df['hist_vol_21d'].shift(lag)

        # Target
        df['target_vol_1d'] = df['hist_vol_21d'].shift(-1)

        return df.dropna()


def prepare_ml_dataset(df: pd.DataFrame, target_col: str = 'target_vol_1d'):
    """Prepare train/test splits"""
    feature_cols = [c for c in df.columns if c not in [
        'Date', 'Price', 'Open', 'High', 'Low', 'Volume',
        'log_return', 'simple_return', target_col
    ] and not c.startswith('target')]

    df_clean = df.dropna()

    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    dates = df_clean['Date'].values

    split_idx = int(len(X) * 0.8)

    return {
        'X_train': X[:split_idx],
        'X_test': X[split_idx:],
        'y_train': y[:split_idx],
        'y_test': y[split_idx:],
        'dates_train': dates[:split_idx],
        'dates_test': dates[split_idx:],
        'feature_names': feature_cols
    }


class MLVolatilitySuite:
    """ML models matching dashboard.py"""

    def __init__(self, dataset: Dict):
        self.dataset = dataset
        self.models = {}
        self.predictions = {}
        self.importance = {}
        self.y_test = dataset['y_test']
        self.dates_test = dataset['dates_test']
        self.X_train = dataset['X_train']
        self.X_test = dataset['X_test']
        self.y_train = dataset['y_train']

    def fit_random_forest(self):
        """Train Random Forest"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(self.X_train, self.y_train)
        preds = model.predict(self.X_test)

        self.models['Random Forest'] = model
        self.predictions['Random Forest'] = preds

        importance = pd.DataFrame({
            'Feature': self.dataset['feature_names'],
            'Importance': model.feature_importances_,
            'Model': 'Random Forest'
        }).sort_values('Importance', ascending=False)

        self.importance['Random Forest'] = importance
        logger.info(f"   Random Forest: RMSE={np.sqrt(mean_squared_error(self.y_test, preds)):.6f}")
        return self

    def fit_ridge(self):
        """Train Ridge Regression"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, self.y_train)
        preds = model.predict(X_test_scaled)

        self.models['Ridge'] = model
        self.predictions['Ridge'] = preds
        logger.info(f"   Ridge: RMSE={np.sqrt(mean_squared_error(self.y_test, preds)):.6f}")
        return self

    def fit_xgboost(self):
        """Train XGBoost (optional)"""
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42)
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            self.models['XGBoost'] = model
            self.predictions['XGBoost'] = preds
            logger.info(f"   XGBoost: RMSE={np.sqrt(mean_squared_error(self.y_test, preds)):.6f}")
        except ImportError:
            logger.warning("   XGBoost not available")
        return self

    def get_feature_importance(self, top_n: int = 20):
        """Get feature importance"""
        if 'Random Forest' in self.importance:
            return self.importance['Random Forest'].head(top_n)
        return pd.DataFrame()


# =============================================================================
# GARCH MODELS - FIXED FOR NEW ARCH LIBRARY API
# =============================================================================

class GARCHSuite:
    """GARCH models with fixed API compatibility"""

    def __init__(self, returns: pd.Series, dates: pd.Series):
        self.returns = returns.dropna()
        self.dates = dates
        self.models = {}
        self.results = {}

    def fit_garch(self, p=1, q=1):
        """Fit GARCH(p,q) model"""
        try:
            from arch import arch_model

            model = arch_model(self.returns, vol='Garch', p=p, q=q)
            result = model.fit(disp='off')

            # FIXED: Use convergence_flag instead of convergence
            convergence_status = getattr(result, 'convergence_flag', 0)
            if convergence_status != 0:
                logger.warning(f"   GARCH({p},{q}) did not converge (flag: {convergence_status})")

            self.models[f'GARCH({p},{q})'] = model
            self.results[f'GARCH({p},{q})'] = result

            logger.info(f"   GARCH({p},{q}) - AIC: {result.aic:.2f}")
            return result

        except Exception as e:
            logger.error(f"   GARCH({p},{q}) failed: {str(e)[:60]}")
            return None

    def fit_egarch(self, p=1, o=1, q=1):
        """Fit EGARCH(p,o,q) model"""
        try:
            from arch import arch_model

            model = arch_model(self.returns, vol='EGARCH', p=p, o=o, q=q)
            result = model.fit(disp='off')

            # FIXED: Use convergence_flag instead of convergence
            convergence_status = getattr(result, 'convergence_flag', 0)
            if convergence_status != 0:
                logger.warning(f"   EGARCH({p},{o},{q}) did not converge (flag: {convergence_status})")

            self.models[f'EGARCH({p},{o},{q})'] = model
            self.results[f'EGARCH({p},{o},{q})'] = result

            logger.info(f"   EGARCH({p},{o},{q}) - AIC: {result.aic:.2f}")
            return result

        except Exception as e:
            logger.error(f"   EGARCH({p},{o},{q}) failed: {str(e)[:60]}")
            return None

    def fit_gjr_garch(self, p=1, o=1, q=1):
        """Fit GJR-GARCH(p,o,q) model"""
        try:
            from arch import arch_model

            model = arch_model(self.returns, vol='GARCH', p=p, o=o, q=q, 
                             dist='normal')  # GJR is GARCH with o>0
            result = model.fit(disp='off')

            # FIXED: Use convergence_flag instead of convergence
            convergence_status = getattr(result, 'convergence_flag', 0)
            if convergence_status != 0:
                logger.warning(f"   GJR-GARCH({p},{o},{q}) did not converge (flag: {convergence_status})")

            self.models[f'GJR-GARCH({p},{o},{q})'] = model
            self.results[f'GJR-GARCH({p},{o},{q})'] = result

            logger.info(f"   GJR-GARCH({p},{o},{q}) - AIC: {result.aic:.2f}")
            return result

        except Exception as e:
            logger.error(f"   GJR-GARCH({p},{o},{q}) failed: {str(e)[:60]}")
            return None

    def get_best_model(self, criterion='AIC'):
        """Get best model by criterion"""
        if not self.results:
            return None

        best_model = None
        best_value = float('inf')

        for name, result in self.results.items():
            if result is None:
                continue
            value = getattr(result, criterion.lower(), float('inf'))
            if value < best_value:
                best_value = value
                best_model = name

        return best_model

    def rolling_forecast(self, model_name: str, window: int = 500) -> pd.DataFrame:
        """Generate rolling forecasts"""
        if model_name not in self.results:
            logger.error(f"Model {model_name} not found")
            return pd.DataFrame()

        result = self.results[model_name]
        returns = self.returns

        forecasts = []
        dates = []

        # Use last 'window' observations for forecasting
        start_idx = max(0, len(returns) - window)

        for i in range(start_idx, len(returns)):
            try:
                # Forecast one step ahead
                forecast = result.forecast(horizon=1, start=i, align='target')
                variance = forecast.variance.iloc[-1].values[0]
                volatility = np.sqrt(variance)
                forecasts.append(volatility)
                dates.append(self.dates.iloc[i] if i < len(self.dates) else None)
            except Exception:
                forecasts.append(np.nan)
                dates.append(None)

        df = pd.DataFrame({
            'Date': dates,
            'forecasted_vol': forecasts
        })

        return df.dropna()


class ModelEvaluator:
    """Evaluation matching dashboard.py - stores results as dicts for compatibility"""

    def __init__(self, stock_name: str):
        self.stock_name = stock_name
        self._results = {}

    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray, model_name: str):
        """Calculate metrics and store as dict"""
        n = min(len(actual), len(predicted))
        actual = actual[:n]
        predicted = predicted[:n]

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
        r2 = r2_score(actual, predicted)

        if len(actual) > 1:
            dir_true = np.sign(np.diff(actual))
            dir_pred = np.sign(np.diff(predicted))
            dir_acc = np.mean(dir_true == dir_pred) * 100
        else:
            dir_acc = 0

        # Store as dict for compatibility with evaluation.py
        self._results[model_name] = {
            'Stock': self.stock_name,
            'Model': model_name,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': dir_acc,
            'MSE': rmse**2,
            'QLIKE': 0.0,  # Placeholder
            'MZ_Alpha': 0.0,  # Placeholder
            'MZ_Beta': 0.0  # Placeholder
        }
        return self._results[model_name]

    def compare_all_models(self) -> pd.DataFrame:
        """Return comparison dataframe"""
        if not self._results:
            return pd.DataFrame()
        df = pd.DataFrame(list(self._results.values()))
        return df.sort_values('RMSE')


def process_stock(stock_name: str, filename: str) -> StockResult:
    """Process single stock with dashboard-compatible output"""
    start = datetime.now()

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {stock_name}")
    logger.info(f"{'='*60}")

    try:
        # Step 1: Load
        logger.info("[1/6] Loading raw data...")
        loader = DataLoader(DIRS['data_raw'] / filename)
        df = loader.load()
        if df is None:
            raise ValueError("Failed to load data")
        logger.info(f"   Loaded {len(df)} rows, columns: {list(df.columns)}")

        # Step 2: Feature Engineering
        logger.info("[2/6] Engineering features...")
        feature_eng = VolatilityFeatures(df)
        df_features = feature_eng.create_all_features(stock_name=stock_name)

        if len(df_features) < 50:
            raise ValueError(f"Insufficient data after features: {len(df_features)}")
        logger.info(f"   Created {len(df_features)} rows with {len(df_features.columns)} features")

        # Step 3: Save processed data (DASHBOARD FORMAT)
        logger.info("[3/6] Saving processed data...")
        processed_path = DIRS['data_processed'] / f"{stock_name}_features.csv"
        df_features.to_csv(processed_path, index=False)
        logger.info(f"   Saved: {processed_path}")

        # Step 4: Prepare ML dataset
        logger.info("[4/6] Preparing ML dataset...")
        dataset = prepare_ml_dataset(df_features, target_col='target_vol_1d')

        # Step 5: Train models
        logger.info("[5/6] Training ML models...")
        ml_suite = MLVolatilitySuite(dataset)
        ml_suite.fit_random_forest().fit_ridge().fit_xgboost()

        # Step 6: Evaluate and save results (DASHBOARD FORMAT)
        logger.info("[6/6] Evaluating and saving results...")
        evaluator = ModelEvaluator(stock_name)
        actual = dataset['y_test']

        for model_name, preds in ml_suite.predictions.items():
            evaluator.calculate_metrics(actual, preds, model_name)

        comparison = evaluator.compare_all_models()

        # Save model comparison (CRITICAL for dashboard)
        comparison_path = DIRS['tables'] / f"{stock_name}_model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        logger.info(f"   Saved comparison: {comparison_path}")

        # Save predictions (CRITICAL for dashboard)
        pred_df = pd.DataFrame(ml_suite.predictions)
        pred_df['Date'] = dataset['dates_test']
        pred_df['Actual'] = actual
        pred_path = DIRS['forecasts'] / f"{stock_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"   Saved predictions: {pred_path}")

        # Get best model
        if not comparison.empty:
            best = comparison.iloc[0]
            elapsed = (datetime.now() - start).total_seconds()
            logger.info(f"[OK] Best: {best['Model']}, RMSE: {best['RMSE']:.6f}, MAPE: {best['MAPE']:.2f}%")

            return StockResult(
                stock_name=stock_name,
                success=True,
                data_points=len(df_features),
                test_rmse=best['RMSE'],
                test_mape=best['MAPE'],
                directional_accuracy=best['Directional_Accuracy'],
                best_model=best['Model'],
                processing_time=elapsed
            )
        else:
            raise ValueError("No models trained successfully")

    except Exception as e:
        elapsed = (datetime.now() - start).total_seconds()
        error_msg = str(e)
        logger.error(f"[FAIL] {error_msg[:100]}")
        logger.debug(traceback.format_exc())
        return StockResult(
            stock_name=stock_name,
            success=False,
            error_message=error_msg[:200],
            processing_time=elapsed
        )


def generate_report(all_results: List[StockResult]):
    """Generate summary report"""
    successful = [r for r in all_results if r.success]
    failed = [r for r in all_results if not r.success]

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total': len(all_results),
        'successful': len(successful),
        'failed': len(failed),
        'success_rate': len(successful) / len(all_results) * 100 if all_results else 0
    }

    # Save JSON
    report_file = DIRS['tables'] / 'project_report.json'
    with open(report_file, 'w') as f:
        json.dump({
            'summary': summary,
            'stocks': [r.to_dict() for r in all_results]
        }, f, indent=2)

    # Save Markdown
    md_lines = [
        "# NSE Volatility Prediction Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Summary**: {summary['successful']}/{summary['total']} stocks processed successfully",
        "\n| Stock | Status | Model | RMSE | MAPE | Dir Acc |",
        "|-------|--------|-------|------|------|---------|"
    ]

    for r in all_results:
        if r.success:
            md_lines.append(
                f"| {r.stock_name} | OK | {r.best_model} | {r.test_rmse:.6f} | "
                f"{r.test_mape:.2f}% | {r.directional_accuracy:.1f}% |"
            )
        else:
            md_lines.append(f"| {r.stock_name} | FAIL | - | - | - | {r.error_message[:30]} |")

    md_file = DIRS['tables'].parent / 'FINAL_REPORT.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    logger.info(f"\n[OK] Reports saved to {DIRS['tables']}")


def main():
    """Main execution"""
    start = datetime.now()

    logger.info(f"\n{'='*70}")
    logger.info("NSE STOCK VOLATILITY PREDICTION PIPELINE")
    logger.info(f"Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"{'='*70}")

    stocks = {
        'DANGCEM': 'DANGCEM.csv', 'DANGSUG': 'DANGSUG.csv', 'MTNN': 'MTNN.csv', 'GTCO': 'GTCO.csv',
        'SEPLAT': 'SEPLAT.csv', 'AIRTEL': 'AIRTEL.csv', 'INTERBREW': 'INTERBREW.csv',
        'FIRSTHOLDCO': 'FIRSTHOLDCO.csv', 'ETI': 'ETI.csv', 'ZENITH': 'ZENITH.csv',
        'CWG': 'CWG.csv', 'NESTLE': 'NESTLE.csv', 'NB': 'NB.csv',
        'ACCESS': 'ACCESS.csv', 'WAPCO': 'WAPCO.csv'
    }

    available = {k: v for k, v in stocks.items() if (DIRS['data_raw'] / v).exists()}
    missing = [k for k in stocks if k not in available]

    if missing:
        logger.warning(f"[WARN] Missing: {', '.join(missing)}")

    if not available:
        logger.error("[FAIL] No stocks found!")
        logger.info(f"[INFO] Looking in: {DIRS['data_raw']}")
        return 1

    logger.info(f"[OK] Found {len(available)} stocks to process")

    results = []
    for stock, filename in available.items():
        result = process_stock(stock, filename)
        results.append(result)

    generate_report(results)

    elapsed = (datetime.now() - start).total_seconds()
    successful = sum(1 for r in results if r.success)

    logger.info(f"\n{'='*70}")
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Success: {successful}/{len(results)} stocks")
    logger.info(f"Time: {elapsed/60:.1f} minutes")
    logger.info(f"{'='*70}")
    logger.info("Dashboard-compatible files created:")
    logger.info("   - data/processed/{stock}_features.csv")
    logger.info("   - results/tables/{stock}_model_comparison.csv")
    logger.info("   - results/forecasts/{stock}_predictions.csv")
    logger.info("\nRun: streamlit run app.py")
    logger.info(f"{'='*70}")

    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())