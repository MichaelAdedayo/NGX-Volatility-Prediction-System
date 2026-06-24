"""
=============================================================================
BACKEND LOGIC FOR STREAMLIT WEB APPLICATION
=============================================================================
Nigerian Exchange Group (NGX) Volatility Prediction System
=============================================================================
"""

import logging
import json
import secrets
import hashlib
import base64
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sys
import os
import time
import threading
import traceback
from types import SimpleNamespace
from typing import Dict, List, Optional, Callable, Any

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

AUTH_USERS_FILE = os.path.join(root_dir, 'auth_users.json')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _hash_password(password: str, salt: Optional[str] = None, iterations: int = 200000) -> Dict[str, Any]:
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), iterations)
    return {
        'salt': salt,
        'hash': digest.hex(),
        'iterations': iterations
    }


def _load_user_accounts() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(AUTH_USERS_FILE):
        return {}
    try:
        with open(AUTH_USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load auth_users.json: {e}")
        return {}


def _save_user_accounts(accounts: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(AUTH_USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(accounts, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save auth_users.json: {e}")


def _get_role_permissions(role: str) -> Dict[str, bool]:
    mapping = {
        'Administrator': {
            'can_upload': True,
            'can_train': True,
            'can_manage_users': True,
            'can_view_all_predictions': True,
            'can_monitor_activity': True,
            'can_view_sample_dashboard': True
        },
        'Analyst': {
            'can_upload': True,
            'can_train': True,
            'can_manage_users': False,
            'can_view_all_predictions': False,
            'can_monitor_activity': False,
            'can_view_sample_dashboard': True
        },
        'Guest': {
            'can_upload': False,
            'can_train': False,
            'can_manage_users': False,
            'can_view_all_predictions': False,
            'can_monitor_activity': False,
            'can_view_sample_dashboard': True
        }
    }
    return mapping.get(role, mapping['Guest'])


def _render_role_summary(role: str) -> None:
    perms = _get_role_permissions(role)
    if not perms:
        return

    if role == 'Administrator':
        title = 'Admin Access'
        items = [
            'Manage users and accounts',
            'View all model predictions and evaluation outputs',
            'Monitor system activity and dataset history',
            'Upload datasets and run full training pipelines'
        ]
    elif role == 'Analyst':
        title = 'Analyst Access'
        items = [
            'Upload stock datasets',
            'Run predictions and compare model outputs',
            'View volatility forecasts and evaluation metrics'
        ]
    else:
        title = 'Guest Access'
        items = [
            'View sample dashboards and demo results',
            'Explore system features and summary analytics',
            'Load pre-trained forecasts in demo mode'
        ]

    st.sidebar.markdown(f"""
        <div style="padding: 14px 16px; border-radius: 14px; background: rgba(1,115,178,0.08); border: 1px solid rgba(1,115,178,0.16);">
            <div style="font-weight: 600; margin-bottom: 8px;">{title}</div>
            <ul style="margin: 0; padding-left: 18px; color: #e8f5ff;">
                {''.join([f'<li>{item}</li>' for item in items])}
            </ul>
        </div>
        """, unsafe_allow_html=True)


def _make_download_link(data: Any, filename: str, label: str) -> str:
    if isinstance(data, str):
        data = data.encode('utf-8')
    b64 = base64.b64encode(data).decode('utf-8')
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="display:inline-block;padding:10px 18px;background:#0173B2;color:#fff;border-radius:10px;text-decoration:none;font-weight:600;">{label}</a>'
    return href


def _load_system_activity() -> Dict[str, Any]:
    activity = {
        'available_stocks': 0,
        'processed_datasets': 0,
        'evaluation_reports': 0,
        'latest_files': []
    }
    try:
        if os.path.exists('data/processed'):
            activity['available_stocks'] = len([f for f in os.listdir('data/processed') if f.endswith('_features.csv')])
        if os.path.exists('results/tables'):
            activity['evaluation_reports'] = len([f for f in os.listdir('results/tables') if f.endswith('.csv')])
        if os.path.exists('results/forecasts'):
            activity['processed_datasets'] = len([f for f in os.listdir('results/forecasts') if f.endswith('.csv')])
        recent = []
        for folder in ['data/processed', 'results/tables', 'results/forecasts']:
            if os.path.exists(folder):
                for fname in os.listdir(folder):
                    path = os.path.join(folder, fname)
                    recent.append((fname, os.path.getmtime(path)))
        recent.sort(key=lambda x: x[1], reverse=True)
        activity['latest_files'] = [name for name, _ in recent[:6]]
    except Exception as e:
        logger.warning(f"System activity load failed: {e}")
    return activity


def _render_admin_console(controller: 'DashboardController') -> None:
    st.subheader('Admin Console')
    st.markdown('Manage user accounts, review project activity, and inspect prediction artifacts across the system.')

    accounts = _load_user_accounts()
    if not accounts:
        st.warning('No user accounts found in auth_users.json.')
        accounts = {}

    with st.expander('User Management'):
        st.markdown('Create, update, or delete user accounts for the NGX Volatility Prediction System.')
        new_user = st.text_input('New username', key='admin_new_user')
        new_full_name = st.text_input('New user full name', key='admin_new_full_name')
        new_role = st.selectbox('New user role', ['Administrator', 'Analyst', 'Guest'], key='admin_new_role')
        new_password = st.text_input('New user password', type='password', key='admin_new_password')

        if st.button('Create user', key='admin_create_user'):
            if not new_user or not new_password:
                st.error('Enter a username and password to create a new user.')
            elif new_user in accounts:
                st.error('A user with that username already exists.')
            else:
                accounts[new_user] = {
                    **_hash_password(new_password),
                    'role': new_role,
                    'full_name': new_full_name or new_user
                }
                _save_user_accounts(accounts)
                st.success(f'User {new_user} created successfully.')

        delete_user = st.selectbox('Delete existing user', [u for u in accounts if u != st.session_state.get('username', '')], key='admin_delete_user')
        if st.button('Delete selected user', key='admin_delete_user_btn'):
            if delete_user and delete_user in accounts:
                del accounts[delete_user]
                _save_user_accounts(accounts)
                st.success(f'User {delete_user} deleted successfully.')

        if accounts:
            account_table = pd.DataFrame([
                {'Username': u, 'Role': info.get('role', 'Unknown'), 'Full Name': info.get('full_name', '')}
                for u, info in accounts.items()
            ])
            st.dataframe(account_table, width='stretch')

    with st.expander('System Activity'):
        activity = _load_system_activity()
        st.markdown(f"""
            - Available stock feature datasets: **{activity['available_stocks']}**
            - Forecast files: **{activity['processed_datasets']}**
            - Evaluation report files: **{activity['evaluation_reports']}**
        """)
        if activity['latest_files']:
            st.markdown('**Recent project artifacts:**')
            for fname in activity['latest_files']:
                st.markdown(f'- {fname}')

    with st.expander('All Predictions Snapshot'):
        rows = []
        forecast_dir = 'results/forecasts'
        if os.path.exists(forecast_dir):
            for fname in os.listdir(forecast_dir):
                if fname.endswith('.csv'):
                    rows.append({'File': fname})
        if rows:
            st.dataframe(pd.DataFrame(rows), width='stretch')
        else:
            st.info('No forecast artifacts found. Load or train a stock to generate predictions.')

try:
    from data_collection import NSEDataLoader
except ImportError:
    logger.warning("data_collection not found, using fallback")
    NSEDataLoader = None

try:
    from features import VolatilityFeatures, prepare_ml_dataset
except ImportError:
    logger.warning("features not found, using fallback")
    VolatilityFeatures = None
    prepare_ml_dataset = None

try:
    from ml_models import MLVolatilitySuite
except ImportError:
    logger.warning("ml_models not found, using fallback")
    MLVolatilitySuite = None

try:
    from evaluation import ModelEvaluator, ALL_STOCKS, EvaluationMetrics
except ImportError as e:
    logger.warning(f"evaluation import failed: {e}")
    ModelEvaluator = None
    ALL_STOCKS = [
        'DANGSUG', 'DANGCEM', 'MTNN', 'GTCO', 'SEPLAT',
        'AIRTEL', 'INTERBREW', 'FIRSTHOLDCO', 'ETI', 'ZENITH',
        'CWG', 'NESTLE', 'NB', 'ACCESS', 'WAPCO'
    ]
    EvaluationMetrics = None

try:
    from garch_models import GARCHSuite, evaluate_garch_forecast
except ImportError:
    logger.warning("garch_models not found, using fallback")
    GARCHSuite = None
    evaluate_garch_forecast = None

try:
    from visualization import (
        plot_return_distribution_plotly,
        plot_rolling_volatility_plotly,
        plot_prediction_correlation_heatmap,
        plot_residual_error_plotly,
        plot_volatility_regime_detection
    )
except ImportError:
    logger.warning("visualization helpers not found, plots will be limited")
    plot_return_distribution_plotly = None
    plot_rolling_volatility_plotly = None
    plot_prediction_correlation_heatmap = None
    plot_residual_error_plotly = None
    plot_volatility_regime_detection = None


class TrainingStatus:
    """Thread-safe container for training progress tracking."""

    def __init__(self):
        self._lock = threading.Lock()
        self.running = False
        self.progress = 0.0
        self.message = "Idle"
        self.start_time = None
        self.error = None
        self.completed = False

    def update(self, progress: float, message: str):
        with self._lock:
            self.progress = min(max(progress, 0.0), 1.0)
            self.message = message

    def start(self):
        with self._lock:
            self.running = True
            self.start_time = time.time()
            self.progress = 0.0
            self.error = None
            self.completed = False

    def finish(self, success: bool = True, error: Optional[str] = None):
        with self._lock:
            self.running = False
            self.completed = success
            self.error = error
            self.progress = 1.0 if success else self.progress

    def get_status(self) -> Dict:
        with self._lock:
            return {
                'running': self.running,
                'progress': self.progress,
                'message': self.message,
                'error': self.error,
                'completed': self.completed
            }


class DashboardController:
    """
    Central controller for the volatility prediction web application.
    """

    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.raw_data_dir = "data/raw"

        self.current_stock: Optional[str] = None
        self.current_data: Optional[pd.DataFrame] = None
        self.is_uploaded: bool = False

        self.ml_suite: Optional[Any] = None
        self.evaluator: Optional[Any] = None
        self.garch_forecasts: Optional[pd.DataFrame] = None

        self.training_status = TrainingStatus()

    def get_available_stocks(self) -> List[str]:
        """Return official 15 stocks plus any uploaded ones."""
        stocks = list(ALL_STOCKS)

        if os.path.exists(self.data_dir):
            for f in os.listdir(self.data_dir):
                if f.endswith('_features.csv'):
                    name = f.replace('_features.csv', '')
                    if name not in stocks:
                        stocks.append(f"{name} (uploaded)")

        return sorted(stocks)


    def load_stock_data(self, stock_name: str) -> Optional[pd.DataFrame]:
        """Load processed data for a stock."""
        clean_name = stock_name.replace(" (uploaded)", "")

        filepath = os.path.join(self.data_dir, f"{clean_name}_features.csv")
        if not os.path.exists(filepath):
            return None

        try:
            df = pd.read_csv(filepath, parse_dates=['Date'])
            self.current_stock = clean_name
            self.current_data = df
            self.is_uploaded = "(uploaded)" in stock_name

            # Try to load existing evaluator results
            try:
                eval_path = f"results/tables/{clean_name}_model_comparison.csv"
                if os.path.exists(eval_path) and ModelEvaluator:
                    eval_df = pd.read_csv(eval_path)
                    self.evaluator = ModelEvaluator(stock_name=clean_name)
                    for _, row in eval_df.iterrows():
                        # Store as dict for compatibility
                        self.evaluator._results[row['Model']] = row.to_dict()
            except Exception as e:
                logger.warning(f"Could not load evaluator: {e}")

            # Try to load saved predictions so the Predictions tab can display pre-trained results.
            try:
                pred_path = f"results/forecasts/{clean_name}_predictions.csv"
                if os.path.exists(pred_path):
                    pred_df = pd.read_csv(pred_path, parse_dates=['Date'])
                    predictions = {
                        col: pred_df[col].values
                        for col in pred_df.columns
                        if col not in ['Date', 'Actual']
                    }
                    if predictions:
                        self.ml_suite = SimpleNamespace(
                            predictions=predictions,
                            dates_test=pred_df['Date'],
                            y_test=pred_df['Actual'] if 'Actual' in pred_df.columns else np.array([]),
                            get_feature_importance=lambda top_n=15: pd.DataFrame()
                        )
            except Exception as e:
                logger.warning(f"Could not load predictions: {e}")

            return df
        except Exception as e:
            logger.error(f"Error loading {stock_name}: {e}")
            return None

    def validate_uploaded_csv(self, uploaded_file: Any) -> tuple:
        """
        Validate user-uploaded CSV file.

        Returns: (is_valid: bool, message: str, dataframe: Optional[pd.DataFrame])
        """
        try:
            df = pd.read_csv(uploaded_file)

            # Check required columns
            missing = [c for c in ['Date', 'Price', 'Open', 'High', 'Low'] 
                      if c not in df.columns]

            if missing:
                return False, f"Missing columns: {missing}. Need: Date, Price, Open, High, Low", None

            # Check data sufficiency
            if len(df) < 252:
                return False, f"Need at least 252 trading days (1 year), got {len(df)}", None

            # Validate dates
            df['Date'] = pd.to_datetime(df['Date'])

            # Convert numeric columns (handle commas)
            numeric_cols = ['Price', 'Open', 'High', 'Low']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Check for required price data
            if df[['Price', 'Open', 'High', 'Low']].isnull().any().any():
                return False, "Missing or invalid values in Price/Open/High/Low columns", None

            return True, f"Valid: {len(df)} rows, {df['Date'].min().date()} to {df['Date'].max().date()}", df

        except Exception as e:
            return False, f"Error reading file: {str(e)}", None

    def process_uploaded_data(
        self, 
        df: pd.DataFrame, 
        stock_name: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        fast_mode: bool = False
    ) -> bool:
        """
        Execute full training pipeline for uploaded stock data.

        Args:
            df: Input dataframe
            stock_name: Stock ticker name
            progress_callback: Progress update function
            fast_mode: If True, skip GARCH and LSTM models for faster training (~50% reduction)
        """
        self.training_status.start()

        try:
            # Step 1: Feature Engineering (10%)
            if progress_callback:
                progress_callback(0.05, "Calculating returns and volatility measures...")

            if NSEDataLoader:
                loader = NSEDataLoader()
                df = loader.calculate_returns(df)
            else:
                # Fallback
                df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))

            if progress_callback:
                progress_callback(0.10, "Feature Engineering, Adding technical indicators...")

            if VolatilityFeatures:
                feature_eng = VolatilityFeatures(df)
                df_features = feature_eng.create_all_features()
            else:
                df_features = df.copy()

            # Save processed and feature data
            os.makedirs(self.data_dir, exist_ok=True)
            processed_path = os.path.join(self.data_dir, f"{stock_name}_processed.csv")
            feature_path = os.path.join(self.data_dir, f"{stock_name}_features.csv")
            df.to_csv(processed_path, index=False)
            df_features.to_csv(feature_path, index=False)

            # Step 2: Prepare ML Dataset (20%)
            if progress_callback:
                progress_callback(0.20, "Preparing train/test splits...")

            if prepare_ml_dataset:
                dataset = prepare_ml_dataset(df_features, target_col='target_vol_1d')
            else:
                # Fallback
                dataset = self._fallback_prepare_dataset(df_features)

            # Step 3: GARCH Models (40%) - Skip in fast mode
            if not fast_mode and GARCHSuite and progress_callback:
                progress_callback(0.30, "Training GARCH(1,1)...")

                garch = GARCHSuite(df['log_return'].dropna(), df['Date'])
                garch.fit_garch()

                progress_callback(0.35, "Training EGARCH...")
                garch.fit_egarch()

                progress_callback(0.40, "Training GJR-GARCH...")
                garch.fit_gjr_garch()

                best_garch = garch.get_best_model('AIC')

                progress_callback(0.50, f"Best GARCH: {best_garch}. Generating forecasts...")
                self.garch_forecasts = garch.rolling_forecast(best_garch, window=min(500, len(df)//3))
            elif progress_callback:
                progress_callback(0.50, "GARCH models skipped...")

            # Step 4: ML Models (80%)
            if MLVolatilitySuite and progress_callback:
                progress_callback(0.60, "Training Random Forest...")

                self.ml_suite = MLVolatilitySuite(dataset)
                self.ml_suite.fit_random_forest()

                progress_callback(0.70, "Training XGBoost...")
                self.ml_suite.fit_xgboost()

                progress_callback(0.75, "Training Ridge & SVR...")
                self.ml_suite.fit_ridge()
                if hasattr(self.ml_suite, 'fit_svr'):
                    self.ml_suite.fit_svr()

                # LSTM removed from pipeline to improve reliability and speed
                progress_callback(0.80, "LSTM removed from pipeline")
            elif progress_callback:
                progress_callback(0.80, "ML models skipped (not available)...")

            # Step 5: Evaluation (90%)
            if progress_callback:
                progress_callback(0.90, "Evaluating all models...")

            if ModelEvaluator:
                self.evaluator = ModelEvaluator(stock_name=stock_name)
                actual = dataset['y_test'].values if 'y_test' in dataset else dataset.get('y_test', np.array([]))

                # Add GARCH results
                if self.garch_forecasts is not None and GARCHSuite:
                    garch_pred = self.garch_forecasts['forecasted_vol'].values[:len(actual)]
                    self.evaluator.calculate_metrics(actual[:len(garch_pred)], garch_pred, 
                                                    f"GARCH-{best_garch}")

                # Add ML results
                if self.ml_suite and hasattr(self.ml_suite, 'predictions'):
                    for name, preds in self.ml_suite.predictions.items():
                        self.evaluator.calculate_metrics(actual[:len(preds)], preds, name)

                # Save results
                os.makedirs("results/tables", exist_ok=True)
                os.makedirs("results/forecasts", exist_ok=True)

                comparison = self.evaluator.compare_all_models()
                comparison.to_csv(f"results/tables/{stock_name}_model_comparison.csv", index=False)

                # Save predictions
                if self.ml_suite and hasattr(self.ml_suite, 'predictions'):
                    pred_df = pd.DataFrame(self.ml_suite.predictions)
                    if 'dates_test' in dataset:
                        pred_df['Date'] = dataset['dates_test'][:len(pred_df)]
                    pred_df['Actual'] = actual[:len(pred_df)]
                    pred_df.to_csv(f"results/forecasts/{stock_name}_predictions.csv", index=False)

            # Step 6: Visualization (100%)
            if progress_callback:
                progress_callback(0.95, "Generating figures...")

            # Update state
            self.current_stock = stock_name
            self.current_data = df_features

            if progress_callback:
                progress_callback(1.0, "Complete!")

            self.training_status.finish(success=True)
            return True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Training error: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            self.training_status.finish(success=False, error=error_msg)
            return False

    def _fallback_prepare_dataset(self, df: pd.DataFrame) -> Dict:
        """Fallback dataset preparation if features module not available."""
        feature_cols = [c for c in df.columns if c not in [
            'Date', 'Price', 'Open', 'High', 'Low', 'Volume',
            'log_return', 'simple_return', 'target_vol_1d'
        ] and not c.startswith('target')]

        df_clean = df.dropna()

        X = df_clean[feature_cols].values
        y = df_clean['target_vol_1d'].values if 'target_vol_1d' in df_clean.columns else np.zeros(len(df_clean))
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

    def get_price_chart(self) -> Optional[go.Figure]:
        """Create Plotly price chart."""
        if self.current_data is None:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.current_data['Date'],
            y=self.current_data['Price'],
            mode='lines',
            name='Price',
            line=dict(color='#0173B2', width=2)
        ))

        fig.update_layout(
            title=f"{self.current_stock}: Historical Stock Price",
            xaxis_title="Date",
            yaxis_title="Price (NGN)",
            template="plotly_white",
            height=600,
            hovermode='x unified'
        )
        return fig

    def get_volatility_chart(self) -> Optional[go.Figure]:
        """Create Plotly volatility chart."""
        if self.current_data is None or 'hist_vol_21d' not in self.current_data.columns:
            return None

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.current_data['Date'],
            y=self.current_data['hist_vol_21d'] * 100,
            mode='lines',
            name='21-Day Volatility',
            line=dict(color='#DE8F05', width=2),
            fill='tozeroy',
            fillcolor='rgba(222,143,5,0.1)'
        ))

        fig.update_layout(
            title=f"{self.current_stock}: Historical Volatility",
            xaxis_title="Date",
            yaxis_title="Annualized Volatility (%)",
            template="plotly_white",
            height=600
        )
        return fig

    def get_return_distribution_chart(self) -> Optional[go.Figure]:
        """Return distribution histogram (Plotly)."""
        if self.current_data is None or plot_return_distribution_plotly is None:
            return None
        try:
            return plot_return_distribution_plotly(self.current_data)
        except Exception as e:
            logger.debug(f"Return distribution plot failed: {e}")
            return None

    def get_rolling_volatility_chart(self, window: int = 21) -> Optional[go.Figure]:
        """Rolling volatility chart (Plotly)."""
        if self.current_data is None or plot_rolling_volatility_plotly is None:
            return None
        try:
            return plot_rolling_volatility_plotly(self.current_data, window=window)
        except Exception as e:
            logger.debug(f"Rolling vol plot failed: {e}")
            return None

    def get_predictions_chart(self, selected_models: Optional[List[str]] = None) -> Optional[go.Figure]:
        """Create prediction comparison chart.

        Adds GARCH forecasts to the plotted models if available and guards
        against non-array predictions to avoid Streamlit crashes when a
        selected model produced invalid output.
        """
        try:
            if self.ml_suite is None or self.evaluator is None:
                return None

            # Normalize dates and actual arrays
            dates = self.ml_suite.dates_test if hasattr(self.ml_suite, 'dates_test') else []
            if hasattr(dates, 'values'):
                dates = dates.values
            actual = self.ml_suite.y_test.values if hasattr(self.ml_suite, 'y_test') else np.array([])

            fig = go.Figure()

            # Actual (plot only if valid)
            try:
                if len(dates) > 0 and len(actual) > 0:
                    fig.add_trace(go.Scatter(
                        x=dates, y=actual, mode='lines',
                        name='Actual', line=dict(color='black', width=3)
                    ))
            except Exception:
                logger.debug('Skipping Actual trace: invalid dates/actual arrays')

            # Predictions (start with ML predictions)
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
            predictions = dict(self.ml_suite.predictions) if hasattr(self.ml_suite, 'predictions') else {}

            # Include GARCH forecasts if available
            try:
                if hasattr(self, 'garch_forecasts') and self.garch_forecasts is not None:
                    garch_pred = None
                    if isinstance(self.garch_forecasts, dict) and 'forecasted_vol' in self.garch_forecasts:
                        garch_pred = self.garch_forecasts['forecasted_vol']
                    elif hasattr(self.garch_forecasts, 'forecasted_vol'):
                        garch_pred = getattr(self.garch_forecasts, 'forecasted_vol')

                    if garch_pred is not None:
                        try:
                            garch_arr = garch_pred.values if hasattr(garch_pred, 'values') else np.array(garch_pred)
                        except Exception:
                            garch_arr = np.array(garch_pred)
                        predictions['GARCH'] = garch_arr
            except Exception as e:
                logger.debug(f"Could not include GARCH forecasts in predictions plot: {e}")

            models = selected_models or list(predictions.keys())

            skipped = []
            for idx, model_name in enumerate(models):
                if model_name not in predictions:
                    skipped.append(model_name)
                    continue

                pred_raw = predictions[model_name]

                # Convert pandas Series to numpy array
                try:
                    if hasattr(pred_raw, 'values'):
                        pred_arr = np.asarray(pred_raw.values)
                    else:
                        pred_arr = np.asarray(pred_raw)
                except Exception:
                    logger.debug(f"Skipping {model_name}: cannot convert predictions to array")
                    skipped.append(model_name)
                    continue

                # Handle scalar predictions (single value)
                if pred_arr.ndim == 0:
                    if len(dates) > 0:
                        y_vals = np.full(len(dates), float(pred_arr))
                        x_vals = dates
                    else:
                        skipped.append(model_name)
                        continue
                else:
                    # If lengths differ, align to the end of dates (common with rolling forecasts)
                    try:
                        n_pred = len(pred_arr)
                    except Exception:
                        skipped.append(model_name)
                        continue

                    if len(dates) == 0:
                        # No dates to align to; skip plotting
                        skipped.append(model_name)
                        continue

                    if n_pred == len(dates):
                        x_vals = dates
                        y_vals = pred_arr
                    elif n_pred < len(dates) and n_pred > 0:
                        x_vals = dates[-n_pred:]
                        y_vals = pred_arr
                    elif n_pred > len(dates) and len(dates) > 0:
                        # Predictions longer than dates: take last len(dates)
                        x_vals = dates
                        y_vals = pred_arr[-len(dates):]
                    else:
                        skipped.append(model_name)
                        continue

                # If y_vals contains all NaNs, skip
                try:
                    if np.all(np.isnan(y_vals)):
                        skipped.append(model_name)
                        continue
                except Exception:
                    pass

                # Safe add trace
                try:
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=y_vals, mode='lines',
                        name=model_name,
                        line=dict(color=colors[idx % len(colors)], width=2),
                        opacity=0.85
                    ))
                except Exception as e:
                    logger.debug(f"Failed to plot {model_name}: {e}")
                    skipped.append(model_name)

            # If any selected models were skipped, include a small annotation
            if skipped:
                logger.debug(f"Predictions skipped for models: {skipped}")

            fig.update_layout(
                title=f"{self.current_stock}: Volatility Predictions vs Actual",
                xaxis_title="Date",
                yaxis_title="Volatility",
                template="plotly_white",
                height=700,
                hovermode='x unified'
            )
            return fig
        except Exception as e:
            logger.error(f"Error building predictions chart: {e}")
            logger.debug(traceback.format_exc())
            return None

    def get_performance_table(self) -> pd.DataFrame:
        """Get performance metrics DataFrame."""
        if self.evaluator is None:
            return pd.DataFrame()
        return self.evaluator.compare_all_models()

    def get_feature_importance_chart(self, model_name: str = "Random Forest", 
                                      top_n: int = 15) -> Optional[go.Figure]:
        """Create feature importance bar chart."""
        if self.ml_suite is None:
            return None

        importance = self.ml_suite.get_feature_importance(top_n=top_n) if hasattr(self.ml_suite, 'get_feature_importance') else pd.DataFrame()
        if importance.empty:
            return None

        # Filter by model
        if 'Model' in importance.columns:
            model_imp = importance[importance['Model'] == model_name]
        else:
            model_imp = importance

        if model_imp.empty:
            return None

        fig = go.Figure(go.Bar(
            x=model_imp['Importance'],
            y=model_imp['Feature'],
            orientation='h',
            marker_color='#0173B2'
        ))

        fig.update_layout(
            title=f"Top {top_n} Features: {model_name}",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            template="plotly_white",
            height=600,
            yaxis=dict(autorange="reversed")
        )
        return fig

    def get_prediction_correlation_heatmap(self) -> Optional[go.Figure]:
        """Create a correlation heatmap for model predictions and actual values."""
        if self.ml_suite is None or plot_prediction_correlation_heatmap is None:
            return None

        try:
            predictions = getattr(self.ml_suite, 'predictions', {})
            actual = getattr(self.ml_suite, 'y_test', np.array([]))
            if len(predictions) == 0 or len(actual) == 0:
                return None

            return plot_prediction_correlation_heatmap(predictions, np.array(actual))
        except Exception as e:
            logger.debug(f"Prediction correlation heatmap failed: {e}")
            return None

    def get_residual_error_chart(self) -> Optional[go.Figure]:
        """Create a residual/error analysis chart for the best model."""
        if self.evaluator is None or self.ml_suite is None or plot_residual_error_plotly is None:
            return None

        try:
            results = self.get_performance_table()
            if results.empty:
                return None

            best_model = results.iloc[0]['Model']
            predictions = getattr(self.ml_suite, 'predictions', {})
            actual = getattr(self.ml_suite, 'y_test', np.array([]))
            dates = getattr(self.ml_suite, 'dates_test', None)
            if len(predictions) == 0 or len(actual) == 0:
                return None

            return plot_residual_error_plotly(np.array(actual), predictions, best_model, dates=np.array(dates) if dates is not None else None)
        except Exception as e:
            logger.debug(f"Residual/error chart failed: {e}")
            return None

    def get_volatility_regime_chart(self) -> Optional[go.Figure]:
        """Create a volatility regime detection chart for the current stock."""
        if self.current_data is None or plot_volatility_regime_detection is None:
            return None

        try:
            return plot_volatility_regime_detection(self.current_data)
        except Exception as e:
            logger.debug(f"Volatility regime chart failed: {e}")
            return None

    def forecast_next_day(self) -> Optional[Dict]:
        """
        Generate next-day volatility forecast using the most recent data.
        Returns dict with model predictions and metadata.
        """
        if self.current_data is None or self.ml_suite is None:
            return None

        try:
            # First, try to use the last prediction from trained models as proxy
            if hasattr(self.ml_suite, 'predictions') and self.ml_suite.predictions:
                forecasts = {}
                for model_name, preds in self.ml_suite.predictions.items():
                    if len(preds) > 0 and not np.isnan(preds[-1]):
                        forecasts[model_name] = float(preds[-1])

                if forecasts:
                    ensemble_forecast = np.mean(list(forecasts.values()))

                    # Use the latest available data date for forecasting, not only the test set end.
                    last_date = pd.to_datetime(self.current_data['Date']).max()
                    if pd.isna(last_date):
                        last_date = pd.to_datetime(self.current_data['Date'].iloc[-1])
                    forecast_date = last_date + pd.Timedelta(days=1)

                    # Add decision-support summary using recent realized volatility.
                    historical_vol = None
                    if 'hist_vol_21d' in self.current_data.columns:
                        recent_vals = self.current_data['hist_vol_21d'].dropna()
                        if len(recent_vals) > 0:
                            historical_vol = float(recent_vals.iloc[-1])
                    elif 'realized_vol_21d' in self.current_data.columns:
                        recent_vals = self.current_data['realized_vol_21d'].dropna()
                        if len(recent_vals) > 0:
                            historical_vol = float(recent_vals.iloc[-1])

                    forecast_summary = None
                    stock_name = self.current_stock or "This stock"

                    # Classify volatility level
                    def classify_volatility(vol):
                        if vol < 0.15:
                            return "low", "calm"
                        elif vol < 0.25:
                            return "moderate", "balanced"
                        elif vol < 0.40:
                            return "elevated", "turbulent"
                        else:
                            return "high", "extremely turbulent"

                    vol_level, vol_desc = classify_volatility(ensemble_forecast)

                    if historical_vol is not None and historical_vol > 0:
                        change_pct = (ensemble_forecast - historical_vol) / historical_vol * 100

                        if change_pct > 5:
                            forecast_summary = (
                                f" **{stock_name}** is heating up! Next-day volatility is expected to spike by "
                                f"**{change_pct:.1f}%** above recent levels (from {historical_vol:.3f} to {ensemble_forecast:.3f}). "
                                f"This signals **elevated risk** with potentially larger price swings. "
                                f" **Investors**: Consider tightening stop losses and reducing position sizes. "
                                f"**Traders**: Watch for higher trading ranges and potential breakout opportunities."
                            )
                        elif change_pct < -5:
                            forecast_summary = (
                                f" **{stock_name}** is calming down! Next-day volatility is expected to decline by "
                                f"**{abs(change_pct):.1f}%** below recent levels (from {historical_vol:.3f} to {ensemble_forecast:.3f}). "
                                f"This suggests **more stable trading conditions** ahead with tighter price ranges. "
                                f" **Investors**: A good window for position adjustments or rebalancing. "
                                f"**Traders**: Expect reduced trading ranges and potential consolidation patterns."
                            )
                        else:
                            forecast_summary = (
                                f" **{stock_name}** is holding steady! Next-day volatility is expected to remain "
                                f"**similar to recent levels** (±5%), staying around {ensemble_forecast:.3f}. "
                                f"Market conditions appear {vol_desc} and predictable. "
                                f" **Investors**: Continue with your current risk management strategy. "
                                f"**Traders**: Maintain current position sizing and trend-following strategies. "
                                f"**Monitor**: Watch for any news or market catalysts that could change this outlook."
                            )
                    else:
                        forecast_summary = (
                            f"**{stock_name}** forecast shows {vol_level} volatility ({ensemble_forecast:.3f}). "
                            f"Limited historical data is available, but models predict {vol_desc} market conditions. "
                            f"Use these predictions alongside other technical indicators for best results."
                        )

                    return {
                        'forecast_date': forecast_date,
                        'generated_at': pd.Timestamp.now(),
                        'model_forecasts': forecasts,
                        'ensemble_forecast': ensemble_forecast,
                        'confidence_range': {
                            'low': ensemble_forecast * 0.8,
                            'high': ensemble_forecast * 1.2
                        },
                        'forecast_summary': forecast_summary
                    }

            # Fallback: predict on the most recent row of features
            recent_data = self.current_data.iloc[-1:]
            feature_cols = [c for c in recent_data.columns
                          if c not in ['Date', 'Price', 'Open', 'High', 'Low', 'Volume']
                          and not c.startswith('target')
                          and pd.api.types.is_numeric_dtype(recent_data[c])]

            if not feature_cols:
                return None

            # Prepare features for prediction
            X_recent = recent_data[feature_cols].fillna(0).values

            # Generate predictions from all available models
            forecasts = {}
            if hasattr(self.ml_suite, 'models'):
                for model_name, model in self.ml_suite.models.items():
                    # LSTM removed; no special-case skipping required
                    try:
                        if model_name in ['Ridge', 'SVR']:
                            # Scale for models trained on scaled data
                            if hasattr(self.ml_suite, 'scaler'):
                                X_input = self.ml_suite.scaler.transform(X_recent)
                            else:
                                X_input = X_recent
                        else:
                            X_input = X_recent
                        pred = model.predict(X_input)[0]
                        forecasts[model_name] = float(pred)
                    except Exception as e:
                        logger.warning(f"Failed to predict with {model_name}: {e}")
                        continue

            if not forecasts:
                return None

            # Calculate ensemble forecast (average of all models)
            ensemble_forecast = np.mean(list(forecasts.values()))

            # Get the date for tomorrow
            last_date = pd.to_datetime(self.current_data['Date'].iloc[-1])
            forecast_date = last_date + pd.Timedelta(days=1)

            return {
                'forecast_date': forecast_date,
                'generated_at': pd.Timestamp.now(),
                'model_forecasts': forecasts,
                'ensemble_forecast': ensemble_forecast,
                'confidence_range': {
                    'low': ensemble_forecast * 0.8,  # Rough estimate
                    'high': ensemble_forecast * 1.2
                }
            }

        except Exception as e:
            logger.error(f"Error generating next-day forecast: {e}")
            logger.error(traceback.format_exc())
            return None


# =============================================================================
# AUTHENTICATION UI HELPERS
# =============================================================================

def _render_auth_sidebar():
    """Render user info and logout in the sidebar. Called from create_streamlit_app."""
    import streamlit as st

    # Check if we're running under the auth system (app.py)
    if "authenticated" not in st.session_state or not st.session_state.authenticated:
        return  # Not in auth mode, skip

    # Add a subtle user badge at the top of sidebar
    username = st.session_state.get("username", "User")
    full_name = st.session_state.get("user_full_name", username)
    role = st.session_state.get("user_role", "User")

    st.sidebar.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(1,115,178,0.15), rgba(0,166,251,0.08));
        border: 1px solid rgba(1,115,178,0.2);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 20px;
    ">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="
                width: 38px; height: 38px;
                background: linear-gradient(135deg, #0173B2, #00A6FB);
                border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                font-size: 16px; color: white; font-weight: 700;
            ">{username[0].upper()}</div>
            <div>
                <div style="font-weight: 600; font-size: 14px; color: #ffffff;">{full_name}</div>
                <div style="font-size: 11px; color: rgba(255,255,255,0.5);">{role}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Logout button
    if st.sidebar.button("Log Out", use_container_width=True, type="secondary"):
        # Clear auth state
        for key in ["authenticated", "username", "login_time", "user_role", "user_full_name", "login_error"]:
            if key in st.session_state:
                if key == "authenticated":
                    st.session_state[key] = False
                else:
                    st.session_state.pop(key, None)
        st.rerun()

    st.sidebar.markdown("---")


# =============================================================================
# STREAMLIT APP CREATION FUNCTION
# =============================================================================

def create_streamlit_app():
    """
    Create and run the Streamlit web application.

    This is the MAIN ENTRY POINT called by app.py.
    """
    try:
        import streamlit as st
    except ImportError:
        print("ERROR: Streamlit not installed.")
        print("Run: pip install streamlit")
        return

    # Page configuration — only call if not already configured by app.py
    try:
        st.set_page_config(
            page_title="NGX Volatility Prediction",
            page_icon="chart_with_upwards_trend",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception:
        # set_page_config already called (by app.py), ignore
        pass

    # Custom Font Styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Poppins', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    @keyframes gradientSlide {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes floatText {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-4px); }
    }

    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(24px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.75; }
    }

    .animated-heading {
        display: inline-block;
        background: linear-gradient(90deg, #0173B2, #00A6FB, #00D4FF);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        color: transparent;
        animation: gradientSlide 4s ease infinite, floatText 6s ease-in-out infinite;
    }

    .animated-subtitle {
        opacity: 0.95;
        animation: fadeInUp 1.2s ease-out both;
        line-height: 1.6;
    }

    .pulse-banner {
        display: inline-block;
        background: rgba(1, 115, 178, 0.12);
        color: #ffffff;
        padding: 14px 18px;
        border-radius: 18px;
        margin-top: 12px;
        animation: pulse 3s ease-in-out infinite;
        border: 1px solid rgba(1, 115, 178, 0.2);
    }

    .stButton>button,
    .stTextInput>div>input,
    .stSelectbox>div>div[data-baseweb],
    .stCheckbox>div>label,
    .stRadio>div>label {
        transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.3s ease, border-color 0.2s ease;
    }

    .stButton>button:hover,
    .stTextInput>div>input:hover,
    .stSelectbox>div>div[data-baseweb]:hover,
    .stCheckbox>div>label:hover,
    .stRadio>div>label:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
    }

    .stButton>button:active,
    .stTextInput>div>input:focus,
    .stSelectbox>div>div[data-baseweb]:focus-within,
    .stCheckbox>div>label:active,
    .stRadio>div>label:active {
        transform: scale(0.98);
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.16);
        outline: none !important;
    }

    .workflow-card {
        background: transparent;
        border: none;
        padding: 0;
        margin-bottom: 28px;
    }

    .workflow-flow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        justify-content: center;
        margin-top: 16px;
    }

    .workflow-step {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.16);
        padding: 12px 16px;
        border-radius: 12px;
        min-width: 160px;
        text-align: center;
        font-weight: 600;
        color: #ffffff;
    }

    .workflow-title {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 12px;
        color: #ffffff;
        text-align: center;
    }

    .workflow-arrow {
        font-size: 22px;
        color: #ffffff;
    }

    .workflow-card {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    .workflow-flow {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
        justify-content: center;
        margin-top: 16px;
    }

    div[data-testid="stPlotlyChart"] {
        animation: fadeInUp 0.9s ease-out both;
    }

    .stMetric {
        animation: fadeInUp 0.9s ease-out both;
    }

    .app-footer {
        background: transparent;
        color: #ffffff;
        padding: 18px 0;
        margin-top: 56px;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        border-top: 1px solid rgba(255,255,255,0.15);
    }

    .app-footer a {
        color: #ffffff;
        text-decoration: none;
    }

    </style>
    """, unsafe_allow_html=True)

    def render_footer():
        st.markdown(
            """
            <div class='app-footer'>
                Developed by: Michael Adedayo Iseoluwa | Department of Computer Science, Crawford University | Copyright © 2026 NGX Volatility Prediction System. All rights reserved.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Header
    st.markdown(
        """
        <h1 class="animated-heading">Nigerian Exchange Group (NGX) Volatility Prediction System</h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <div class="animated-subtitle">
        This system implements an automated machine learning pipeline for stock market volatility prediction using historical Nigerian Exchange Group (NGX) data.
        </div>
        <div class="animated-subtitle"> 
        The platform integrates data preprocessing, feature engineering, model training, evaluation, and forecasting within a unified framework.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Initialize controller
    controller = DashboardController()

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # === AUTHENTICATION SIDEBAR SECTION ===
    _render_auth_sidebar()
    # =====================================

    role = st.session_state.get("user_role", "Guest")
    permissions = _get_role_permissions(role)
    _render_role_summary(role)

    if role == "Guest":
        mode = st.sidebar.radio(
            "Select Mode",
            ["Quick Demo"],
            help="Guest users can view sample dashboards and explore system features."
        )
    else:
        mode = st.sidebar.radio(
            "Select Mode",
            ["Quick Load (Pre-trained)", "Upload & Train (New Stock)"],
            help="Quick Load uses pre-computed models. Upload & Train builds new models (5-10 minutes)."
        )

    # Mode: Quick Load / Demo
    if mode in ["Quick Load (Pre-trained)", "Quick Demo"]:
        available = controller.get_available_stocks()
        pre_trained = [s for s in available if "(uploaded)" not in s]
        if role == "Guest":
            pre_trained = pre_trained[:6]

        stock = st.sidebar.selectbox(
            "Select Stock",
            pre_trained if pre_trained else ["No data available"],
            disabled=len(pre_trained) == 0
        )

        if st.sidebar.button("Load Data", disabled=len(pre_trained) == 0):
            with st.spinner(f"Loading {stock}..."):
                df = controller.load_stock_data(stock)
                if df is not None:
                    st.sidebar.success(f"Loaded {len(df):,} observations")
                    st.sidebar.info(f"Features: {df.shape[1]}")
                else:
                    st.sidebar.error("Data not found. Run main_pipeline.py first.")

    # Mode: Upload & Train
    elif mode == "Upload & Train (New Stock)":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Upload New Stock Data")

        uploaded = st.sidebar.file_uploader(
            "Upload CSV (Investing.com format)",
            type=['csv'],
            help="Required columns: Date, Price, Open, High, Low. Minimum 252 rows (1 year)."
        )

        if uploaded is not None:
            is_valid, msg, df = controller.validate_uploaded_csv(uploaded)

            if not is_valid:
                st.sidebar.error(f"Error: {msg}")
            else:
                st.sidebar.success(f"Valid: {msg}")

                stock_name = st.sidebar.text_input(
                    "Stock Ticker Name",
                    value=uploaded.name.replace('.csv', '').upper()[:10]
                )

                # Fast training mode option
                fast_mode = st.sidebar.checkbox(
                    "Fast Mode (skip GARCH & LSTM)",
                    value=False,
                    help="Reduces training time significantly by skipping slower models"
                )

                if st.sidebar.button("Train All Models", type="primary"):
                    progress_bar = st.sidebar.progress(0)
                    status_text = st.sidebar.empty()

                    def update_progress(pct, msg):
                        progress_bar.progress(min(int(pct * 100), 100))
                        status_text.text(f"{int(pct*100)}% - {msg}")

                    with st.spinner("Training in progress... This takes less than a minute..."):
                        success = controller.process_uploaded_data(
                            df, stock_name, progress_callback=update_progress, fast_mode=fast_mode
                        )

                        if success:
                            st.sidebar.success("Training complete!")
                            st.sidebar.info("Figures saved to results/figures/")
                        else:
                            st.sidebar.error(f"Training failed: {controller.training_status.error}")

    # Main content area
    if controller.current_data is None:
        st.markdown(
            "<div style='color: #ffffff; font-size: 1rem; padding: 10px 12px; background: rgba(30, 144, 255, 0.1); border-radius: 12px;'>"
            "Select a mode from the sidebar to begin exploring volatility predictions for Nigerian Stocks "
            "</div>",
            unsafe_allow_html=True
        )

        # Show workflow
        st.markdown(
            """
            <div class="workflow-card">
                <div class="workflow-title">System Workflow</div>
                <div class="workflow-flow">
                    <div class="workflow-step">Upload Dataset</div>
                    <div class="workflow-arrow">→</div>
                    <div class="workflow-step">Data Validation</div>
                    <div class="workflow-arrow">→</div>
                    <div class="workflow-step">Feature Engineering</div>
                    <div class="workflow-arrow">→</div>
                    <div class="workflow-step">Volatility Computation</div>
                    <div class="workflow-arrow">→</div>
                    <div class="workflow-step">Prediction Engine</div>
                    <div class="workflow-arrow">→</div>
                    <div class="workflow-step">Forecast Dashboard</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Show capabilities
        st.subheader("System Capabilities")
        cols = st.columns(3)

        with cols[0]:
            st.markdown("**Core Modules**")
            st.markdown(
                """
                - Data Ingestion Engine
                - Preprocessing Module
                - Feature Engineering Engine
                - Volatility Computation Module
                - Machine Learning Prediction Engine
                - Evaluation Framework
                - Forecast Visualization Dashboard
                """
            )

        with cols[1]:
            st.markdown("**Machine Learning Models**")
            st.markdown(
                """
                1. Random Forest
                2. XGBoost
                3. Ridge Regression
                4. Support Vector Regression
                5. GARCH Family (baseline)
                """
            )

        with cols[2]:
            st.markdown("**Evaluation & Deployment**")
            st.markdown(
                """
                - RMSE, MAE, R2 metrics
                - Diebold-Mariano tests
                - Feature importance
                - Interactive dashboard
                """
            )

        if role == "Administrator":
            st.markdown("---")
            st.header("Admin Console")
            _render_admin_console(controller)

        render_footer()
        return

    # Display current stock info
    summary = {
        'name': controller.current_stock,
        'rows': len(controller.current_data),
        'features': len([c for c in controller.current_data.columns 
                        if not c.startswith('target')])
    }

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock Ticker", summary['name'])
    c2.metric("Observations", f"{summary['rows']:,}")
    c3.metric("Features", summary['features'])
    c4.metric("Status", "Trained" if controller.ml_suite else "Loaded")

    st.markdown(
        """
        **What this means:** The charts and tables below explain how this stock behaved in the past, how our models expect volatility to move, and how accurate those predictions were.
        - Use the sidebar to load or upload stock data.
        - The first tab shows price and volatility over time.
        - The second tab compares model forecasts to actual outcomes.
        - The third tab shows simple metrics that tell you which model did best.
        - The fourth tab gives you tomorrow's volatility forecast with actionable insights for risk management.
        """
    )
    st.markdown(
        """
        <div class="pulse-banner">
        Ready to explore volatility with lively model insights and animated charts? Scroll down to interact with the dashboard.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tabs
    tabs = ["Historical Data", "Predictions", "Results", "Forecast"]
    if role == "Administrator":
        tabs.append("Admin Console")

    tab1, tab2, tab3, tab4, *extra_tabs = st.tabs(tabs)
    admin_tab = extra_tabs[0] if extra_tabs else None

    with tab1:
        st.subheader("Historical Price and Volatility")
        st.markdown(
            """
            This tab shows how the stock traded over time and how much the price jumped around.
            The price chart is the actual stock value, and the volatility chart shows how unstable the price was.
            If volatility is high, the stock made bigger moves; if it is low, it moved more gently.
            """
        )
        col1, col2 = st.columns(2)
        with col1:
            fig = controller.get_price_chart()
            if fig:
                # FIXED: Use width="stretch" instead of use_container_width=True
                st.plotly_chart(fig, width="stretch")
                st.markdown(
                    "**Price Chart Explanation:** This line shows how the stock's closing price moved over time. "
                    "Use it to see if the stock is trending up, down, or moving sideways."
                )
        with col2:
            fig = controller.get_volatility_chart()
            if fig:
                # FIXED: Use width="stretch" instead of use_container_width=True
                st.plotly_chart(fig, width="stretch")
                st.markdown(
                    "**Volatility Chart Explanation:** This chart shows how much the stock's returns changed over time. "
                    "When the line is higher, the stock is swinging more widely and the market is more unpredictable."
                )

        # Additional visualizations
        col3, col4 = st.columns(2)
        with col3:
            fig = controller.get_return_distribution_chart()
            if fig:
                st.plotly_chart(fig, width="stretch")
                st.markdown(
                    "**Return Distribution Explanation:** This histogram shows how often different daily return sizes happened. "
                    "A narrow peak means most days were calm, while a wide spread means big jumps were common."
                )
        with col4:
            fig = controller.get_rolling_volatility_chart()
            if fig:
                st.plotly_chart(fig, width="stretch")
                st.markdown(
                    "**Rolling Volatility Explanation:** This line shows the recent trend of volatility, averaged over about one month. "
                    "It helps you see whether risk is increasing, stable, or falling."
                )

    with tab2:
        st.subheader("Model Predictions")
        st.markdown(
            """
            Here we compare each model's forecast to what actually happened.
            If a model line follows the actual volatility line closely, that model is doing a good job.
            A model that stays far away from the black actual line is less reliable.
            """
        )

        if controller.ml_suite is None:
            st.info("Models not trained for this stock. Use 'Upload & Train' mode.")
        else:
            available = list(controller.ml_suite.predictions.keys()) if hasattr(controller.ml_suite, 'predictions') else []
            # Inform user if some expected models (e.g. SVR) are missing
            expected_models = ['Random Forest', 'XGBoost', 'Ridge', 'SVR']
            missing_models = [m for m in expected_models if m not in available]
            if missing_models:
                st.info(f"Note: the following models are not available for this stock: {', '.join(missing_models)}. "
                        "They may have failed during training, produced invalid predictions, or were skipped.")
            selected = st.multiselect(
                "Select models to display",
                available,
                default=available[:3] if available else []
            )

            if selected:
                fig = controller.get_predictions_chart(selected)
                if fig:
                    # FIXED: Use width="stretch" instead of use_container_width=True
                    st.plotly_chart(fig, width="stretch")
                    st.markdown(
                        "**Prediction Comparison Explanation:** The black line is actual volatility and the colored lines are model estimates. "
                        "A good model stays close to the black line most of the time."
                    )

            corr_fig = controller.get_prediction_correlation_heatmap()
            if corr_fig:
                st.subheader("Prediction Correlation Heatmap")
                st.plotly_chart(corr_fig, width="stretch")
                st.markdown(
                    "**Correlation Heatmap Explanation:** This chart shows how closely each model follows the others and the actual values. "
                    "Values near 1 mean two lines move together; values near -1 mean they move opposite."
                )

            # Latest predictions table
            st.subheader("Latest Predictions")
            pred_data = []
            predictions = controller.ml_suite.predictions if hasattr(controller.ml_suite, 'predictions') else {}
            for model, preds in predictions.items():
                if len(preds) > 0:
                    pred_data.append({
                        'Model': model,
                        'Latest_Volatility': f"{preds[-1]:.4f}",
                        'Mean_Predicted': f"{np.mean(preds):.4f}"
                    })
            if pred_data:
                st.dataframe(pd.DataFrame(pred_data))

    with tab4:
        st.subheader("Next-Day Forecast")
        forecast = controller.forecast_next_day()
        if forecast:
            st.markdown(
                """
                This tab gives you what the stock will likely do **tomorrow**.
                It combines all our best models to forecast tomorrow's volatility and tells you whether the stock will be more or less turbulent than today.
                Use this to decide your position sizes, set stop losses, and assess risk for the next trading day.
                **Key insight**: If volatility is expected to spike, consider reducing exposure or widening your risk bands.
                """
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Predicted Volatility",
                    f"{forecast['ensemble_forecast']:.4f}",
                    help="Average prediction from all models (annualized %)"
                )

            with col2:
                confidence_range = forecast['confidence_range']
                st.metric(
                    "Confidence Range",
                    f"{confidence_range['low']:.3f} - {confidence_range['high']:.3f}",
                    help="Estimated range based on model variation"
                )

            st.markdown("**Individual Model Forecasts:**")
            model_cols = st.columns(min(len(forecast['model_forecasts']), 4))
            for i, (model_name, pred) in enumerate(forecast['model_forecasts'].items()):
                with model_cols[i % len(model_cols)]:
                    st.metric(model_name, f"{pred:.4f}")

            # --- Export forecast as CSV
            try:
                rows = []
                for model_name, pred in forecast['model_forecasts'].items():
                    rows.append({
                        'Model': model_name,
                        'Predicted_Volatility': float(pred)
                    })

                # Add ensemble row
                rows.append({
                    'Model': 'Ensemble',
                    'Predicted_Volatility': float(forecast.get('ensemble_forecast', np.nan))
                })

                df_forecast = pd.DataFrame(rows)
                # Include forecast date for context
                try:
                    df_forecast['Forecast_Date'] = pd.to_datetime(forecast.get('forecast_date')).strftime('%Y-%m-%d')
                except Exception:
                    df_forecast['Forecast_Date'] = str(forecast.get('forecast_date'))

                csv_bytes = df_forecast.to_csv(index=False).encode('utf-8')
                forecast_link = _make_download_link(csv_bytes, f"{controller.current_stock}_forecast.csv", "Download Forecast CSV")
                st.markdown(forecast_link, unsafe_allow_html=True)
            except Exception as e:
                logger.debug(f"Forecast CSV export failed: {e}")

            if forecast.get('forecast_summary'):
                st.markdown("---")
                st.markdown("### Forecast Insight")
                st.info(forecast['forecast_summary'])
                st.markdown("---")

            regime_fig = controller.get_volatility_regime_chart()
            if regime_fig:
                st.subheader("Volatility Regime Detection")
                st.plotly_chart(regime_fig, width="stretch")
                st.markdown(
                    "**Regime Detection Explanation:** This chart shows whether the stock is currently in a low, medium, or high volatility regime. "
                    "Low means calmer markets, high means more risk and larger price swings."
                )

            st.info(f"This forecast was generated on {forecast['generated_at'].strftime('%Y-%m-%d %H:%M')} using the most recent available data. Past performance does not guarantee future results.")
        else:
            st.warning("No forecast available. Make sure models are trained or loaded for this stock.")

    if admin_tab is not None:
        with admin_tab:
            _render_admin_console(controller)

    with tab3:
        st.subheader("Performance Evaluation")
        st.markdown(
            """
            This table shows how well each model predicted volatility. Read it like this:
            - Lower RMSE, MAE, and MAPE are better.
            - Higher R2 means the model explains more of what actually happened.
            - Directional Accuracy tells you how often the model got the up/down direction right.
            The best model is the one with the smallest error numbers and the highest R2.
            """
        )

        residual_fig = controller.get_residual_error_chart()
        if residual_fig:
            st.subheader("Residual / Error Analysis")
            st.plotly_chart(residual_fig, width="stretch")
            st.markdown(
                "**Residual Plot Explanation:** The top chart shows the difference between actual and predicted volatility. "
                "The bottom chart shows how often those errors happened. Smaller, more centered errors mean better forecasts."
            )

        perf_df = controller.get_performance_table()
        if not perf_df.empty:
            # Highlight best values
            styled = perf_df.style.highlight_min(
                subset=['RMSE', 'MAE', 'MAPE'], color='#90EE90'
            ).highlight_max(
                subset=['R2', 'Directional_Accuracy'], color='#90EE90'
            )
            # FIXED: Use width="stretch" instead of use_container_width=True
            st.dataframe(styled, width="stretch")

            best = perf_df.iloc[0]
            st.success(f"Best Model: **{best['Model']}** | RMSE: {best['RMSE']:.6f} | R2: {best['R2']:.4f}")

            # Download evaluation results
            csv = perf_df.to_csv(index=False).encode('utf-8')
            results_link = _make_download_link(csv, f"{controller.current_stock}_results.csv", "Download Results CSV")
            st.markdown(results_link, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Download Processed Datasets")
            processed_path = os.path.join(controller.data_dir, f"{controller.current_stock}_processed.csv")
            features_path = os.path.join(controller.data_dir, f"{controller.current_stock}_features.csv")
            downloaded = False

            col1, col2 = st.columns(2)
            if os.path.exists(processed_path):
                with open(processed_path, 'rb') as f:
                    processed_bytes = f.read()
                processed_link = _make_download_link(processed_bytes, f"{controller.current_stock}_processed.csv", "Download Processed CSV")
                with col1:
                    st.markdown(processed_link, unsafe_allow_html=True)
                downloaded = True

            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    features_bytes = f.read()
                features_link = _make_download_link(features_bytes, f"{controller.current_stock}_features.csv", "Download Features CSV")
                with col2:
                    st.markdown(features_link, unsafe_allow_html=True)
                downloaded = True

            if not downloaded:
                st.info("No processed or feature CSV files were found for this stock.")
        else:
            st.info("No evaluation results available")

    # Footer
    render_footer()


# =============================================================================
# DIRECT EXECUTION (for testing)
# =============================================================================

if __name__ == "__main__":
    # When run directly, launch the app
    create_streamlit_app()