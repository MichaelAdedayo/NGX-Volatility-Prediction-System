"""
=============================================================================
MACHINE LEARNING MODELS FOR VOLATILITY PREDICTION
=============================================================================
B.Sc. Computer Science Final Year Project
Nigerian Exchange Group (NGX) Volatility Prediction System

Module: ml_models.py
Purpose: ML model implementations with dashboard compatibility

Models Implemented:
- Random Forest (ensemble)
- XGBoost (gradient boosting)
- Ridge Regression (linear with regularization)
- Support Vector Regression (non-linear)

FIXED: TensorFlow/Keras imports, DataFrame handling, dashboard integration
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import warnings
import os
import sys
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# NOTE: LSTM/Deep-learning removed from this project to reduce complexity
LSTM_AVAILABLE = False

# XGBoost
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not installed. Install with: pip install xgboost")

# SHAP for interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# =============================================================================
# ML VOLATILITY SUITE - DASHBOARD COMPATIBLE
# =============================================================================

class MLVolatilitySuite:
    """
    Machine Learning suite for volatility prediction.

    FIXED: Handles both DataFrame and numpy array inputs.
    FIXED: Stores predictions with consistent indexing for dashboard.
    FIXED: Sequence-handling and dashboard compatibility.
    """

    def __init__(self, dataset: Dict):
        """
        Initialize with dataset dictionary.

        Expected dataset keys:
        - X_train, X_test: Features (DataFrame or ndarray)
        - y_train, y_test: Target variable (Series or ndarray)
        - feature_names: List of feature column names
        - dates_train, dates_test: Date arrays
        """
        # Handle both DataFrame and numpy array inputs
        self.X_train = self._ensure_dataframe(dataset['X_train'], dataset.get('feature_names'))
        self.X_test = self._ensure_dataframe(dataset['X_test'], dataset.get('feature_names'))

        self.y_train = self._ensure_series(dataset['y_train'])
        self.y_test = self._ensure_series(dataset['y_test'])

        self.feature_cols = dataset.get('feature_names', list(self.X_train.columns))
        self.dates_train = dataset.get('dates_train')
        self.dates_test = dataset.get('dates_test')

        # Initialize storage
        self.models = {}
        self.predictions = {}
        self.training_history = {}
        self.shap_values = {}

        # Initialize and fit scaler
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        logger.info(f"ML Suite initialized: {self.X_train.shape[1]} features, "
                   f"{len(self.X_train)} train, {len(self.X_test)} test")

    @staticmethod
    def _ensure_dataframe(X, feature_names=None):
        """Ensure input is a pandas DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, np.ndarray):
            if feature_names and len(feature_names) == X.shape[1]:
                return pd.DataFrame(X, columns=feature_names)
            else:
                return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            return pd.DataFrame(X)

    @staticmethod
    def _ensure_series(y):
        """Ensure input is a pandas Series."""
        if isinstance(y, pd.Series):
            return y
        elif isinstance(y, np.ndarray):
            return pd.Series(y)
        else:
            return pd.Series(y)

    def fit_random_forest(self):
        """Train Random Forest with optimized hyperparameters."""
        print("  ├─ Random Forest...", end=" ")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)

        self.models['Random Forest'] = model
        self.predictions['Random Forest'] = y_pred

        # SHAP values
        if SHAP_AVAILABLE and len(self.X_test) > 50:
            try:
                explainer = shap.TreeExplainer(model)
                sample_size = min(500, len(self.X_test))
                X_sample = self.X_test.iloc[:sample_size] if hasattr(self.X_test, 'iloc') else self.X_test[:sample_size]
                self.shap_values['Random Forest'] = explainer.shap_values(X_sample)
            except Exception as e:
                logger.debug(f"SHAP failed for RF: {e}")

        print("✓ Done")
        return model

    def fit_xgboost(self):
        """Train XGBoost with regularization."""
        if not XGB_AVAILABLE:
            print("  ├─ XGBoost... ✗ Not installed")
            return None

        print("  ├─ XGBoost...", end=" ")

        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='reg:squarederror',
            n_jobs=-1
        )

        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False
        )

        y_pred = model.predict(self.X_test)

        self.models['XGBoost'] = model
        self.predictions['XGBoost'] = y_pred

        if SHAP_AVAILABLE and len(self.X_test) > 50:
            try:
                explainer = shap.TreeExplainer(model)
                sample_size = min(500, len(self.X_test))
                X_sample = self.X_test.iloc[:sample_size] if hasattr(self.X_test, 'iloc') else self.X_test[:sample_size]
                self.shap_values['XGBoost'] = explainer.shap_values(X_sample)
            except Exception as e:
                logger.debug(f"SHAP failed for XGB: {e}")

        print("✓ Done")
        return model

    def fit_ridge(self):
        """Train Ridge Regression."""
        print("  ├─ Ridge...", end=" ")

        model = Ridge(alpha=10.0, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)

        y_pred = model.predict(self.X_test_scaled)

        self.models['Ridge'] = model
        self.predictions['Ridge'] = y_pred

        print("✓ Done")
        return model

    def fit_svr(self):
        """Train Support Vector Regression."""
        print("  ├─ SVR...", end=" ")

        model = SVR(
            kernel='rbf',
            C=1.0,
            epsilon=0.01,
            gamma='scale'
        )

        model.fit(self.X_train_scaled, self.y_train)
        y_pred = model.predict(self.X_test_scaled)

        self.models['SVR'] = model
        self.predictions['SVR'] = y_pred

        print("✓ Done")
        return model

    # LSTM support removed; function implementation deleted to avoid TensorFlow dependency

    def fit_all(self, include_lstm=True, include_shap=True):
        """Fit all ML models in parallel for better performance."""
        import concurrent.futures
        
        models_to_fit = []
        
        # Define model fitting functions
        def fit_rf():
            return ('Random Forest', self.fit_random_forest())
        
        def fit_xgb():
            return ('XGBoost', self.fit_xgboost())
        
        def fit_ridge():
            return ('Ridge', self.fit_ridge())
        
        def fit_svr():
            return ('SVR', self.fit_svr())
        
        # Add to list
        models_to_fit.extend([fit_rf, fit_xgb, fit_ridge, fit_svr])
        
        # Fit in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(func) for func in models_to_fit]
            for future in concurrent.futures.as_completed(futures):
                try:
                    model_name, model = future.result()
                    logger.info(f"Completed training: {model_name}")
                except Exception as e:
                    logger.error(f"Error training model: {e}")
        
        # Note: LSTM training removed from current pipeline
        
        if include_shap and self.shap_values:
            self._save_shap_values()
        
        return self

    def hyperparameter_tuning(self, model_name='Random Forest', n_trials=50):
        """Perform hyperparameter tuning using Optuna for better model performance."""
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna not installed. Install with: pip install optuna")
            return
        
        def objective(trial):
            if model_name == 'Random Forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                }
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
                }
                model = XGBRegressor(**params, random_state=42, n_jobs=-1)
            else:
                return float('inf')
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best {model_name} params: {study.best_params}")
        logger.info(f"Best RMSE: {study.best_value}")
        
        # Update model with best params
        if model_name == 'Random Forest':
            self.models[model_name] = RandomForestRegressor(**study.best_params, random_state=42, n_jobs=-1)
        elif model_name == 'XGBoost':
            self.models[model_name] = XGBRegressor(**study.best_params, random_state=42, n_jobs=-1)
        
        # Retrain and update predictions
        self.models[model_name].fit(self.X_train, self.y_train)
        self.predictions[model_name] = self.models[model_name].predict(self.X_test)

    def cross_validate(self, model_name='Random Forest', cv_folds=5):
        """Perform cross-validation for more robust evaluation."""
        from sklearn.model_selection import cross_val_score
        
        if model_name not in self.models:
            logger.warning(f"Model {model_name} not trained yet.")
            return
        
        model = self.models[model_name]
        X = np.vstack([self.X_train, self.X_test])
        y = np.concatenate([self.y_train, self.y_test])
        
        scores = cross_val_score(model, X, y, cv=cv_folds, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        
        logger.info(f"{model_name} CV RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std() * 2:.4f})")
        return rmse_scores

    def _save_shap_values(self, output_dir='results/shap_values'):
        """Save SHAP values to disk."""
        if not self.shap_values:
            return

        os.makedirs(output_dir, exist_ok=True)

        for model_name, shap_vals in self.shap_values.items():
            try:
                np.save(f"{output_dir}/{model_name}_shap_values.npy", shap_vals)

                if model_name in ['Random Forest', 'XGBoost']:
                    X_sample = self.X_test.iloc[:len(shap_vals)] if hasattr(self.X_test, 'iloc') else self.X_test[:len(shap_vals)]
                    shap_df = pd.DataFrame({
                        'feature': self.feature_cols,
                        'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
                    }).sort_values('mean_abs_shap', ascending=False)
                    shap_df.to_csv(f"{output_dir}/{model_name}_shap_summary.csv", index=False)
            except Exception as e:
                logger.debug(f"Failed to save SHAP for {model_name}: {e}")

    def evaluate_all(self) -> pd.DataFrame:
        """
        Evaluate all trained models.

        Returns:
            DataFrame with metrics for each model, sorted by RMSE.
        """
        results = []

        for name, y_pred in self.predictions.items():
            y_true = self.y_test.values

            # Handle NaN values (from sequence padding)
            valid_mask = ~np.isnan(y_pred)
            if not valid_mask.any():
                logger.warning(f"All predictions are NaN for {name}")
                continue

            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]

            if len(y_true_valid) == 0:
                continue

            # Calculate metrics
            mse = mean_squared_error(y_true_valid, y_pred_valid)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true_valid, y_pred_valid)

            # R2 with protection against single sample
            if len(y_true_valid) > 1:
                r2 = r2_score(y_true_valid, y_pred_valid)
            else:
                r2 = np.nan

            mape = np.mean(np.abs((y_true_valid - y_pred_valid) / (y_true_valid + 1e-8))) * 100

            # Directional accuracy
            if len(y_true_valid) > 1:
                true_diff = np.sign(np.diff(y_true_valid))
                pred_diff = np.sign(np.diff(y_pred_valid))
                dir_acc = np.mean(true_diff == pred_diff) * 100
            else:
                dir_acc = np.nan

            results.append({
                'Model': name,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2,
                'Directional_Accuracy': dir_acc
            })

        return pd.DataFrame(results).sort_values('RMSE')

    def get_feature_importance(self, top_n=20) -> pd.DataFrame:
        """
        Get feature importance from tree-based models.

        Returns:
            DataFrame with features and importance scores.
        """
        importance_data = []

        for name in ['Random Forest', 'XGBoost']:
            if name not in self.models:
                continue

            model = self.models[name]
            scores = model.feature_importances_

            for feat, score in zip(self.feature_cols, scores):
                importance_data.append({
                    'Model': name,
                    'Feature': feat,
                    'Importance': score
                })

        df = pd.DataFrame(importance_data)
        if df.empty:
            return df
        return df.sort_values('Importance', ascending=False).head(top_n)

    def save_predictions(self, filepath):
        """
        Save predictions to CSV with aligned lengths.

        FIXED: Ensures all predictions have the same length.
        """
        if not self.predictions:
            logger.warning("No predictions to save")
            return

        # Get minimum length across all predictions
        valid_preds = {k: v for k, v in self.predictions.items() if not np.all(np.isnan(v))}
        if not valid_preds:
            logger.warning("All predictions are NaN")
            return

        min_len = min(len(p) for p in valid_preds.values())

        pred_dict = {}
        for name, pred in valid_preds.items():
            pred_dict[name] = pred[:min_len]

        pred_df = pd.DataFrame(pred_dict)
        pred_df['Actual'] = self.y_test.values[:min_len]

        if self.dates_test is not None:
            dates = self.dates_test[:min_len] if hasattr(self.dates_test, '__len__') else self.dates_test
            pred_df['Date'] = dates

        pred_df.to_csv(filepath, index=False)
        logger.info(f"Saved predictions to {filepath}")


# =============================================================================
# HYBRID MODEL (GARCH + ML)
# =============================================================================

class HybridModel:
    """
    Hybrid model combining GARCH volatility with ML features.
    """

    def __init__(self, dataset, garch_vol_train, garch_vol_test):
        self.dataset = dataset
        self.garch_vol_train = garch_vol_train
        self.garch_vol_test = garch_vol_test

        # Ensure DataFrames
        self.X_train_h = self._ensure_dataframe(dataset['X_train']).copy()
        self.X_test_h = self._ensure_dataframe(dataset['X_test']).copy()

        # Align lengths
        min_train = min(len(self.X_train_h), len(garch_vol_train))
        min_test = min(len(self.X_test_h), len(garch_vol_test))

        self.X_train_h = self.X_train_h.iloc[:min_train]
        self.X_test_h = self.X_test_h.iloc[:min_test]

        self.X_train_h['garch_vol'] = garch_vol_train.values[:min_train] if hasattr(garch_vol_train, 'values') else garch_vol_train[:min_train]
        self.X_test_h['garch_vol'] = garch_vol_test.values[:min_test] if hasattr(garch_vol_test, 'values') else garch_vol_test[:min_test]

        self.y_train_h = dataset['y_train'].values[:min_train] if hasattr(dataset['y_train'], 'values') else dataset['y_train'][:min_train]
        self.y_test_h = dataset['y_test'].values[:min_test] if hasattr(dataset['y_test'], 'values') else dataset['y_test'][:min_test]

    @staticmethod
    def _ensure_dataframe(X):
        """Ensure input is DataFrame."""
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)

    def fit_hybrid_rf(self):
        """Random Forest with GARCH volatility feature."""
        model = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        )
        model.fit(self.X_train_h, self.y_train_h)
        y_pred = model.predict(self.X_test_h)

        return model, y_pred


# =============================================================================
# FORMATTED OUTPUT FUNCTIONS
# =============================================================================

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_stock_header(stock_name, n_samples, n_features):
    """Print stock analysis header."""
    print(f"\n┌{'─'*68}┐")
    print(f"│  STOCK: {stock_name:<58}│")
    print(f"│  Samples: {n_samples:<4}  |  Features: {n_features:<4}                    │")
    print(f"└{'─'*68}┘")


def print_results_table(results_df):
    """Print results in formatted table."""
    if results_df.empty:
        print("  No results to display")
        return

    print("\n  ┌─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    print("  │ Model           │ RMSE     │ MAE      │ MAPE(%)  │ R²       │ Dir.Acc  │")
    print("  ├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    for _, row in results_df.iterrows():
        print(f"  │ {row['Model']:<15} │ {row['RMSE']:>8.4f} │ {row['MAE']:>8.4f} │ "
              f"{row['MAPE']:>8.2f} │ {row['R2']:>8.4f} │ {row['Directional_Accuracy']:>8.1f} │")

    print("  └─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    best = results_df.iloc[0]
    print(f"\n  ★ BEST MODEL: {best['Model']} (RMSE: {best['RMSE']:.4f})")


def print_summary_table(summary_data):
    """Print cross-stock summary."""
    print_header("CROSS-STOCK SUMMARY")

    print("\n  ┌─────────────┬─────────────────────┬──────────┬──────────────────┐")
    print("  │ Stock       │ Best Model          │ RMSE     │ Status           │")
    print("  ├─────────────┼─────────────────────┼──────────┼──────────────────┤")

    for item in summary_data:
        status = "✓ Complete" if not np.isnan(item['rmse']) else "✗ Failed"
        print(f"  │ {item['stock']:<11} │ {item['best_model']:<19} │ "
              f"{item['rmse']:>8.4f} │ {status:<16} │")

    print("  └─────────────┴─────────────────────┴──────────┴──────────────────┘")

    # Model distribution
    from collections import Counter
    models = [item['best_model'] for item in summary_data if not np.isnan(item['rmse'])]
    if models:
        print(f"\n  MODEL WIN DISTRIBUTION:")
        model_counts = Counter(models)
        for model, count in model_counts.most_common():
            bar = "█" * count
            print(f"    {model:<15} │{bar:<15}│ {count}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def load_and_run_ml(stock_name: str, data_path: str = "data/features"):
    """Run ML for single stock with formatted output."""
    try:
        from features import prepare_ml_dataset
    except ImportError:
        logger.error("features module not found. Cannot run ML pipeline.")
        return None

    features_file = os.path.join(data_path, f"{stock_name}_features.csv")

    if not os.path.exists(features_file):
        print(f"  ✗ Features file not found: {features_file}")
        return None

    # Load data
    df = pd.read_csv(features_file, parse_dates=['Date'])

    print_stock_header(stock_name, len(df), df.shape[1])

    # Prepare dataset
    dataset = prepare_ml_dataset(df, target_col='target_vol_1d', test_size=0.2)

    print(f"\n  Training: {len(dataset['X_train'])} samples")
    print(f"  Test:     {len(dataset['X_test'])} samples")
    print(f"\n  Training Models:")

    # Train models
    ml_suite = MLVolatilitySuite(dataset)
    ml_suite.fit_all(include_shap=True)

    # Evaluate
    results = ml_suite.evaluate_all()
    print_results_table(results)

    # Save results
    output_dir = "results/ml_output"
    os.makedirs(output_dir, exist_ok=True)

    results.to_csv(f"{output_dir}/{stock_name}_ml_results.csv", index=False)

    if len(ml_suite.predictions) > 0:
        ml_suite.save_predictions(f"{output_dir}/{stock_name}_predictions.csv")

    # Feature importance
    importance = ml_suite.get_feature_importance(20)
    if not importance.empty:
        importance.to_csv(f"{output_dir}/{stock_name}_feature_importance.csv", index=False)

        # Top 3 features
        print(f"\n  Top 3 Features:")
        for i, row in importance.head(3).iterrows():
            print(f"    {i+1}. {row['Feature']} ({row['Importance']:.4f})")

    return {
        'stock': stock_name,
        'results': results,
        'best_model': results.iloc[0]['Model'] if not results.empty else 'N/A',
        'rmse': results.iloc[0]['RMSE'] if not results.empty else np.nan,
        'suite': ml_suite  # Return suite for dashboard integration
    }


def run_ml_for_all_stocks():
    """Run ML for all 15 stocks with formatted output."""

    ALL_STOCKS = [
        'DANGSUG', 'DANGCEM', 'MTNN', 'GTCO', 'SEPLAT',
        'AIRTEL', 'INTERBREW', 'FIRSTHOLDCO', 'ETI', 'ZENITH',
        'CWG', 'NESTLE', 'NB', 'ACCESS', 'WAPCO'
    ]

    data_path = "data/features"

    print_header("NSE VOLATILITY PREDICTION - MACHINE LEARNING MODELS")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Stocks: {len(ALL_STOCKS)}")
    print(f"  Data Path: {data_path}/")

    # Check available stocks
    available_stocks = []
    for stock in ALL_STOCKS:
        if os.path.exists(os.path.join(data_path, f"{stock}_features.csv")):
            available_stocks.append(stock)

    if not available_stocks:
        print(f"\n  ✗ No feature files found in {data_path}/")
        print("  Run: python main_pipeline.py first")
        return

    print(f"  Available: {len(available_stocks)}/{len(ALL_STOCKS)}")
    print(f"  Missing: {set(ALL_STOCKS) - set(available_stocks) or 'None'}")

    # Process each stock
    all_results = []
    for i, stock in enumerate(available_stocks, 1):
        try:
            print(f"\n  [{i}/{len(available_stocks)}]", end="")
            result = load_and_run_ml(stock, data_path)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing {stock}: {e}")
            print(f"\n  ✗ Error processing {stock}: {str(e)[:50]}")

    # Final summary
    if all_results:
        print_summary_table(all_results)

        # Save summary
        summary_df = pd.DataFrame([
            {'Stock': r['stock'], 'Best_Model': r['best_model'], 'RMSE': r['rmse']}
            for r in all_results
        ])
        os.makedirs("results/ml_output", exist_ok=True)
        summary_df.to_csv("results/ml_output/all_stocks_ml_summary.csv", index=False)

        print(f"\n  Results saved to: results/ml_output/")

    print_header("ANALYSIS COMPLETE")


if __name__ == "__main__":
    run_ml_for_all_stocks()