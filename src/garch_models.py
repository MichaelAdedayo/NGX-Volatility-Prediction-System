"""
GARCH-Family Models for NSE Volatility Forecasting
Based on research: GJR-GARCH with Student-t distribution works best for NSE
"""

import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing arch - handle if not installed
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    logger.warning("arch package not installed. Install with: pip install arch")
    ARCH_AVAILABLE = False
    arch_model = None

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not installed
    def tqdm(iterable, **kwargs):
        return iterable


class GARCHSuite:
    def __init__(self, returns: pd.Series, dates: pd.Series = None):
        """
        Initialize with return series
        Returns should be in decimal form (e.g., 0.01 for 1%)
        """
        if not ARCH_AVAILABLE:
            raise ImportError("arch package required. Install: pip install arch")
            
        self.returns = returns.dropna()
        self.dates = dates if dates is not None else returns.index
        self.models = {}
        self.results = {}
        self.forecasts_df = None
        
        # Scale returns for numerical stability (ARCH package prefers this)
        self.scale = 100
        self.scaled_returns = self.returns * self.scale
        
    def fit_garch(self, p=1, q=1, dist='t') -> Dict:
        """Standard GARCH(p,q)"""
        try:
            model = arch_model(
                self.scaled_returns, 
                vol='Garch', 
                p=p, 
                q=q,
                dist=dist,
                rescale=False
            )
            result = model.fit(disp='off', show_warning=False)
            
            self.models['GARCH'] = model
            self.results['GARCH'] = result
            
            logger.info(f"GARCH({p},{q}) - AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")
            
            return {
                'model_name': 'GARCH',
                'aic': result.aic,
                'bic': result.bic,
                'loglik': result.loglikelihood,
                'params': result.params.to_dict(),
                'convergence': result.convergence
            }
        except Exception as e:
            logger.error(f"GARCH failed: {e}")
            return None
    
    def fit_egarch(self, p=1, o=1, q=1, dist='t') -> Dict:
        """EGARCH - captures asymmetric volatility"""
        try:
            model = arch_model(
                self.scaled_returns,
                vol='EGARCH',
                p=p,
                o=o,  # Asymmetric term
                q=q,
                dist=dist,
                rescale=False
            )
            result = model.fit(disp='off', show_warning=False)
            
            self.models['EGARCH'] = model
            self.results['EGARCH'] = result
            
            logger.info(f"EGARCH({p},{o},{q}) - AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")
            
            return {
                'model_name': 'EGARCH',
                'aic': result.aic,
                'bic': result.bic,
                'loglik': result.loglikelihood,
                'params': result.params.to_dict(),
                'convergence': result.convergence
            }
        except Exception as e:
            logger.error(f"EGARCH failed: {e}")
            return None
    
    def fit_gjr_garch(self, p=1, o=1, q=1, dist='t') -> Dict:
        """
        GJR-GARCH - best for NSE according to research
        Asymmetric response to positive/negative shocks
        """
        try:
            model = arch_model(
                self.scaled_returns,
                vol='GARCH',
                p=p,
                o=o,  # Asymmetric term (gamma)
                q=q,
                dist=dist,
                rescale=False
            )
            result = model.fit(disp='off', show_warning=False)
            
            self.models['GJR-GARCH'] = model
            self.results['GJR-GARCH'] = result
            
            logger.info(f"GJR-GARCH({p},{o},{q}) - AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")
            
            return {
                'model_name': 'GJR-GARCH',
                'aic': result.aic,
                'bic': result.bic,
                'loglik': result.loglikelihood,
                'params': result.params.to_dict(),
                'convergence': result.convergence
            }
        except Exception as e:
            logger.error(f"GJR-GARCH failed: {e}")
            return None
    
    def fit_harch(self, lags=[1, 5, 22], dist='t') -> Dict:
        """HARCH - Heterogeneous ARCH (Dacorogna et al.)"""
        try:
            model = arch_model(
                self.scaled_returns,
                vol='HARCH',
                lags=lags,
                dist=dist,
                rescale=False
            )
            result = model.fit(disp='off', show_warning=False)
            
            self.models['HARCH'] = model
            self.results['HARCH'] = result
            
            logger.info(f"HARCH{lags} - AIC: {result.aic:.2f}, BIC: {result.bic:.2f}")
            
            return {
                'model_name': 'HARCH',
                'aic': result.aic,
                'bic': result.bic,
                'loglik': result.loglikelihood,
                'params': result.params.to_dict(),
                'convergence': result.convergence
            }
        except Exception as e:
            logger.error(f"HARCH failed: {e}")
            return None
    
    def fit_all(self, dist='t') -> pd.DataFrame:
        """Fit all GARCH-family models and return comparison"""
        results = []
        
        models_to_fit = [
            ('GARCH(1,1)', lambda: self.fit_garch(dist=dist)),
            ('EGARCH(1,1,1)', lambda: self.fit_egarch(dist=dist)),
            ('GJR-GARCH(1,1,1)', lambda: self.fit_gjr_garch(dist=dist)),
            ('HARCH', lambda: self.fit_harch(dist=dist))
        ]
        
        for name, fit_func in models_to_fit:
            res = fit_func()
            if res:
                results.append({
                    'Model': name,
                    'AIC': res['aic'],
                    'BIC': res['bic'],
                    'Log-Likelihood': res['loglik'],
                    'Converged': res['convergence']
                })
        
        comparison_df = pd.DataFrame(results)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('AIC')
        
        return comparison_df
    
    def get_best_model(self, criterion='AIC'):
        """Get best model name based on criterion"""
        if not self.results:
            return None
        
        best_aic = float('inf')
        best_model = None
        
        for name, result in self.results.items():
            if criterion == 'AIC':
                val = result.aic
            elif criterion == 'BIC':
                val = result.bic
            
            if val < best_aic:
                best_aic = val
                best_model = name
        
        return best_model
    
    def rolling_forecast(self, model_name: str, window: int = 1000, 
                        horizon: int = 1, step: int = 1, refit_every: int = 1,
                        cache_path: str = None) -> pd.DataFrame:
        """
        Rolling one-step-ahead volatility forecast
        This is the proper way to evaluate volatility models
        """
        if model_name not in self.results:
            logger.error(f"Model {model_name} not fitted")
            return None
        
        n_obs = len(self.scaled_returns)
        forecasts = []
        actuals = []
        dates_list = []

        logger.info(f"Generating rolling forecasts for {model_name} (step={step}, refit_every={refit_every})...")

        # If cached forecast exists, return it
        if cache_path is not None and os.path.exists(cache_path):
            try:
                df = pd.read_csv(cache_path, parse_dates=['date'])
                self.forecasts_df = df
                logger.info(f"Loaded cached forecasts from {cache_path}")
                return self.forecasts_df
            except Exception:
                logger.debug("Failed to load cache, regenerating forecasts")

        # Initial fit on the first window
        init_i = window
        try:
            train_data = self.scaled_returns.iloc[init_i - window:init_i]
            if 'EGARCH' in model_name:
                vol_spec = 'EGARCH'
                o = 1
            elif 'GJR' in model_name:
                vol_spec = 'GARCH'
                o = 1
            else:
                vol_spec = 'GARCH'
                o = 0

            model = arch_model(
                train_data,
                vol=vol_spec,
                p=1, o=o, q=1,
                dist='t',
                rescale=False
            )
            res = model.fit(disp='off', show_warning=False)
        except Exception as e:
            logger.error(f"Initial GARCH fit failed: {e}")
            return None

        # Rolling loop with stride and periodic refitting
        for pos_idx, i in enumerate(tqdm(range(window, n_obs - horizon + 1, step), desc="Forecasting")):
            try:
                # Refit if requested periodically
                if refit_every > 1 and (pos_idx % refit_every == 0) and i != init_i:
                    train_data = self.scaled_returns.iloc[i - window:i]
                    model = arch_model(
                        train_data,
                        vol=vol_spec,
                        p=1, o=o, q=1,
                        dist='t',
                        rescale=False
                    )
                    res = model.fit(disp='off', show_warning=False)

                # Forecast using the last fitted result
                forecast = res.forecast(horizon=horizon)

                # Annualized volatility (convert back from scaled)
                pred_vol = np.sqrt(forecast.variance.values[-1, 0]) / self.scale * np.sqrt(252)

                # Actual realized volatility (absolute return as proxy)
                actual_vol = np.abs(self.returns.iloc[i + horizon - 1]) * np.sqrt(252)

                forecasts.append(pred_vol)
                actuals.append(actual_vol)
                dates_list.append(self.dates.iloc[i + horizon - 1])

            except Exception as e:
                logger.debug(f"Rolling forecast step failed at i={i}: {e}")
                continue
        
        self.forecasts_df = pd.DataFrame({
            'date': dates_list,
            'forecasted_vol': forecasts,
            'actual_vol_proxy': actuals
        })
        
        return self.forecasts_df
    
    def get_conditional_volatility(self, model_name: str) -> pd.Series:
        """Get fitted conditional volatility (in-sample)"""
        if model_name not in self.results:
            return None
        
        # Convert back from scaled values, annualize
        cond_vol = self.results[model_name].conditional_volatility / self.scale * np.sqrt(252)
        return pd.Series(cond_vol, index=self.returns.index)
    
    def get_model_summary(self, model_name: str):
        """Get full model summary"""
        if model_name in self.results:
            return self.results[model_name].summary()
        return None


def evaluate_garch_forecast(forecasts_df: pd.DataFrame) -> Dict:
    """Evaluate GARCH forecast performance"""
    if forecasts_df is None or forecasts_df.empty:
        return {}
    
    actual = forecasts_df['actual_vol_proxy']
    predicted = forecasts_df['forecasted_vol']
    
    # Standard metrics
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual - predicted))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    
    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Directional accuracy (did we predict increase/decrease correctly?)
    actual_diff = np.sign(np.diff(actual))
    pred_diff = np.sign(np.diff(predicted))
    dir_accuracy = np.mean(actual_diff == pred_diff) * 100
    
    # QLIKE loss (preferred for volatility)
    qlike = np.mean(np.log(predicted**2) + (actual**2 / (predicted**2)))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': dir_accuracy,
        'QLIKE': qlike
    }


def analyze_single_stock(stock_name: str, data_path: str = "data/processed") -> Dict:
    """Analyze a single stock with GARCH models"""
    file_path = os.path.join(data_path, f"{stock_name}_processed.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Analyzing {stock_name}")
    logger.info(f"{'='*60}")
    
    # Load data
    df = pd.read_csv(file_path, parse_dates=['Date'])
    logger.info(f"Loaded {len(df)} observations")
    
    # Initialize GARCH
    garch = GARCHSuite(df['log_return'], df['Date'])
    
    # Fit all models
    comparison = garch.fit_all(dist='t')
    logger.info(f"\nModel Comparison:\n{comparison.to_string(index=False)}")
    
    # Get best model
    best_model = garch.get_best_model('AIC')
    logger.info(f"\nBest model: {best_model}")
    
    # Generate forecasts
    forecasts = garch.rolling_forecast(best_model, window=500)
    metrics = evaluate_garch_forecast(forecasts)
    
    logger.info(f"\nForecast Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric:20s}: {value:.6f}")
    
    # Save results
    output_dir = "results/garch_output"
    os.makedirs(output_dir, exist_ok=True)
    
    comparison.to_csv(f"{output_dir}/{stock_name}_model_comparison.csv", index=False)
    forecasts.to_csv(f"{output_dir}/{stock_name}_forecasts.csv", index=False)
    
    # Return summary for cross-stock comparison
    return {
        'stock': stock_name,
        'best_model': best_model,
        'aic': comparison.iloc[0]['AIC'] if not comparison.empty else None,
        'rmse': metrics['RMSE'],
        'mae': metrics['MAE'],
        'r2': metrics['R2'],
        'directional_accuracy': metrics['Directional_Accuracy']
    }


# Example usage: Analyze ALL stocks
if __name__ == "__main__":
    # List of all your NSE stocks
    ALL_STOCKS = [
        'DANGSUG', 'DANGCEM', 'MTNN', 'GTCO', 'SEPLAT',
        'AIRTEL', 'INTERBREW', 'FIRSTHOLDCO', 'ETI', 'ZENITH',
        'CWG', 'NESTLE', 'NB', 'ACCESS', 'WAPCO'
    ]
    
    data_path = "data/processed"
    
    # Check if processed data exists
    available_stocks = []
    for stock in ALL_STOCKS:
        if os.path.exists(os.path.join(data_path, f"{stock}_processed.csv")):
            available_stocks.append(stock)
    
    if not available_stocks:
        print("No processed data found!")
        print(f"Expected files in {data_path}/")
        print("Run data preprocessing first.")
        exit(1)
    
    print(f"Found {len(available_stocks)} stocks: {available_stocks}")
    
    # Analyze all stocks
    all_results = []
    for stock in available_stocks:
        try:
            result = analyze_single_stock(stock, data_path)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze {stock}: {e}")
    
    # Create cross-stock summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        print("\n" + "="*80)
        print("CROSS-STOCK GARCH ANALYSIS SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv("results/garch_output/all_stocks_garch_summary.csv", index=False)
        
        # Count best model occurrences
        print(f"\nBest Model Distribution:")
        print(summary_df['best_model'].value_counts().to_string())
        
        # Average performance
        print(f"\nAverage Performance Across All Stocks:")
        print(f"  RMSE: {summary_df['rmse'].mean():.6f}")
        print(f"  MAE:  {summary_df['mae'].mean():.6f}")
        print(f"  R²:   {summary_df['r2'].mean():.6f}")