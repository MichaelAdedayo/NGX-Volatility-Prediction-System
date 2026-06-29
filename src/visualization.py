"""
=============================================================================
VISUALIZATION MODULE FOR NGX VOLATILITY PREDICTION SYSTEM
=============================================================================
B.Sc. Computer Science Final Year Project
Nigerian Exchange Group (NGX) Volatility Prediction System

Module: visualization.py
Purpose: Generate publication-ready figures and charts

This module creates:
- Time series plots of volatility forecasts vs actual
- Model comparison bar charts
- Feature importance visualizations
- Residual analysis plots

All figures are saved in high-resolution format suitable for academic
publication in Chapter 4 of the thesis.
=============================================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Set publication-ready style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
sns.set_palette("husl")

# Academic color scheme
COLORS = {
    'actual': '#000000',
    'random_forest': '#E69F00',
    'xgboost': '#56B4E9',
    'ridge': '#009E73',
    'svr': '#F0E442',
    'garch': '#D55E00',
    'error_band': '#CC79A7'
}


def create_stock_report_figures(
    stock_name: str,
    dates: np.ndarray,
    actual: np.ndarray,
    predictions: Dict[str, np.ndarray],
    results_df: pd.DataFrame,
    feature_importance: Optional[pd.DataFrame] = None,
    save: bool = True,
    output_dir: str = "results/figures"
) -> Dict[str, plt.Figure]:
    """
    Generate comprehensive figure set for a single stock analysis.

    Creates:
    1. Volatility forecast comparison (time series)
    2. Model performance bar chart
    3. Feature importance (if available)
    4. Residual analysis

    Args:
        stock_name: NGX ticker symbol
        dates: Test period dates
        actual: Actual realized volatility
        predictions: Dictionary of model predictions
        results_df: Performance metrics DataFrame
        feature_importance: Feature importance DataFrame
        save: Whether to save figures to disk
        output_dir: Directory for saved figures

    Returns:
        Dictionary of generated figures
    """
    figures = {}

    try:
        # Figure 1: Time Series Forecast Comparison
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        # Plot actual
        ax1.plot(dates, actual, color=COLORS['actual'], linewidth=2, 
                label='Actual Volatility', alpha=0.9)

        # Plot predictions
        for model_name, preds in predictions.items():
            n = min(len(dates), len(preds))
            color_key = model_name.lower().replace(' ', '_').replace('-', '_')
            color = COLORS.get(color_key, '#999999')
            ax1.plot(dates[:n], preds[:n], color=color, linewidth=1.5, 
                    label=model_name, alpha=0.7)

        ax1.set_xlabel('Date', fontsize=11)
        ax1.set_ylabel('Annualized Volatility', fontsize=11)
        ax1.set_title(f'{stock_name}: Volatility Forecasts vs Actual', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        figures['forecasts'] = fig1

        # Figure 2: Model Performance Comparison
        fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE comparison
        if 'RMSE' in results_df.columns:
            models = results_df['Model'].values
            rmse_vals = results_df['RMSE'].values
            colors_list = [COLORS.get(m.lower().replace(' ', '_').replace('-', '_'), '#999999') 
                          for m in models]

            bars1 = ax2a.barh(models, rmse_vals, color=colors_list, alpha=0.8)
            ax2a.set_xlabel('RMSE (lower is better)', fontsize=11)
            ax2a.set_title('Model Performance: RMSE', fontsize=12, fontweight='bold')
            ax2a.grid(True, alpha=0.3, axis='x')

            # Add value labels
            for bar, val in zip(bars1, rmse_vals):
                ax2a.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                         f'{val:.4f}', va='center', fontsize=9)

        # R2 comparison
        if 'R2' in results_df.columns:
            r2_vals = results_df['R2'].values
            colors_list = [COLORS.get(m.lower().replace(' ', '_').replace('-', '_'), '#999999') 
                          for m in models]

            bars2 = ax2b.barh(models, r2_vals, color=colors_list, alpha=0.8)
            ax2b.set_xlabel('R² (higher is better)', fontsize=11)
            ax2b.set_title('Model Performance: R-squared', fontsize=12, fontweight='bold')
            ax2b.grid(True, alpha=0.3, axis='x')
            ax2b.set_xlim(0, 1)

            # Add value labels
            for bar, val in zip(bars2, r2_vals):
                ax2b.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                         f'{val:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        figures['performance'] = fig2

        # Figure 3: Feature Importance (if available)
        if feature_importance is not None and not feature_importance.empty:
            fig3, ax3 = plt.subplots(figsize=(10, 8))

            # Get top 20 features
            top_features = feature_importance.head(20)

            ax3.barh(range(len(top_features)), top_features['Importance'].values, 
                    color='#0173B2', alpha=0.8)
            ax3.set_yticks(range(len(top_features)))
            ax3.set_yticklabels(top_features['Feature'].values, fontsize=9)
            ax3.set_xlabel('Importance Score', fontsize=11)
            ax3.set_title(f'{stock_name}: Top 20 Feature Importances (Random Forest)', 
                         fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='x')
            ax3.invert_yaxis()

            plt.tight_layout()
            figures['feature_importance'] = fig3

        # Figure 4: Residual Analysis for Best Model
        if not results_df.empty:
            best_model = results_df.iloc[0]['Model']
            if best_model in predictions:
                fig4, axes = plt.subplots(2, 2, figsize=(12, 10))

                best_preds = predictions[best_model][:len(actual)]
                residuals = actual[:len(best_preds)] - best_preds

                # Residuals over time
                axes[0, 0].plot(dates[:len(residuals)], residuals, color='#E69F00', alpha=0.7)
                axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[0, 0].set_xlabel('Date')
                axes[0, 0].set_ylabel('Residual')
                axes[0, 0].set_title(f'Residuals Over Time ({best_model})')
                axes[0, 0].grid(True, alpha=0.3)

                # Residual histogram
                axes[0, 1].hist(residuals, bins=30, color='#56B4E9', alpha=0.7, edgecolor='black')
                axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                axes[0, 1].set_xlabel('Residual')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Residual Distribution')
                axes[0, 1].grid(True, alpha=0.3)

                # Q-Q plot
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=axes[1, 0])
                axes[1, 0].set_title('Q-Q Plot (Normality Check)')
                axes[1, 0].grid(True, alpha=0.3)

                # Predicted vs Actual
                axes[1, 1].scatter(best_preds, actual[:len(best_preds)], 
                                  alpha=0.5, color='#009E73', edgecolor='black', linewidth=0.5)
                min_val = min(best_preds.min(), actual[:len(best_preds)].min())
                max_val = max(best_preds.max(), actual[:len(best_preds)].max())
                axes[1, 1].plot([min_val, max_val], [min_val, max_val], 
                               'r--', alpha=0.7, label='Perfect Prediction')
                axes[1, 1].set_xlabel('Predicted Volatility')
                axes[1, 1].set_ylabel('Actual Volatility')
                axes[1, 1].set_title(f'Predicted vs Actual ({best_model})')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout()
                figures['residuals'] = fig4

        # Save figures
        if save:
            os.makedirs(output_dir, exist_ok=True)

            for fig_name, fig in figures.items():
                filepath = os.path.join(output_dir, f"{stock_name}_{fig_name}.png")
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Saved figure: {filepath}")

                # Also save as PDF for publication
                pdf_path = os.path.join(output_dir, f"{stock_name}_{fig_name}.pdf")
                fig.savefig(pdf_path, bbox_inches='tight')

        return figures

    except Exception as e:
        logger.error(f"Error creating figures for {stock_name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def create_cross_sectional_figures(
    consolidated_results: pd.DataFrame,
    output_dir: str = "results/figures"
) -> Dict[str, plt.Figure]:
    """
    Generate cross-sectional analysis figures across all stocks.

    Args:
        consolidated_results: DataFrame with all stock results
        output_dir: Directory for saved figures

    Returns:
        Dictionary of generated figures
    """
    figures = {}

    try:
        if consolidated_results.empty:
            logger.warning("No data for cross-sectional figures")
            return {}

        # Figure 1: Model Performance Distribution
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        if 'Model' in consolidated_results.columns and 'RMSE' in consolidated_results.columns:
            model_rmse = consolidated_results.groupby('Model')['RMSE'].apply(list).to_dict()

            positions = range(len(model_rmse))
            bp = ax1.boxplot(model_rmse.values(), positions=positions, patch_artist=True)

            for patch, model in zip(bp['boxes'], model_rmse.keys()):
                color_key = model.lower().replace(' ', '_').replace('-', '_')
                patch.set_facecolor(COLORS.get(color_key, '#999999'))
                patch.set_alpha(0.7)

            ax1.set_xticks(positions)
            ax1.set_xticklabels(model_rmse.keys(), rotation=45, ha='right')
            ax1.set_ylabel('RMSE Distribution', fontsize=11)
            ax1.set_title('Cross-Stock Model Performance Distribution', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            figures['cross_sectional_rmse'] = fig1

        # Figure 2: Best Model Frequency
        if 'Stock' in consolidated_results.columns and 'Model' in consolidated_results.columns:
            fig2, ax2 = plt.subplots(figsize=(10, 6))

            # Find best model per stock
            best_models = consolidated_results.loc[
                consolidated_results.groupby('Stock')['RMSE'].idxmin()
            ]['Model'].value_counts()

            colors_list = [COLORS.get(m.lower().replace(' ', '_').replace('-', '_'), '#999999') 
                          for m in best_models.index]

            bars = ax2.bar(best_models.index, best_models.values, color=colors_list, alpha=0.8)
            ax2.set_xlabel('Model', fontsize=11)
            ax2.set_ylabel('Number of Stocks Won', fontsize=11)
            ax2.set_title('Model Dominance Across NGX Stocks', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, val in zip(bars, best_models.values):
                ax2.text(bar.get_x() + bar.get_width()/2, val + 0.1, 
                        str(val), ha='center', fontsize=10, fontweight='bold')

            plt.tight_layout()
            figures['model_dominance'] = fig2

        # Save figures
        os.makedirs(output_dir, exist_ok=True)

        for fig_name, fig in figures.items():
            filepath = os.path.join(output_dir, f"cross_sectional_{fig_name}.png")
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cross-sectional figure: {filepath}")

        return figures

    except Exception as e:
        logger.error(f"Error creating cross-sectional figures: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def create_dashboard_summary_chart(
    stock_name: str,
    current_vol: float,
    predicted_vol: float,
    risk_level: str,
    recommendation: str
) -> plt.Figure:
    """
    Create a summary gauge chart for dashboard display.

    Args:
        stock_name: Stock ticker
        current_vol: Current volatility level
        predicted_vol: Predicted volatility
        risk_level: Risk assessment string
        recommendation: Investment recommendation

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Create a simple bar comparison
    categories = ['Current\nVolatility', 'Predicted\nVolatility']
    values = [current_vol * 100, predicted_vol * 100]  # Convert to percentage
    colors = ['#E69F00', '#56B4E9']

    bars = ax.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add risk level as text
    risk_color = {'LOW': 'green', 'MODERATE': 'orange', 'HIGH': 'red', 'VERY HIGH': 'darkred'}.get(
        risk_level.upper(), 'gray')

    ax.text(0.5, 0.95, f'Risk Level: {risk_level}', 
            transform=ax.transAxes, ha='center', va='top',
            fontsize=14, fontweight='bold', color=risk_color,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('Annualized Volatility (%)', fontsize=11)
    ax.set_title(f'{stock_name}: Volatility Summary', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


# --- Additional lightweight Plotly visualizations for dashboard (return distribution, rolling vol)

def plot_return_distribution_plotly(df: pd.DataFrame, returns_col: str = 'log_return', bins: int = 60) -> go.Figure:
    """
    Return a Plotly histogram + KDE of returns.
    """
    try:
        if returns_col not in df.columns:
            # compute log returns from Price
            if 'Price' in df.columns:
                df = df.copy()
                df[returns_col] = np.log(df['Price'] / df['Price'].shift(1))
            else:
                raise ValueError('No Price or return column available')

        series = df[returns_col].dropna()
        fig = go.Figure()

        # Histogram
        fig.add_trace(go.Histogram(x=series, nbinsx=bins, name='Returns', histnorm='probability density', marker_color='#56B4E9', opacity=0.8))

        # KDE via scipy if available
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series)
            xs = np.linspace(series.min(), series.max(), 200)
            fig.add_trace(go.Scatter(x=xs, y=kde(xs), mode='lines', name='KDE', line=dict(color='#D55E00', width=2)))
        except Exception:
            # fallback: add a smoothed histogram trace using rolling mean
            hist, edges = np.histogram(series, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            fig.add_trace(go.Scatter(x=centers, y=pd.Series(hist).rolling(3, min_periods=1).mean(), mode='lines', name='Density (smoothed)', line=dict(color='#D55E00', width=2)))

        fig.update_layout(title='Return Distribution', xaxis_title='Return', yaxis_title='Density', template='plotly_white', height=450)
        return fig
    except Exception as e:
        logger.debug(f"plot_return_distribution_plotly failed: {e}")
        return go.Figure()


def plot_rolling_volatility_plotly(df: pd.DataFrame, price_col: str = 'Price', window: int = 21) -> go.Figure:
    """
    Compute and plot rolling annualized volatility (window in trading days).
    """
    try:
        if 'log_return' in df.columns:
            returns = df['log_return'].dropna()
        elif price_col in df.columns:
            returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        else:
            raise ValueError('No Price or return column available')

        roll = returns.rolling(window=window).std() * np.sqrt(252)
        dates = df.loc[roll.index, 'Date'] if 'Date' in df.columns else None

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=roll * 100, mode='lines', name=f'{window}-day Rolling Vol', line=dict(color='#0072B2', width=2)))

        # Overlay historical 21-day vol if exists
        if 'hist_vol_21d' in df.columns:
            fig.add_trace(go.Scatter(x=df['Date'], y=df['hist_vol_21d'] * 100, mode='lines', name='Hist Vol (21d)', line=dict(color='#DE8F05', width=1.5), opacity=0.6))

        fig.update_layout(title=f'{window}-Day Rolling Annualized Volatility (%)', xaxis_title='Date', yaxis_title='Volatility (%)', template='plotly_white', height=450)
        return fig
    except Exception as e:
        logger.debug(f"plot_rolling_volatility_plotly failed: {e}")
        return go.Figure()


def plot_prediction_correlation_heatmap(predictions: Dict[str, np.ndarray], actual: np.ndarray) -> go.Figure:
    """Create a correlation heatmap for predictions and actual values."""
    try:
        data = predictions.copy()
        if actual is not None and len(actual) > 0:
            data['Actual'] = np.array(actual[:len(next(iter(predictions.values())))] if predictions else actual)

        corr_df = pd.DataFrame(data).corr()
        if corr_df.empty:
            return go.Figure()

        fig = px.imshow(
            corr_df,
            text_auto='.2f',
            color_continuous_scale='RdBu',
            zmin=-1,
            zmax=1,
            title='Prediction Correlation Heatmap'
        )
        fig.update_layout(height=550, template='plotly_white')
        return fig
    except Exception as e:
        logger.debug(f"plot_prediction_correlation_heatmap failed: {e}")
        return go.Figure()


def plot_residual_error_plotly(actual: np.ndarray, predictions: Dict[str, np.ndarray], best_model: str, dates: Optional[np.ndarray] = None) -> go.Figure:
    """Create residual and error distribution plots for the best model."""
    try:
        if best_model not in predictions:
            return go.Figure()

        preds = np.array(predictions[best_model])
        actual_arr = np.array(actual)
        n = min(len(preds), len(actual_arr))
        if n == 0:
            return go.Figure()

        residuals = actual_arr[:n] - preds[:n]
        x = dates[:n] if dates is not None else np.arange(n)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
                            subplot_titles=('Residuals Over Time', 'Residual Distribution'))

        fig.add_trace(
            go.Scatter(x=x, y=residuals, mode='lines+markers', name='Residuals', line=dict(color='#E69F00')),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash='dash', line_color='black', row=1, col=1)

        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, name='Residual Distribution', marker_color='#56B4E9', opacity=0.8),
            row=2, col=1
        )

        fig.update_xaxes(title_text='Date' if dates is not None else 'Index', row=1, col=1)
        fig.update_yaxes(title_text='Residual', row=1, col=1)
        fig.update_xaxes(title_text='Residual', row=2, col=1)
        fig.update_yaxes(title_text='Count', row=2, col=1)

        fig.update_layout(title_text=f'Residual and Error Analysis ({best_model})', height=700, template='plotly_white')
        return fig
    except Exception as e:
        logger.debug(f"plot_residual_error_plotly failed: {e}")
        return go.Figure()


def plot_volatility_regime_detection(df: pd.DataFrame, price_col: str = 'Price', window: int = 21) -> go.Figure:
    """Detect volatility regimes using rolling volatility and show regime labels."""
    try:
        if 'log_return' in df.columns:
            returns = df['log_return'].dropna()
        elif price_col in df.columns:
            returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        else:
            raise ValueError('No Price or return column available')

        roll = returns.rolling(window=window).std() * np.sqrt(252)
        roll = roll.dropna()
        if roll.empty:
            return go.Figure()

        q1 = roll.quantile(0.33)
        q2 = roll.quantile(0.66)
        regimes = pd.cut(roll, bins=[-np.inf, q1, q2, np.inf], labels=['Low', 'Medium', 'High'])
        dates = df.loc[roll.index, 'Date'] if 'Date' in df.columns else roll.index

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=roll * 100, mode='lines', name=f'{window}-day Rolling Vol', line=dict(color='#0072B2', width=2)))
        fig.add_trace(go.Scatter(x=dates, y=(regimes == 'Low').astype(int) * roll.max() * 100, mode='markers', name='Low Regime', marker=dict(color='green', size=6)))
        fig.add_trace(go.Scatter(x=dates, y=(regimes == 'Medium').astype(int) * roll.max() * 100, mode='markers', name='Medium Regime', marker=dict(color='orange', size=6)))
        fig.add_trace(go.Scatter(x=dates, y=(regimes == 'High').astype(int) * roll.max() * 100, mode='markers', name='High Regime', marker=dict(color='red', size=6)))

        fig.add_hline(y=q1 * 100, line_dash='dash', line_color='gray', annotation_text='Low/Medium', annotation_position='bottom right')
        fig.add_hline(y=q2 * 100, line_dash='dash', line_color='gray', annotation_text='Medium/High', annotation_position='bottom right')

        fig.update_layout(title=f'Volatility Regime Detection ({window}-Day Rolling Vol)', xaxis_title='Date', yaxis_title='Volatility (%)', template='plotly_white', height=550)
        return fig
    except Exception as e:
        logger.debug(f"plot_volatility_regime_detection failed: {e}")
        return go.Figure()


# =============================================================================
# MODULE TESTING
# =============================================================================

if __name__ == "__main__":
    # Test visualization with sample data
    print("Testing visualization module...")

    # Create sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2023-01-01', periods=n, freq='B')
    actual = np.abs(np.random.randn(n)) * 0.2 + 0.15

    predictions = {
        'Random Forest': actual + np.random.randn(n) * 0.02,
        'XGBoost': actual + np.random.randn(n) * 0.018,
        'Ridge': actual + np.random.randn(n) * 0.025
    }

    results_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Ridge'],
        'RMSE': [0.020, 0.018, 0.025],
        'MAE': [0.015, 0.014, 0.019],
        'R2': [0.85, 0.88, 0.78],
        'Directional_Accuracy': [65.0, 68.0, 62.0]
    })

    feature_imp = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(30)],
        'Importance': np.random.rand(30)
    }).sort_values('Importance', ascending=False)

    # Generate figures
    figures = create_stock_report_figures(
        stock_name='TEST',
        dates=dates,
        actual=actual,
        predictions=predictions,
        results_df=results_df,
        feature_importance=feature_imp,
        save=True,
        output_dir='results/figures'
    )

    print(f"Generated {len(figures)} test figures")
    print("Visualization module test complete!")