import numpy as np
import pandas as pd

from src.main_pipeline import VolatilityFeatures


def test_create_all_features_adds_news_columns_for_named_stock():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    price = 100 + np.cumsum(np.random.randn(len(dates)) * 0.2)

    df = pd.DataFrame(
        {
            "Date": dates,
            "Price": price,
            "Open": price,
            "High": price + 0.5,
            "Low": price - 0.5,
            "Volume": np.linspace(1000, 2000, len(dates)),
        }
    )

    features = VolatilityFeatures(df).create_all_features(stock_name="MTNN")

    assert "news_article_count" in features.columns
    assert "news_sentiment_score" in features.columns
    assert "news_policy_flag" in features.columns
    assert "news_volatility_adjustment" in features.columns
