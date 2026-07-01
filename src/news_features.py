"""
Lightweight news and policy feature enrichment for volatility forecasting.

This module tries to fetch recent headlines for the uploaded stock and derives
simple features such as sentiment score, news count, policy mentions, and a
news-driven volatility adjustment factor.

The implementation is intentionally lightweight so it works without extra
packages and degrades gracefully when network access is unavailable.
"""

import logging
import re
from typing import Dict, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

POSITIVE_WORDS = {
    "gain", "gains", "rise", "rises", "up", "boost", "boosts", "surge", "surges",
    "strong", "stronger", "positive", "improve", "improves", "improved", "growth",
    "profit", "profits", "expansion", "increase", "increases", "rebound", "stable"
}
NEGATIVE_WORDS = {
    "fall", "falls", "drop", "drops", "decline", "declines", "down", "slump", "slumps",
    "weak", "weaker", "negative", "loss", "losses", "decrease", "decreases", "risk",
    "crisis", "default", "fraud", "scandal", "bearish", "pressure", "stumble", "cuts"
}
POLICY_WORDS = {
    "cbn", "policy", "inflation", "interest rate", "interest-rate", "fx", "exchange rate",
    "ban", "regulation", "regulatory", "fiscal", "monetary", "central bank", "rate hike"
}

STOCK_QUERY_MAP = {
    "ACCESS": "Access Bank",
    "AIRTEL": "Airtel Africa",
    "CWG": "CWG Nigeria",
    "DANGCEM": "Dangote Cement",
    "DANGSUG": "Dangote Sugar",
    "ETI": "Ecobank Transnational",
    "FIRSTHOLDCO": "First HoldCo",
    "GTCO": "Guaranty Trust Holding",
    "INTERBREW": "International Breweries",
    "MTNN": "MTN Nigeria",
    "NB": "Nigerian Breweries",
    "NESTLE": "Nestle Nigeria",
    "SEPLAT": "Seplat Energy",
    "WAPCO": "Wapco",
    "ZENITH": "Zenith Bank",
}


def _normalize_query(stock_name: Optional[str]) -> str:
    if not stock_name:
        return "Nigerian stock market"

    text = str(stock_name).strip().upper()
    if text in STOCK_QUERY_MAP:
        return STOCK_QUERY_MAP[text]

    cleaned = re.sub(r"[^A-Za-z0-9 ]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        return cleaned
    return "Nigerian stock market"


def _fetch_google_rss(query: str, max_articles: int = 8) -> List[Dict[str, str]]:
    try:
        encoded = quote(query)
        rss_url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        req = Request(rss_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(req, timeout=10) as response:
            xml_text = response.read().decode("utf-8", errors="ignore")

        root = ET.fromstring(xml_text)
        items: List[Dict[str, str]] = []
        for item in root.findall("./channel/item")[:max_articles]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            description = (item.findtext("description") or "").strip()
            if title:
                items.append({
                    "title": title,
                    "link": link,
                    "pub_date": pub_date,
                    "description": description,
                })
        return items
    except Exception as exc:
        logger.debug(f"News fetch failed for query '{query}': {exc}")
        return []


def _dedupe_articles(articles: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique: List[Dict[str, str]] = []
    for article in articles:
        key = re.sub(r"\s+", " ", article.get("title", "").strip().lower())
        if key and key not in seen:
            seen.add(key)
            unique.append(article)
    return unique


def _score_headline(text: str) -> Dict[str, float]:
    lowered = text.lower()
    positive_hits = sum(1 for word in POSITIVE_WORDS if word in lowered)
    negative_hits = sum(1 for word in NEGATIVE_WORDS if word in lowered)
    policy_hits = sum(1 for phrase in POLICY_WORDS if phrase in lowered)

    sentiment_score = 0.0
    if positive_hits + negative_hits > 0:
        sentiment_score = (positive_hits - negative_hits) / (positive_hits + negative_hits)

    return {
        "sentiment_score": sentiment_score,
        "positive_hits": float(positive_hits),
        "negative_hits": float(negative_hits),
        "policy_hits": float(policy_hits),
    }


def summarize_recent_news(stock_name: Optional[str], max_articles: int = 8) -> Dict[str, float]:
    """Fetch recent headlines and derive simple sentiment/policy features."""
    base_query = _normalize_query(stock_name)
    queries = [base_query, f"{base_query} Nigeria", f"{base_query} stock news", f"{base_query} CBN policy"]

    articles: List[Dict[str, str]] = []
    for query in queries:
        articles.extend(_fetch_google_rss(query, max_articles=max_articles))

    articles = _dedupe_articles(articles)[:max_articles]

    if not articles:
        return {
            "news_article_count": 0.0,
            "news_sentiment_score": 0.0,
            "news_negative_ratio": 0.0,
            "news_policy_mentions": 0.0,
            "news_policy_flag": 0.0,
            "news_volatility_adjustment": 0.0,
        }

    total = len(articles)
    sentiment_total = 0.0
    negative_count = 0
    policy_mentions = 0

    for article in articles:
        text = f"{article.get('title', '')} {article.get('description', '')}"
        score = _score_headline(text)
        sentiment_total += score["sentiment_score"]
        if score["negative_hits"] > 0:
            negative_count += 1
        policy_mentions += int(score["policy_hits"] > 0)

    avg_sentiment = sentiment_total / total if total else 0.0
    negative_ratio = negative_count / total if total else 0.0
    policy_mentions_total = float(policy_mentions)

    adjustment = (negative_ratio * 0.07) + (policy_mentions_total * 0.02) + (avg_sentiment * -0.03)
    adjustment = float(np.clip(adjustment, -0.15, 0.15))

    return {
        "news_article_count": float(total),
        "news_sentiment_score": float(avg_sentiment),
        "news_negative_ratio": float(negative_ratio),
        "news_policy_mentions": float(policy_mentions_total),
        "news_policy_flag": 1.0 if policy_mentions_total > 0 else 0.0,
        "news_volatility_adjustment": adjustment,
    }


def add_news_features_to_dataframe(df: pd.DataFrame, stock_name: Optional[str] = None) -> pd.DataFrame:
    """Attach recent news and policy features to a stock dataframe."""
    if df is None or df.empty:
        return df

    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    summary = summarize_recent_news(stock_name)

    for key, value in summary.items():
        df[key] = float(value)

    # Create a simple boolean proxy for the latest news signal.
    df["news_signal_strength"] = np.where(df["news_volatility_adjustment"] >= 0, 1, -1)

    return df
