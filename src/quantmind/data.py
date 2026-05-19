"""Market data acquisition via yfinance with synthetic fallback."""

from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch(ticker: str = "SPY", period_years: int = 3) -> pd.DataFrame:
    """Fetch daily OHLCV data for *ticker* over the last *period_years* years.

    Falls back to synthetic generation (geometric Brownian motion) when
    yfinance is unavailable or returns an empty frame.
    """
    end = pd.Timestamp.now()
    start = end - timedelta(days=period_years * 365)

    try:
        import yfinance as yf

        logger.info("Downloading %s [%s  %s]", ticker, start.date(), end.date())
        df = yf.download(ticker, start=str(start.date()), end=str(end.date()), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if not df.empty:
            logger.info("Loaded %d rows for %s", len(df), ticker)
            return df
        logger.warning("yfinance returned no data; using synthetic generator")
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance unavailable (%s); using synthetic generator", exc)

    return _generate_synthetic(ticker, period_years)


# ---------------------------------------------------------------------------
# Synthetic data (for environments without network access)
# ---------------------------------------------------------------------------

_SEED_PRICES = {"SPY": 430, "AAPL": 170, "QQQ": 360, "MSFT": 330, "TSLA": 240}


def _generate_synthetic(ticker: str, period_years: int) -> pd.DataFrame:
    """Geometric Brownian motion with stochastic volatility."""
    dates = pd.bdate_range(
        end=pd.Timestamp.now(),
        periods=period_years * 252,
    )
    n = len(dates)
    rng = np.random.default_rng(42)

    mu = 0.08 / 252
    base_sigma = 0.16 / np.sqrt(252)
    price = float(_SEED_PRICES.get(ticker, 200))

    closes = np.empty(n)
    closes[0] = price
    for i in range(1, n):
        sigma = base_sigma * (1.0 + 0.3 * np.sin(2 * np.pi * i / 252))
        closes[i] = closes[i - 1] * np.exp(mu + sigma * rng.standard_normal())

    spread = np.abs(rng.standard_normal(n)) * 0.008 + 0.003
    df = pd.DataFrame(
        {
            "Open": closes * (1 + rng.standard_normal(n) * 0.003),
            "High": closes * (1 + spread),
            "Low": closes * (1 - spread),
            "Close": closes,
            "Volume": rng.lognormal(mean=17.5, sigma=0.4, size=n).astype(int),
        },
        index=dates,
    )
    logger.info("Generated %d synthetic rows for %s", len(df), ticker)
    return df
