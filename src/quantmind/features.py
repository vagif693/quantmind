"""Technical indicator computation and target construction."""

from __future__ import annotations

import logging

import pandas as pd
import ta

logger = logging.getLogger(__name__)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append ~25 technical indicator columns to an OHLCV DataFrame.

    Categories: trend, momentum, volatility, volume, and derived ratios.
    """
    out = df.copy()
    c, h, l, v = out["Close"], out["High"], out["Low"], out["Volume"]

    # -- trend --
    out["sma_20"] = ta.trend.sma_indicator(c, window=20)
    out["sma_50"] = ta.trend.sma_indicator(c, window=50)
    out["ema_12"] = ta.trend.ema_indicator(c, window=12)
    out["ema_26"] = ta.trend.ema_indicator(c, window=26)

    macd = ta.trend.MACD(c)
    out["macd"] = macd.macd()
    out["macd_signal"] = macd.macd_signal()
    out["macd_hist"] = macd.macd_diff()
    out["adx"] = ta.trend.ADXIndicator(h, l, c).adx()

    # -- momentum --
    out["rsi_14"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    stoch = ta.momentum.StochasticOscillator(h, l, c)
    out["stoch_k"] = stoch.stoch()
    out["stoch_d"] = stoch.stoch_signal()
    out["williams_r"] = ta.momentum.WilliamsRIndicator(h, l, c).williams_r()
    out["roc_10"] = ta.momentum.ROCIndicator(c, window=10).roc()

    # -- volatility --
    bb = ta.volatility.BollingerBands(c)
    out["bb_upper"] = bb.bollinger_hband()
    out["bb_lower"] = bb.bollinger_lband()
    out["bb_width"] = bb.bollinger_wband()
    out["bb_pct"] = bb.bollinger_pband()
    out["atr_14"] = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()

    # -- volume --
    out["obv"] = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    out["mfi_14"] = ta.volume.MFIIndicator(h, l, c, v, window=14).money_flow_index()

    # -- derived --
    out["returns_1d"] = c.pct_change(1)
    out["returns_5d"] = c.pct_change(5)
    out["volatility_20d"] = out["returns_1d"].rolling(20).std()
    out["price_vs_sma20"] = (c - out["sma_20"]) / out["sma_20"]
    out["volume_ratio"] = v / v.rolling(20).mean()

    n_new = len(out.columns) - len(df.columns)
    logger.info("Added %d technical features", n_new)
    return out


def create_target(
    df: pd.DataFrame,
    horizon: int = 5,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """Binary target: 1 if forward return over *horizon* days exceeds *threshold*."""
    out = df.copy()
    out["future_return"] = out["Close"].shift(-horizon) / out["Close"] - 1
    out["target"] = (out["future_return"] > threshold).astype(int)

    balance = out["target"].value_counts(normalize=True)
    logger.info(
        "Target (horizon=%d, threshold=%.2f%%): %.1f%% positive",
        horizon,
        threshold * 100,
        balance.get(1, 0) * 100,
    )
    return out
