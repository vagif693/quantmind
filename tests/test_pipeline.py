"""Unit tests for quantmind pipeline components."""

import numpy as np
import pandas as pd
import pytest

from quantmind.data import _generate_synthetic
from quantmind.features import add_technical_indicators, create_target
from quantmind.sentiment import add_sentiment_features


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    """Three years of synthetic OHLCV data."""
    return _generate_synthetic("SPY", period_years=3)


class TestData:
    def test_synthetic_shape(self, ohlcv: pd.DataFrame) -> None:
        assert len(ohlcv) > 500
        assert list(ohlcv.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_no_nans(self, ohlcv: pd.DataFrame) -> None:
        assert not ohlcv.isna().any().any()

    def test_high_gt_low(self, ohlcv: pd.DataFrame) -> None:
        assert (ohlcv["High"] >= ohlcv["Low"]).all()


class TestFeatures:
    def test_indicator_count(self, ohlcv: pd.DataFrame) -> None:
        out = add_technical_indicators(ohlcv)
        new_cols = set(out.columns) - set(ohlcv.columns)
        assert len(new_cols) >= 20

    def test_target_binary(self, ohlcv: pd.DataFrame) -> None:
        out = add_technical_indicators(ohlcv)
        out = create_target(out, horizon=5)
        valid = out["target"].dropna()
        assert set(valid.unique()).issubset({0, 1})

    def test_target_horizon_effect(self, ohlcv: pd.DataFrame) -> None:
        out = add_technical_indicators(ohlcv)
        short = create_target(out, horizon=1)
        long = create_target(out, horizon=20)
        # longer horizon produces more NaN future_returns at the tail
        assert short["future_return"].notna().sum() > long["future_return"].notna().sum()


class TestSentiment:
    def test_columns_added(self, ohlcv: pd.DataFrame) -> None:
        out = add_technical_indicators(ohlcv)
        out = add_sentiment_features(out, use_ai=False)
        for col in ("sentiment_raw", "sentiment_sma5", "sentiment_sma20", "sentiment_momentum"):
            assert col in out.columns

    def test_sentiment_range(self, ohlcv: pd.DataFrame) -> None:
        out = add_technical_indicators(ohlcv)
        out = add_sentiment_features(out, use_ai=False)
        raw = out["sentiment_raw"].dropna()
        assert raw.min() >= -1.0
        assert raw.max() <= 1.0
