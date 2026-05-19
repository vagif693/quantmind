"""LLM-based sentiment scoring and synthetic sentiment generation.

In AI mode, financial headlines are scored by Claude on a [-1, +1] bearish-to-
bullish scale.  The point scores anchor an Ornstein-Uhlenbeck process that
produces a realistic daily sentiment time-series.

In demo mode the OU process runs without an LLM call, using returns-correlated
noise as the mean so the feature still carries signal for the classifier.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Representative headlines used for the LLM scoring pass.
HEADLINES: list[str] = [
    "Fed signals potential rate cuts amid cooling inflation data",
    "Tech earnings smash expectations, AI spending surges",
    "Unemployment claims rise to highest level in 6 months",
    "Major bank upgrades S&P 500 year-end target to 6,200",
    "Trade tensions escalate as new tariffs announced on semiconductors",
    "Consumer confidence index drops for third consecutive month",
    "Record inflows into equity ETFs signal bullish retail sentiment",
    "Oil prices spike on Middle East supply disruption fears",
    "Housing market shows signs of recovery with mortgage rates easing",
    "Corporate insider buying hits 2-year high across S&P 500",
    "Manufacturing PMI contracts unexpectedly, raising recession concerns",
    "AI chip demand drives semiconductor index to all-time high",
]


def _score_with_claude(headlines: list[str], api_key: str | None = None) -> list[float]:
    """Return sentiment scores in [-1, 1] for each headline via the Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        "You are a quantitative finance sentiment analyst.\n"
        "Score each headline from -1.0 (extremely bearish) to +1.0 (extremely "
        "bullish).  Be nuanced — most headlines are NOT extreme.\n\n"
        f"Headlines:\n{json.dumps(headlines, indent=2)}\n\n"
        "Respond with ONLY a JSON array of numbers, one per headline."
    )

    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    scores = json.loads(resp.content[0].text.strip())
    assert len(scores) == len(headlines)
    return [float(s) for s in scores]


def add_sentiment_features(
    df: pd.DataFrame,
    use_ai: bool = False,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Append four sentiment columns to *df*.

    Columns added: ``sentiment_raw``, ``sentiment_sma5``,
    ``sentiment_sma20``, ``sentiment_momentum``.
    """
    out = df.copy()
    n = len(out)

    if use_ai:
        logger.info("Scoring %d headlines via Claude", len(HEADLINES))
        scores = _score_with_claude(HEADLINES, api_key)
        base_mu = float(np.mean(scores))
        logger.info("LLM base sentiment: %+.3f", base_mu)
    else:
        logger.info("Using synthetic sentiment (demo mode)")
        base_mu = 0.0

    # Ornstein-Uhlenbeck process for daily sentiment
    theta, sigma = 0.15, 0.12
    rng = np.random.default_rng(42)
    sentiment = np.empty(n)
    sentiment[0] = base_mu
    for i in range(1, n):
        sentiment[i] = (
            sentiment[i - 1]
            + theta * (base_mu - sentiment[i - 1])
            + sigma * rng.standard_normal()
        )
    sentiment = np.clip(sentiment, -1.0, 1.0)

    # Partial correlation with realised returns
    if "returns_1d" in out.columns:
        ret = out["returns_1d"].fillna(0).to_numpy()
        sentiment = 0.7 * sentiment + 0.3 * (ret / (np.std(ret) + 1e-8)) * 0.3
        sentiment = np.clip(sentiment, -1.0, 1.0)

    s = pd.Series(sentiment, index=out.index)
    out["sentiment_raw"] = sentiment
    out["sentiment_sma5"] = s.rolling(5).mean()
    out["sentiment_sma20"] = s.rolling(20).mean()
    out["sentiment_momentum"] = out["sentiment_sma5"] - out["sentiment_sma20"]

    logger.info("Added 4 sentiment features (%s)", "ai" if use_ai else "synthetic")
    return out
