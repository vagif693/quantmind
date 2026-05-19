"""Long/flat strategy backtester with commission modelling."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantmind.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Portfolio equity curve, performance metrics, and trade log."""

    portfolio: pd.DataFrame
    metrics: dict[str, float]
    trades: pd.DataFrame


def run(
    df: pd.DataFrame,
    predictions: pd.Series,
    probabilities: pd.Series,
    cfg: PipelineConfig,
) -> BacktestResult:
    """Simulate a long/flat strategy on the test window.

    Entry rule: go 100 % long when the model predicts *up* with probability
    above ``cfg.confidence_threshold``.  All other days are flat (cash).
    Positions are lagged by one day to avoid lookahead.
    """
    data = df.loc[predictions.index].copy()
    data["pred"] = predictions.values
    data["prob"] = probabilities.values
    data["daily_return"] = data["Close"].pct_change()

    # signal → position (1-day lag)
    signal = ((data["pred"] == 1) & (data["prob"] >= cfg.confidence_threshold)).astype(int)
    data["position"] = signal.shift(1).fillna(0).astype(int)

    # commissions on state changes
    rate = cfg.commission_bps / 10_000
    data["trade"] = data["position"].diff().abs().fillna(0)
    data["commission"] = data["trade"] * rate

    data["strategy_return"] = data["position"] * data["daily_return"] - data["commission"]
    data["strategy_equity"] = cfg.initial_capital * (1 + data["strategy_return"]).cumprod()
    data["buyhold_equity"] = cfg.initial_capital * (1 + data["daily_return"]).cumprod()

    metrics = _compute_metrics(data, cfg.initial_capital)
    trades = data.loc[data["trade"] > 0, ["Close", "position", "prob"]].copy()
    trades["action"] = trades["position"].map({1: "BUY", 0: "SELL"})

    logger.info(
        "Backtest — return=%+.2f%%  sharpe=%.2f  maxdd=%.1f%%  trades=%d",
        metrics["strategy_total_return"] * 100,
        metrics["sharpe"],
        metrics["max_drawdown"] * 100,
        metrics["total_trades"],
    )
    return BacktestResult(portfolio=data, metrics=metrics, trades=trades)


def _compute_metrics(df: pd.DataFrame, capital: float) -> dict[str, float]:
    eq = df["strategy_equity"]
    bh = df["buyhold_equity"]

    strat_ret = eq.iloc[-1] / capital - 1
    bh_ret = bh.iloc[-1] / capital - 1

    daily = df["strategy_return"].dropna()
    sharpe = (daily.mean() / (daily.std() + 1e-10)) * np.sqrt(252)

    neg = daily[daily < 0]
    sortino = (daily.mean() / (neg.std() + 1e-10)) * np.sqrt(252)

    running_max = eq.cummax()
    dd = ((eq - running_max) / running_max).min()

    annual = (1 + strat_ret) ** (252 / max(len(df), 1)) - 1
    calmar = annual / abs(dd) if dd != 0 else 0.0

    in_market = df[df["position"] == 1]
    wins = (in_market["strategy_return"] > 0).sum()
    n_market = len(in_market)

    return {
        "strategy_total_return": strat_ret,
        "buyhold_total_return": bh_ret,
        "alpha": strat_ret - bh_ret,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": dd,
        "win_rate": wins / n_market if n_market else 0.0,
        "total_trades": int(df["trade"].sum() / 2),
        "days_in_market": n_market,
        "pct_time_in_market": n_market / len(df),
        "final_equity": eq.iloc[-1],
    }
