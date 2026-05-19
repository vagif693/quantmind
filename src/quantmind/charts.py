"""Chart generation for pipeline outputs.

All plots use a dark colour scheme suitable for technical presentations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# -- palette ----------------------------------------------------------------

_C = {
    "bg": "#0D1117",
    "panel": "#161B22",
    "grid": "#21262D",
    "text": "#C9D1D9",
    "muted": "#7D8590",
    "green": "#3FB950",
    "red": "#F85149",
    "blue": "#58A6FF",
    "cyan": "#39D2C0",
    "yellow": "#D29922",
}


def _apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": _C["bg"],
        "axes.facecolor": _C["panel"],
        "axes.edgecolor": _C["grid"],
        "axes.labelcolor": _C["text"],
        "axes.grid": True,
        "grid.color": _C["grid"],
        "grid.alpha": 0.5,
        "text.color": _C["text"],
        "xtick.color": _C["muted"],
        "ytick.color": _C["muted"],
        "legend.facecolor": _C["panel"],
        "legend.edgecolor": _C["grid"],
        "font.family": "monospace",
        "font.size": 11,
    })


def _save(fig: plt.Figure, path: Path | str) -> None:
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor=_C["bg"])
    plt.close(fig)
    logger.info("Saved %s", path)


# -- individual charts ------------------------------------------------------

def equity_curve(portfolio: pd.DataFrame, metrics: dict, path: Path) -> None:
    """Strategy vs buy-and-hold equity with drawdown subplot."""
    _apply_theme()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    gridspec_kw={"hspace": 0.08})

    ax1.plot(portfolio.index, portfolio["strategy_equity"],
             color=_C["cyan"], lw=2, label="Strategy")
    ax1.plot(portfolio.index, portfolio["buyhold_equity"],
             color=_C["muted"], lw=1.5, ls="--", label="Buy & Hold", alpha=0.7)

    ax1.fill_between(portfolio.index, portfolio["strategy_equity"], portfolio["buyhold_equity"],
                      where=portfolio["strategy_equity"] >= portfolio["buyhold_equity"],
                      alpha=0.1, color=_C["green"])
    ax1.fill_between(portfolio.index, portfolio["strategy_equity"], portfolio["buyhold_equity"],
                      where=portfolio["strategy_equity"] < portfolio["buyhold_equity"],
                      alpha=0.1, color=_C["red"])

    ax1.set_title(
        f"Equity Curve  |  Sharpe {metrics['sharpe']:.2f}  |  "
        f"Return {metrics['strategy_total_return']:+.1%}  vs  B&H {metrics['buyhold_total_return']:+.1%}",
        fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.tick_params(labelbottom=False)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    peak = portfolio["strategy_equity"].cummax()
    dd = (portfolio["strategy_equity"] - peak) / peak
    ax2.fill_between(portfolio.index, dd, 0, color=_C["red"], alpha=0.4)
    ax2.plot(portfolio.index, dd, color=_C["red"], lw=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    _save(fig, path)


def signals(df: pd.DataFrame, predictions: pd.Series, path: Path) -> None:
    """Price chart with buy/sell markers."""
    _apply_theme()
    test = df.loc[predictions.index].copy()
    test["pred"] = predictions.values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[3, 1],
                                    gridspec_kw={"hspace": 0.08})

    ax1.plot(test.index, test["Close"], color=_C["text"], lw=1, alpha=0.9)
    buys = test[test["pred"].diff() == 1]
    sells = test[test["pred"].diff() == -1]
    ax1.scatter(buys.index, buys["Close"], marker="^", s=70, color=_C["green"],
                zorder=5, label="Buy", edgecolors="white", linewidths=0.5)
    ax1.scatter(sells.index, sells["Close"], marker="v", s=70, color=_C["red"],
                zorder=5, label="Sell", edgecolors="white", linewidths=0.5)

    ax1.set_title("Trading Signals", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", fontsize=10)
    ax1.tick_params(labelbottom=False)

    colors = [_C["green"] if r >= 0 else _C["red"]
              for r in test["Close"].pct_change().fillna(0)]
    ax2.bar(test.index, test["Volume"], color=colors, alpha=0.6, width=1)
    ax2.set_ylabel("Volume")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    _save(fig, path)


def feature_importance(importance: pd.Series, path: Path, top_n: int = 15) -> None:
    """Horizontal bar chart of top features."""
    _apply_theme()
    top = importance.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = [_C["cyan"] if "sentiment" in f else _C["blue"] for f in top.index]
    ax.barh(range(len(top)), top.values, color=colors, height=0.65)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=11)
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Feature Importance", fontsize=13, fontweight="bold", pad=12)

    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(fc=_C["cyan"], label="Sentiment"), Patch(fc=_C["blue"], label="Technical")],
        loc="lower right", fontsize=10,
    )
    _save(fig, path)


def confusion(cm: np.ndarray, path: Path) -> None:
    """Confusion matrix heatmap."""
    _apply_theme()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Down/Flat", "Up"], yticklabels=["Down/Flat", "Up"],
                ax=ax, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 18, "weight": "bold"},
                linewidths=2, linecolor=_C["bg"])
    ax.set_xlabel("Predicted", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual", fontsize=12, labelpad=10)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    _save(fig, path)


def sentiment_overlay(df: pd.DataFrame, path: Path) -> None:
    """Price with sentiment subplot."""
    _apply_theme()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), height_ratios=[2, 1],
                                    gridspec_kw={"hspace": 0.12})

    ax1.plot(df.index, df["Close"], color=_C["text"], lw=1.2)
    ax1.set_title("Sentiment vs Price", fontsize=13, fontweight="bold", pad=12)
    ax1.set_ylabel("Price ($)")
    ax1.tick_params(labelbottom=False)

    sent = df["sentiment_raw"].dropna()
    ax2.fill_between(sent.index, sent.clip(lower=0), 0, color=_C["green"], alpha=0.4)
    ax2.fill_between(sent.index, sent.clip(upper=0), 0, color=_C["red"], alpha=0.4)
    ax2.plot(sent.index, df.loc[sent.index, "sentiment_sma20"],
             color=_C["yellow"], lw=1.5, label="20-day MA")
    ax2.axhline(0, color=_C["muted"], lw=0.8, ls="--")
    ax2.set_ylabel("Sentiment")
    ax2.set_ylim(-1, 1)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    _save(fig, path)


# -- combined dashboard ------------------------------------------------------

def dashboard(
    portfolio: pd.DataFrame,
    metrics: dict,
    df: pd.DataFrame,
    predictions: pd.Series,
    importance_series: pd.Series,
    cm: np.ndarray,
    path: Path,
) -> None:
    """Six-panel summary dashboard."""
    _apply_theme()
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("quantmind  /  signal prediction pipeline",
                 fontsize=18, fontweight="bold", y=0.98, color=_C["cyan"],
                 fontfamily="monospace")

    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.97, top=0.93, bottom=0.05)

    # (0,0:2) equity
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(portfolio.index, portfolio["strategy_equity"], color=_C["cyan"], lw=2, label="Strategy")
    ax1.plot(portfolio.index, portfolio["buyhold_equity"], color=_C["muted"], lw=1.5, ls="--", label="B&H")
    ax1.set_title(f"Equity  |  {metrics['strategy_total_return']:+.1%}", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # (0,2) metrics card
    ax_m = fig.add_subplot(gs[0, 2])
    ax_m.axis("off")
    lines = [
        f"sharpe        {metrics['sharpe']:>8.2f}",
        f"sortino       {metrics['sortino']:>8.2f}",
        f"max drawdown  {metrics['max_drawdown']:>8.1%}",
        f"win rate      {metrics['win_rate']:>8.1%}",
        f"alpha         {metrics['alpha']:>+8.2%}",
        f"trades        {metrics['total_trades']:>8d}",
        f"time in mkt   {metrics['pct_time_in_market']:>8.1%}",
    ]
    ax_m.text(0.1, 0.92, "\n".join(lines), transform=ax_m.transAxes,
              fontsize=12, fontfamily="monospace", color=_C["text"],
              verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.5", fc=_C["panel"], ec=_C["grid"]))

    # (1,0:2) signals
    ax2 = fig.add_subplot(gs[1, :2])
    test_df = df.loc[predictions.index]
    ax2.plot(test_df.index, test_df["Close"], color=_C["text"], lw=1)
    buys = test_df.index[predictions.diff() == 1]
    sells = test_df.index[predictions.diff() == -1]
    ax2.scatter(buys, test_df.loc[buys, "Close"], marker="^", s=50, color=_C["green"], zorder=5)
    ax2.scatter(sells, test_df.loc[sells, "Close"], marker="v", s=50, color=_C["red"], zorder=5)
    ax2.set_title("Signal Overlay", fontsize=12, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    # (1,2) confusion
    ax3 = fig.add_subplot(gs[1, 2])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["D", "U"], yticklabels=["D", "U"],
                ax=ax3, cbar=False, annot_kws={"size": 14, "weight": "bold"},
                linewidths=1.5, linecolor=_C["bg"])
    ax3.set_title("Confusion", fontsize=12, fontweight="bold")

    # (2,0) importance
    ax4 = fig.add_subplot(gs[2, 0])
    top = importance_series.head(10).sort_values()
    colors = [_C["cyan"] if "sentiment" in f else _C["blue"] for f in top.index]
    ax4.barh(range(len(top)), top.values, color=colors, height=0.6)
    ax4.set_yticks(range(len(top)))
    ax4.set_yticklabels(top.index, fontsize=9)
    ax4.set_title("Top Features", fontsize=12, fontweight="bold")

    # (2,1:) sentiment
    ax5 = fig.add_subplot(gs[2, 1:])
    sent = df["sentiment_raw"].dropna()
    ax5.fill_between(sent.index, sent.clip(lower=0), 0, color=_C["green"], alpha=0.4)
    ax5.fill_between(sent.index, sent.clip(upper=0), 0, color=_C["red"], alpha=0.4)
    ax5.plot(sent.index, df.loc[sent.index, "sentiment_sma20"], color=_C["yellow"], lw=1.5)
    ax5.axhline(0, color=_C["muted"], lw=0.8, ls="--")
    ax5.set_title("Sentiment", fontsize=12, fontweight="bold")
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

    _save(fig, path)
