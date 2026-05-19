"""Command-line interface for the quantmind pipeline."""

from __future__ import annotations

import argparse
import logging
import sys

from quantmind import __version__
from quantmind.config import PipelineConfig


def _parse_args(argv: list[str] | None = None) -> PipelineConfig:
    p = argparse.ArgumentParser(
        prog="quantmind",
        description="Trading signal prediction via technical indicators, LLM sentiment, and XGBoost.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--ticker", default="SPY", help="Ticker symbol (default: SPY)")
    p.add_argument("--years", type=int, default=3, help="Years of history (default: 3)")
    p.add_argument("--horizon", type=int, default=5, help="Prediction horizon in days (default: 5)")
    p.add_argument("--use-ai", action="store_true", help="Score headlines via Claude API")
    p.add_argument("--api-key", default=None, help="Anthropic API key")
    p.add_argument("--output-dir", default="output", help="Chart output directory")
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")

    args = p.parse_args(argv)
    return PipelineConfig(
        ticker=args.ticker,
        period_years=args.years,
        prediction_horizon=args.horizon,
        use_ai_sentiment=args.use_ai,
        anthropic_api_key=args.api_key,
        output_dir=args.output_dir,
        generate_charts=not args.no_charts,
    ), args.verbose


def main(argv: list[str] | None = None) -> None:
    cfg, verbose = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-5s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("quantmind")

    from quantmind import data, features, sentiment, model, backtest, charts

    log.info("quantmind v%s — %s, %d years, horizon=%d", __version__, cfg.ticker, cfg.period_years, cfg.prediction_horizon)

    # 1. data
    df = data.fetch(cfg.ticker, cfg.period_years)

    # 2. features
    df = features.add_technical_indicators(df)

    # 3. sentiment
    df = sentiment.add_sentiment_features(df, use_ai=cfg.use_ai_sentiment, api_key=cfg.anthropic_api_key)

    # 4. target
    df = features.create_target(df, horizon=cfg.prediction_horizon, threshold=cfg.target_threshold)

    # 5. model
    result = model.train(df, cfg)
    log.info("Classification report:\n%s", result.report)

    # 6. backtest
    bt = backtest.run(df, result.predictions, result.probabilities, cfg)

    # 7. charts
    if cfg.generate_charts:
        out = cfg.output_dir
        charts.equity_curve(bt.portfolio, bt.metrics, out / "equity_curve.png")
        charts.signals(df, result.predictions, out / "signals.png")
        charts.feature_importance(result.feature_importance, out / "feature_importance.png")
        charts.confusion(result.confusion, out / "confusion_matrix.png")
        charts.sentiment_overlay(df, out / "sentiment.png")
        charts.dashboard(
            bt.portfolio, bt.metrics, df,
            result.predictions, result.feature_importance, result.confusion,
            out / "dashboard.png",
        )
        log.info("Charts written to %s/", out)

    log.info("Done.")


if __name__ == "__main__":
    main()
