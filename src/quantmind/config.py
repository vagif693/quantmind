"""Pipeline configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for a single pipeline run."""

    ticker: str = "SPY"
    period_years: int = 3
    prediction_horizon: int = 5
    target_threshold: float = 0.0

    # model
    test_ratio: float = 0.20
    cv_folds: int = 5
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05

    # backtest
    initial_capital: float = 100_000.0
    confidence_threshold: float = 0.55
    commission_bps: float = 5.0

    # sentiment
    use_ai_sentiment: bool = False
    anthropic_api_key: str | None = None

    # output
    output_dir: Path = field(default_factory=lambda: Path("output"))
    generate_charts: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
