"""XGBoost classifier with walk-forward cross-validation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from quantmind.config import PipelineConfig

logger = logging.getLogger(__name__)

FEATURE_COLS: list[str] = [
    # trend
    "sma_20", "sma_50", "ema_12", "ema_26",
    "macd", "macd_signal", "macd_hist", "adx",
    # momentum
    "rsi_14", "stoch_k", "stoch_d", "williams_r", "roc_10",
    # volatility
    "bb_upper", "bb_lower", "bb_width", "bb_pct", "atr_14",
    # volume
    "obv", "mfi_14",
    # derived
    "returns_1d", "returns_5d", "volatility_20d",
    "price_vs_sma20", "volume_ratio",
    # sentiment
    "sentiment_raw", "sentiment_sma5", "sentiment_sma20", "sentiment_momentum",
]


@dataclass
class TrainResult:
    """Outputs of a single training run."""

    model: XGBClassifier
    feature_importance: pd.Series
    metrics: dict[str, float]
    predictions: pd.Series
    probabilities: pd.Series
    confusion: np.ndarray
    report: str
    X_test: pd.DataFrame = field(repr=False)
    y_test: pd.Series = field(repr=False)


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    available = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available)
    if missing:
        logger.warning("Missing features (skipped): %s", missing)

    subset = df[available + ["target"]].dropna()
    return subset[available], subset["target"]


def train(df: pd.DataFrame, cfg: PipelineConfig) -> TrainResult:
    """Train an XGBoost binary classifier and evaluate on a held-out test set.

    The split is purely temporal — the most recent *cfg.test_ratio* fraction of
    the data forms the test set, preventing any lookahead contamination.
    Walk-forward CV is run on the training portion for diagnostics.
    """
    X, y = _prepare(df)
    split = int(len(X) * (1 - cfg.test_ratio))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    logger.info("Train: %d | Test: %d (%.0f%%)", len(X_train), len(X_test), cfg.test_ratio * 100)

    # -- walk-forward CV on training set --
    tscv = TimeSeriesSplit(n_splits=cfg.cv_folds)
    cv_scores: list[float] = []
    for tr_idx, val_idx in tscv.split(X_train):
        fold_model = _make_model(cfg)
        fold_model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        cv_scores.append(accuracy_score(y_train.iloc[val_idx], fold_model.predict(X_train.iloc[val_idx])))
    logger.info("Walk-forward CV accuracy: %.3f +/- %.3f", np.mean(cv_scores), np.std(cv_scores))

    # -- final model --
    model = _make_model(cfg)
    model.fit(X_train, y_train)

    preds = pd.Series(model.predict(X_test), index=X_test.index)
    probs = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "cv_mean": float(np.mean(cv_scores)),
        "cv_std": float(np.std(cv_scores)),
    }
    logger.info(
        "Test — acc=%.3f  prec=%.3f  rec=%.3f  f1=%.3f",
        metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
    )

    importance = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    return TrainResult(
        model=model,
        feature_importance=importance,
        metrics=metrics,
        predictions=preds,
        probabilities=probs,
        confusion=confusion_matrix(y_test, preds),
        report=classification_report(y_test, preds, target_names=["Down/Flat", "Up"]),
        X_test=X_test,
        y_test=y_test,
    )


def _make_model(cfg: PipelineConfig) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
