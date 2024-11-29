# -*- coding: utf-8 -*-
"""Compute metrics for Novartis Datathon 2024.

This auxiliar file is intended to be used by participants in case
you want to test the metric with your own train/validation splits.
"""

# from typing import Tuple
# from pathlib import Path
# from models.models import XgBoost
import warnings
from typing import Callable

import pandas as pd
from sklearn.metrics import make_scorer

warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")


def _CYME(df: pd.DataFrame) -> float:  # noqa: N802
    """Compute the CYME metric, that is 1/2(median(yearly error) + median(monthly error))."""
    # TODO: check with mentors if observed=False is correct
    yearly_agg = df.groupby("cluster_nl", observed=False)[["target", "prediction"]].sum().reset_index()
    yearly_error = abs((yearly_agg["target"] - yearly_agg["prediction"]) / yearly_agg["target"]).median()

    monthly_error = abs((df["target"] - df["prediction"]) / df["target"]).median()

    return 1 / 2 * (yearly_error + monthly_error)


def _metric(df: pd.DataFrame) -> float:
    """Compute metric of submission.

    :param df: Dataframe with target and 'prediction', and identifiers.
    :return: Performance metric
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Split 0 actuals - rest
    zeros = df[df["zero_actuals"] == 1]
    recent = df[df["zero_actuals"] == 0]

    # weight for each group
    zeros_weight = len(zeros) / len(df)
    recent_weight = 1 - zeros_weight

    # Compute CYME for each group
    return round(recent_weight * _CYME(recent) + zeros_weight * min(1, _CYME(zeros)), 8)


def _metric_terms(df: pd.DataFrame) -> tuple[(float, float)]:
    """Compute single terms of metric.

    :param df: Dataframe with target and 'prediction', and identifiers.
    :return: Performance metric for recent and future launches
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Split 0 actuals - rest
    zeros = df[df["zero_actuals"] == 1]
    recent = df[df["zero_actuals"] == 0]

    return round(_CYME(recent), 8), round(min(1, _CYME(zeros)), 8)


# def compute_metric(submission: pd.DataFrame) -> Tuple[float, float]:
def compute_metric(submission: pd.DataFrame) -> float:
    """Compute metric.

    :param submission: Prediction. Requires columns: ['cluster_nl', 'date', 'target', 'prediction']
    :return: Performance metric.
    """
    submission["date"] = pd.to_datetime(submission["date"])
    submission = submission[["cluster_nl", "date", "target", "prediction", "zero_actuals"]]

    return _metric(submission)


def compute_metric_terms(submission: pd.DataFrame) -> tuple[float, float]:
    """Compute single terms of metric.

    :param submission: Prediction. Requires columns: ['cluster_nl', 'date', 'target', 'prediction']
    :return: Performance metric for recent and future launches.
    """
    submission["date"] = pd.to_datetime(submission["date"])
    submission = submission[["cluster_nl", "date", "target", "prediction", "zero_actuals"]]

    return _metric_terms(submission)


def cyme_scorer() -> Callable:
    """Return a scorer for the CYME metric."""
    return make_scorer(score_func=_metric, response_method="predict", greater_is_better=False)
