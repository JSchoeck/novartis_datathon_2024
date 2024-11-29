# -*- coding: utf-8 -*-
"""Compute metrics for Novartis Datathon 2024.

This auxiliar file is intended to be used by participants in case
you want to test the metric with your own train/validation splits.
"""

import pandas as pd

# from typing import Tuple
# from pathlib import Path
# from models.models import XgBoost


def _CYME(df: pd.DataFrame) -> float:  # noqa: N802
    """Compute the CYME metric, that is 1/2(median(yearly error) + median(monthly error))."""
    yearly_agg = df.groupby("cluster_nl")[["target", "prediction"]].sum().reset_index()
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


# def compute_metric(submission: pd.DataFrame) -> Tuple[float, float]:
def compute_metric(submission: pd.DataFrame) -> float:
    """Compute metric.

    :param submission: Prediction. Requires columns: ['cluster_nl', 'date', 'target', 'prediction']
    :return: Performance metric.
    """
    submission["date"] = pd.to_datetime(submission["date"])
    submission = submission[["cluster_nl", "date", "target", "prediction", "zero_actuals"]]

    return _metric(submission)


# if __name__ == "__main__":
#     # Load data
#     root = Path.cwd()

#     PATH = root.joinpath("data/input")
#     train_data_unfiltered = pd.read_csv(PATH / "train_data.csv", parse_dates=["launch_date", "date"])

#     # Split into train and validation set
#     split_date = "2021-01-10"
#     split_date_and_one = "2021-01-11"
#     train_data_unfiltered["launch_date"] = pd.to_datetime(train_data_unfiltered["launch_date"])

#     train_data_unfiltered["launch_day"] = train_data_unfiltered["launch_date"].dt.day
#     train_data_unfiltered["launch_month"] = train_data_unfiltered["launch_date"].dt.month
#     train_data_unfiltered["launch_year"] = train_data_unfiltered["launch_date"].dt.year

#     sorted_train_data = train_data_unfiltered.sort_values("launch_date")
#     train_data = sorted_train_data.iloc[:99999, :]
#     validation = sorted_train_data.iloc[100000:, :]

#     # Train your model
#     category_features = [
#         "brand",
#         "corporation",
#         "country",
#         "drug_id",
#         "indication",
#         "therapeutic_area",
#     ]  # categories that are not numerical
#     num_features = [
#         "launch_day",
#         "launch_month",
#         "launch_year",
#         "che_pc_usd",
#         "che_perc_gdp",
#         "insurance_perc_che",
#         "population",
#         "prev_perc",
#         "price_month",
#         "price_unit",
#         "public_perc_che",
#     ]
#     features = category_features + num_features
#     xgbooster = XgBoost()
#     # Perform predictions on validation set
#     validation["prediction"] = xgbooster.predict(train_data, validation, num_features, category_features)

#     # Assign column ["zero_actuals"] in the depending if in your
#     # split the cluster_nl has already had actuals on train or not

#     validation["zero_actuals"] = False  # unsure what this is, nan if true for prediction

#     # Optionally check performance
#     print("Performance:", compute_metric(validation))

#     # #Prepare submission
#     # submission_data = pd.read_parquet(PATH / "submission_data.csv")
#     # submission = pd.read_csv(PATH / "submission_template.csv")

#     # # Fill in 'prediction' values of submission
#     # submission["prediction"] = #model.predict(submission_data[features])

#     # # ...

#     # # Save submission
#     # SAVE_PATH = Path("path/to/save/folder")
#     # ATTEMPT = "attempt_x"
#     # submission.to_csv(SAVE_PATH / f"submission_{ATTEMPT}.csv", sep=",", index=False)
