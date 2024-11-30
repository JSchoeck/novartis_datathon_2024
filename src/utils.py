import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pandas.core.groupby.generic import DataFrameGroupBy
from scipy import stats
from sklearn.model_selection import cross_validate

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 250)
pd.options.mode.copy_on_write = True


class ColorFormatter(logging.Formatter):
    """Logging colored formatter."""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt) -> None:  # noqa: D107, ANN001
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.fmt,  # self.grey + self.fmt + self.reset,
            logging.INFO: self.fmt,  # self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):  # noqa: ANN201, ANN001
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(
    name: str = "",
    level: Literal["auto", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "auto",
) -> logging.Logger:
    """Creates a common logger for all modules in a project.

    In a project, this should be called once in each module, with the module's __name__ as argument.
    i.e.:
        logger = get_logger(__name__)

        msg = "This is a log message."
        logger.info(msg)

    Args:
        name (str, optional): Name of the logger. Defaults to __name__ if not set.
        level (Literal["auto", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], optional): Log level. Defaults to "auto".

    Returns:
        logging.Logger: Logger object.
    """
    if name == "":
        name = __name__
    if level == "auto":
        debug_mode = os.getenv("DEBUGPY_RUNNING", "false").lower() in ["true", "1", "t"] or os.getenv(
            "DEBUG_MODE", "false"
        ).lower() in ["true", "1", "t"]
        level = "DEBUG" if debug_mode else "INFO"
    logger = logging.getLogger(name)
    match level:
        case "DEBUG":
            log_level = logging.DEBUG
        case "WARNING":
            log_level = logging.WARNING
        case "ERROR":
            log_level = logging.ERROR
        case "CRITICAL":
            log_level = logging.CRITICAL
        case _:
            log_level = logging.INFO
    logger.setLevel(log_level)
    if not any(handler.get_name() == name for handler in logger.handlers):
        log_format = "%(asctime)s - %(levelname)s - %(name)s:\t=== %(message)s"
        ch = logging.StreamHandler()
        ch.set_name(name)
        ch.setFormatter(ColorFormatter(log_format))
        logger.addHandler(ch)

    return logger


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load settings from a YAML file.

    Args:
        path (Path): The path to the settings yaml file. Defaults to setting.yaml if None.

    Returns:
        Dict: A dictionary of settings.
    """
    logging = get_logger(level="auto")
    if path is None:
        path = Path.cwd()
        if path.name == "src":
            path = path.parent
        path = path.joinpath("settings.yaml")
    logging.debug(f"Loading settings from {path.name!s}")
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_data(kind: Literal["train", "train_sample", "predict"]) -> pd.DataFrame:
    """Load data."""
    logging = get_logger(level="auto")
    path = Path.cwd()
    if path.name == "src":
        path = path.parent
    path = path.joinpath("data/input")
    match kind:
        case "train" | "train_sample":
            file = path.joinpath("train_data.csv")
        case "predict":
            file = path.joinpath("submission_data.csv")
    logging.info(f"Loading {kind} data from {file!s}")
    data = pd.read_csv(file)
    if kind == "train_sample":
        data = data.tail(1000)
    logging.info(f"Data loaded successfully: {data.shape[0]:,} rows and {data.shape[1]:,} columns.")
    logging.debug(f"Columns: {data.columns.to_list()}")
    return data


def predict_submission_data(model: Any, features: list[str] | None = None) -> pd.DataFrame:  # noqa: ANN401
    logging = get_logger(level="auto")
    submission = load_data("predict")
    logging.info("Adding predictions to submission data.")
    if features is not None:
        submission["prediction"] = model.predict(submission[features])
    else:
        submission["prediction"] = model.predict(submission.drop(columns=["target"]))
    return submission


def save_submission_file(
    submission: pd.DataFrame, attempt: int = 1, root: Path | None = None, model: str = "", user: str = ""
) -> None:
    logging = get_logger(level="auto")
    if model == "":
        model = Path(sys.argv[0]).stem
    if root is None:
        root = Path.cwd()
    user = "_" + user if user else ""
    submission_file = root.joinpath(f"data/output/submission_{model}_{attempt}{user}.csv")
    while submission_file.exists():
        submission_file = root.joinpath(f"data/output/submission_{model}_{attempt}{user}.csv")
        attempt += 1
    logging.info(f"Saving submission file to {submission_file!s}")
    submission.to_csv(submission_file, sep=",", index=False)


def calculate_month_difference(date1: pd.Series, date2: pd.Series) -> pd.Series:
    """Calculate the number of months between two dates.

    Args:
        date1 (str): The first date.
        date2 (str): The second date.

    Returns:
        int: The number of months between the two dates.
    """
    date1 = pd.to_datetime(date1)
    date2 = pd.to_datetime(date2)
    year_diff = date1.dt.year - date2.dt.year
    month_diff = date1.dt.month - date2.dt.month
    return year_diff * 12 + month_diff


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["sale_month"] = calculate_month_difference(df["date"], df["launch_date"])
    # Features not useful for monthly data
    # df["day"] = df["date"].dt.day
    # df["dayofweek"] = df["date"].dt.dayofweek
    # df["weekofyear"] = df["date"].dt.isocalendar().week
    return df


def identify_future_launches(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Identify future launches.

    Future launches are all cluster_nl that occurr in the test set, but not in the training set. Since the test set is chronologically after the training set, we can identify future launches by checking if the cluster_nl is in the test set, but not in the training set.
    """
    logging = get_logger(level="auto")
    future_launches = set(set(df_test["cluster_nl"].unique()) - set(df_train["cluster_nl"].unique()))
    recent_launches = set(set(df_train["cluster_nl"].unique()) - set(df_test["cluster_nl"].unique()))
    logging.info(f"Identified {len(future_launches):,} future and {len(recent_launches):,} recent launches.")
    df_train["zero_actuals"] = False
    # set zero_actuals to True in df_test for all future launches
    df_test["zero_actuals"] = False
    df_test.loc[df_test["cluster_nl"].isin(future_launches), "zero_actuals"] = True
    logging.info(f"Future launches have {df_test['zero_actuals'].sum():,} rows in the test set.")
    logging.info(
        f"Recent launches have {df_test['zero_actuals'].count()-df_test['zero_actuals'].sum():,} rows in the test set."
    )
    return df_train, df_test


def turn_dates_to_int(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    for date_name in date_columns:
        df[date_name + "_int"] = df[date_name].astype("int64")

    return df


def train_test_validation_split(
    df_features: pd.DataFrame, df_train: pd.DataFrame, validation_year: int, test_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df_features["target"] = df_train["target"]
    X_train = df_features[df_features["year"] < validation_year].copy()
    X_validate = df_features[df_features["year"] == validation_year].copy()
    X_test = df_features[df_features["year"] == test_year].copy()
    y_train = X_train.pop("target")
    y_validate = X_validate.pop("target")
    y_test = X_test.pop("target")
    logging.info(
        f"{X_train.shape=}, {X_validate.shape=}, {X_test.shape=}, {y_train.shape=}, {y_validate.shape=}, {y_test.shape=}"
    )
    return X_train, X_validate, X_test, y_train, y_validate, y_test


def remove_outlier_data(df: pd.DataFrame, column_name: str, threshold_z: int) -> pd.DataFrame:
    """Remove outlier data from a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to remove outliers from.
        column_name (str): The column to remove outliers from.
        threshold_z (int): The threshold for the z-score. i.e. 3 is 3 standard deviations from the mean.

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    df = df[np.abs(stats.zscore(df[column_name])) <= threshold_z]
    return df


def replace_minus_one_with_mean(
    df: pd.DataFrame,
    include_columns: list[str] | None = None,
    exclude_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Replace -1 values with the mean of the column.

    Args:
        df (pd.DataFrame): The DataFrame to replace values in.
        include_columns (list[str] | None): The columns to include. Defaults to None.
        exclude_columns (list[str] | None): The columns to exclude. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with values replaced.
    """
    columns = []
    if include_columns is None and exclude_columns is None:
        columns = df.columns
    elif include_columns is not None and exclude_columns is not None:
        columns = list(set(include_columns) - set(exclude_columns))
    elif include_columns is not None:
        columns = include_columns
    elif exclude_columns is not None:
        columns = df.columns.difference(exclude_columns)
    else:
        return df
    df[columns] = df[columns].replace(-1, np.nan)
    df[columns] = df[columns].fillna(df[columns].mean())
    return df

def add_sum_first_year(df: pd.DataFrame,) -> pd.DataFrame:
    df_copy = df.copy()

    df_copy["min_year_and_one"] = df.groupby(["cluster_nl","drug_id"])["launch_date"].transform(lambda x: pd.to_datetime(x).min() + pd.DateOffset(years=1))

    df_returning = pd.DataFrame()

    for grouping,group_df in  df_copy.groupby(["cluster_nl","drug_id"]):
        under_min = group_df[group_df["date"] <= group_df["min_year_and_one"]]
        group_df["sum_of_first_year_targets"] = under_min["target"].sum()
        df_returning = pd.concat([df_returning,group_df])

    print(df_returning.describe())
    return df_returning

def add_ltm_kpis(
    df: pd.DataFrame,
    fill_strategy: Literal["overall_mean", "cluster_mean", None] = "cluster_mean",
    fill_postprocessing: Literal["ffill", "bfill", None] = "bfill",
    columns: tuple[str] = ("target",),
) -> pd.DataFrame:
    grouped = df.groupby(["cluster_nl"])
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df = calculate_ltm_kpi(df, grouped, col, fill_strategy, fill_postprocessing)
        else:
            msg = f"Column {col} is not numeric, cannot calculate LTM KPI for it."
            raise ValueError(msg)
    return df


def calculate_ltm_kpi(
    df: pd.DataFrame,
    grouped: DataFrameGroupBy,
    col: str = "target",
    fill_strategy: Literal["overall_mean", "cluster_mean", None] = "cluster_mean",
    fill_postprocessing: Literal["ffill", "bfill", None] = "bfill",
) -> pd.DataFrame:
    MONTHS_IN_YEAR = 12
    logging.info(f"Calculating LTM_{col} for each cluster_nl.")
    r_sum = (
        grouped[[col, "date"]]
        .rolling(window=MONTHS_IN_YEAR, on="date", min_periods=MONTHS_IN_YEAR)
        .sum()
        .reset_index(1, drop=True)
    )
    ltm_col = "ltm_" + col
    for cluster in r_sum.index.unique():
        cluster_not_yet_launched_for_12_months = df.loc[df["cluster_nl"] == cluster]["sale_month"] < MONTHS_IN_YEAR
        cluster_months_launched = df.loc[df["cluster_nl"] == cluster]["sale_month"]

        match fill_strategy:
            case "overall_mean":
                fill_func = ltm_fill_with_overall_mean
            case "cluster_mean":
                fill_func = ltm_fill_with_cluster_mean
            case _:
                fill_func = ltm_fill_with_nan

        df.loc[df["cluster_nl"] == cluster, ltm_col] = np.where(
            # cond: if max months_open for store is less than 12
            cluster_not_yet_launched_for_12_months,
            # true: scale values <12 months_open up, based on mean of existing values; ltm starts at 1*mean up to 12*mean
            fill_func(df=df, col=col, cluster=cluster, cluster_months_launched=cluster_months_launched),
            # false: use rolling sum for last twelve months
            ltm_rolling_sum(col, r_sum, cluster),
        ).round(1)
        # Fill missing values for the first 12 entries, if a store was already open for more than 12 months at the beginning of the data
        if fill_postprocessing == "ffill":
            df.loc[df["cluster_nl"] == cluster, ltm_col] = df.loc[df["cluster_nl"] == cluster, ltm_col].ffill(  # type: ignore
                limit=MONTHS_IN_YEAR
            )
        elif fill_postprocessing == "bfill":
            df.loc[df["cluster_nl"] == cluster, ltm_col] = df.loc[df["cluster_nl"] == cluster, ltm_col].bfill(  # type: ignore
                limit=MONTHS_IN_YEAR
            )
        else:
            pass
    df.loc[:, ltm_col] = df.loc[:, ltm_col].round(1)
    return df


def ltm_fill_with_nan(cluster_months_launched: pd.Series, **kwargs) -> pd.Series:
    return pd.Series(np.nan, index=cluster_months_launched.index)


def ltm_fill_with_cluster_mean(
    df: pd.DataFrame, col: str, cluster: str, cluster_months_launched: pd.Series
) -> pd.Series:
    mean_cluster_value = df.loc[df["cluster_nl"] == cluster, col].mean()
    return mean_cluster_value * cluster_months_launched / 12


def ltm_fill_with_overall_mean(
    df: pd.DataFrame, col: str, cluster: str, cluster_months_launched: pd.Series
) -> pd.Series:
    mean_overall_value = df.loc[(df["country"] == df[df["cluster_nl"] == cluster]["country"].iloc[0]), col].mean()  # type: ignore
    return mean_overall_value * cluster_months_launched / 12


def ltm_rolling_sum(col: str, r_sum: pd.DataFrame, cluster: str) -> pd.Series:
    return r_sum.loc[r_sum.index.get_level_values("cluster_nl") == cluster][col]


class MetricEvaluation:
    '''Count of wrong predictions'''
    
    def is_max_optimal(self):
        False

    def evaluate(self, approxes, target, weight):  
        print(approxes)
        print(target)
        print(weight)
        y_pred = np.array(approxes).argmax(0)
        y_true = np.array(target)
                                    
        return sum(y_pred!=y_true), 1

    def get_final_error(self, error, weight):
        return error

if __name__ == "__main__":
    from models.models import Naive

    df_submission = predict_submission_data(Naive())
    # save_submission_file(df_submission)
