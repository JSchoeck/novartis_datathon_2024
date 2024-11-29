import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import cross_validate

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)
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

    def format(self, record):  # noqa: ANN201, D102, ANN001
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
    return data


def predict_submission_data(model: Any, features: list[str] | None = None) -> pd.DataFrame:
    logging = get_logger(level="auto")
    submission = load_data("predict")
    logging.info("Adding predictions to submission data.")
    if features is not None:
        submission["prediction"] = model.predict(submission[features])
    else:
        submission["prediction"] = model.predict(submission.drop(columns=["target"]))
    return submission


def save_submission_file(submission: pd.DataFrame, attempt: int = 1) -> None:
    logging = get_logger(level="auto")
    root = Path.cwd()
    submission_file = root.joinpath(f"data/output/submission_attempt_{attempt}.csv")
    while submission_file.exists():
        submission_file = root.joinpath(f"data/output/submission_attempt_{attempt}.csv")
        attempt += 1
    logging.info(f"Saving submission file to {submission_file!s}")
    submission.to_csv(submission_file, sep=",", index=False)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    # df["day"] = df["date"].dt.day
    # df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week
    return df


# def evaluate(model, X, y, cv, model_prop=None, model_step=None) -> None:
#     cv_results = cross_validate(
#         model,
#         X,
#         y,
#         cv=cv,
#         scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error"],
#         return_estimator=model_prop is not None,
#     )
#     if model_prop is not None:
#         if model_step is not None:
#             values = [getattr(m[model_step], model_prop) for m in cv_results["estimator"]]
#         else:
#             values = [getattr(m, model_prop) for m in cv_results["estimator"]]
#         print(f"Mean model.{model_prop} = {np.mean(values)}")
#     mae = -cv_results["test_neg_mean_absolute_error"]
#     rmse = -cv_results["test_neg_root_mean_squared_error"]
#     print(
#         f"Mean Absolute Error:     {mae.mean():.3f} +/- {mae.std():.3f}\n"
#         f"Root Mean Squared Error: {rmse.mean():.3f} +/- {rmse.std():.3f}"
#     )


if __name__ == "__main__":
    from models.models import Naive

    df_submission = predict_submission_data(Naive())
    save_submission_file(df_submission)
