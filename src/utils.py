import logging
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
)
import numpy as np
from IPython.core.getipython import get_ipython
from plotly.subplots import make_subplots

from catboost import CatBoostRegressor

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path


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
        path = Path.cwd().joinpath("settings.yaml")
    logging.debug(f"Loading settings from {path.name!s}")
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_data() -> dict[str, Any]:
    """Load data."""
    logging = get_logger(level="auto")
    root = Path.cwd()

    example_file = root.joinpath("data/input/example.csv")
    logging.info(f"Loading data from {example_file!s}")
    example_data = pd.read_csv(example_file)
    logging.debug("Data loaded successfully.")

    return {"example_data": example_data}


def predict_submission_data(model) -> pd.DataFrame:
    logging.info("Loading submission data...")
    root = Path.cwd()
    submission_file = root.joinpath("data/input/submission_data.csv")
    # submission_data = pd.read_parquet(PATH / "submission_data.csv")
    submission = pd.read_csv(submission_file)
    logging.info("Adding predictions to submission data.")
    if model is None:
        logging.warning("Model not provided. Using placeholder.")
        submission["prediction"] = 1
    # Fill in 'prediction' values of submission
    #submission["prediction"] = model.predict(submission[features])  # TODO: "features" needs to be defined, maybe just using all columns anyway, so features = submission.columns?
    return submission


def save_submission_file(submission: pd.DataFrame, attempt: int = 1) -> None:
    # TODO: find the next number for revision in the folder that does not yet exist
    root = Path.cwd()
    submission_file = root.joinpath(f"data/output/submission_attempt_{attempt}.csv")
    while submission_file.exists():
        submission_file = root.joinpath(f"data/output/submission_attempt_{attempt}.csv")
        attempt += 1
    logging.info(f"Saving submission file to {submission_file!s}")
    submission.to_csv(submission_file, sep=",", index=False)


if __name__ == "__main__":
    df_submission = predict_submission_data(None)
    save_submission_file(df_submission)