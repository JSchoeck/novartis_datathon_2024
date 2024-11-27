import logging
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from streamlit import session_state as ss


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


def load_data_to_streamlit() -> None:
    """Load data to Streamlit."""
    logging = get_logger(level="auto")
    ss.data_input = load_data()
    logging.debug("Data loaded to Streamlit.")
