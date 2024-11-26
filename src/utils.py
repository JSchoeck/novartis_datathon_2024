import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from streamlit import session_state as ss


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Load settings from a YAML file.

    Args:
        path (Path): The path to the settings yaml file. Defaults to setting.yaml if None.

    Returns:
        Dict: A dictionary of settings.
    """
    if path is None:
        path = Path.cwd().joinpath("settings.yaml")
    logging.debug(f"Loading settings from {path.name!s}")
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_data() -> None:
    """Load data into streamlit session state."""
    root = Path.cwd()
    ss.data_input = pd.read_csv(root.joinpath("data/input/example.csv"))
