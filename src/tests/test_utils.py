import pytest  # noqa: F401

import utils


def test_load_settings() -> None:
    assert isinstance(utils.load_settings(), dict)
