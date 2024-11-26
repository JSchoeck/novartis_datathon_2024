import pandas as pd
import pytest  # noqa: F401
from streamlit import session_state as ss
from streamlit.testing.v1 import AppTest

import app


def test_init() -> None:
    app.init()
    assert isinstance(ss.settings, dict)


# streamlit UI tests
# https://docs.streamlit.io/develop/concepts/app-testing/get-started


def test_main_df() -> None:
    at = AppTest.from_file("../app.py").run()
    pd.testing.assert_frame_equal(at.main[2].value, pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))  # type: ignore
