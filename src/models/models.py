from typing import Protocol

import pandas as pd


class Model(Protocol):
    def predict(self, data: pd.DataFrame) -> pd.Series: ...


class Naive(Model):
    def predict(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series([1.0] * len(data))
