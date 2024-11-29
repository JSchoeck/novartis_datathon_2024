from typing import Protocol

import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import utils

S = utils.load_settings()


class Model(Protocol):
    def predict(self, data: pd.DataFrame, *args, **kwargs) -> pd.Series: ...


class Naive(Model):
    def predict(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series([S["naive"]["value"]] * len(data))


# class XGBoost(Model):
#     def preprocess(
#         self,
#         train_data: pd.DataFrame,
#         test_data: pd.DataFrame,
#         numerical_features: list[str],
#         categorical_features: list[str],
#     ) -> pd.DataFrame:
#         features = categorical_features + numerical_features
#         X_train = train_data[features].astype({col: "category" for col in categorical_features})
#         y_train = train_data["target"]
#         X_test = test_data[features].astype({col: "category" for col in categorical_features})

#         return data

#     def predict(
#         self,
#     ) -> pd.Series:
#         # TODO: tune hyperparameters to achieve better results
#         xgboost = XGBRegressor(tree_method="hist", max_depth=4, enable_categorical=True)
#         xgboost.fit(X_train, y_train)
#         return xgboost.predict(X_test)


class CatBoost(Model):
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        return data

    def predict(self, train_data: pd.DataFrame, test_data: pd.DataFrame, features: list[str]) -> pd.Series:
        X_train = train_data[features].astype({col: "category" for col in features})
        y_train = train_data["target"]
        X_test = test_data[features].astype({col: "category" for col in features})

        # TODO: tune hyperparameters to achieve better results
        catboost = CatBoostRegressor(
            cat_features=features,
            verbose=False,
        )
        catboost.fit(X_train, y_train)
        return catboost.predict(X_test)
