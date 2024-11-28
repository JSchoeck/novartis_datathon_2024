from typing import Protocol

import pandas as pd
import xgboost as xgb

class Model(Protocol):
    def predict(self, data: pd.DataFrame) -> pd.Series: ...


class Naive(Model):
    def predict(self, data: pd.DataFrame) -> pd.Series:
        return pd.Series([1.0] * len(data))

class XgBoost(Model):
    def predict(self, train_data: pd.DataFrame, test_data: pd.DataFrame, numerical_features, categorical_features) -> pd.Series:
        features = categorical_features + numerical_features
        X_train = train_data[features].astype({col: "category" for col in categorical_features})
        y_train = train_data["target"]
        X_test = test_data[features].astype({col: "category" for col in categorical_features})

        # TODO: tune hyperparameters to achieve better results
        xgboost = xgb.XGBRegressor(tree_method="hist", max_depth=4, enable_categorical=True)
        xgboost.fit(X_train, y_train)
        return xgboost.predict(X_test)


class CatBoost(Model):
    def predict(self, train_data: pd.DataFrame,test_data: pd.DataFrame, features) -> pd.Series:
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