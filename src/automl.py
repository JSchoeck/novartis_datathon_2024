# %%
import multiprocessing
from pathlib import Path

import numpy as np
from IPython.display import display
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import all_estimators, shuffle
from xgboost import XGBRegressor

import helper
import utils

print("---")
logging = utils.get_logger(__name__)
P = utils.load_settings()["params"]

# %%
################ Parameters ################
validation_year = 2021
test_year = 2022
add_ltm_columns = False
model_params = {
    "verbose": 0,
    "ignore_warnings": False,
    "custom_metric": None,
    # "enable_categorical": True,  # good?
    # "tree_method": "hist",  # good?
    # "max_depth": 6,  # 4-7 is good, more is overfitting with default number of features
    # "n_estimators": 200,  # 500 for dev, 200 for submit??
    # "max_cat_threshold": 1000,  # good?
    # "n_jobs": multiprocessing.cpu_count() - 1,
}
drop_features = [
    # "price_month",
    # "insurance_perc_che",
    "therapeutic_area",
    "country",
    "corporation",
    "drug_id",
]

################# Settings #################
submit = False  # Set to True to generate submission file # TODO
categorical_features = [
    "brand",
    "cluster_nl",
    "corporation",
    "country",
    "drug_id",
    "indication",
    "therapeutic_area",
]

################# Model-specific additions #################
regressors = [
    "HistGradientBoostingRegressor",  # !
    # "SVR",
    # "BaggingRegressor",
    # "NuSVR",
    # "RandomForestRegressor",
    # "XGBRegressor",
    # "GradientBoostingRegressor",
    "ExtraTreesRegressor",  # !
    # "AdaBoostRegressor",
    # "PoissonRegressor",
    # "LGBMRegressor",
    # "KNeighborsRegressor",
    # "DecisionTreeRegressor",
    # "MLPRegressor",
    # "HuberRegressor",
    # "GammaRegressor",
    # "LinearSVR",
    # "RidgeCV",
    # "BayesianRidge",
    # "Ridge",
    # "TransformedTargetRegressor",
    # "LinearRegression",
    # "ElasticNetCV",
    # "LassoCV",
    # "LassoLarsIC",
    # "LassoLarsCV",
    # "Lars",
    # "LarsCV",
    # "SGDRegressor",
    # "TweedieRegressor",
    # "GeneralizedLinearRegressor",
    # "ElasticNet",
    # "Lasso",
    # "RANSACRegressor",
    # "OrthogonalMatchingPursuitCV",
    # "PassiveAggressiveRegressor",
    # "GaussianProcessRegressor",
    # "OrthogonalMatchingPursuit",
    # "ExtraTreeRegressor",
    # "DummyRegressor",
    # "LassoLars",
    # "KernelRidge",
]
regr_list = [est for est in all_estimators() if (est[0] in regressors)]

# %%
df_train = utils.load_data("train")

# %%
df_features = df_train.copy()

# Feature Engineering
df_features = utils.add_date_features(df_features)
if add_ltm_columns:
    df_features = utils.add_ltm_kpis(df_features, columns=("target",))

# Feature Manipulation
numerical_features = list(df_features.select_dtypes(include=["number"]).drop(columns=["target"]).columns)
df_features[categorical_features] = df_features[categorical_features].astype(
    {col: "category" for col in categorical_features}
)

# Improve Data Quality
df_features = utils.replace_minus_one_with_mean(df_features, include_columns=numerical_features)

# Feature Selection
selected_features = categorical_features + numerical_features
selected_features = [feature for feature in selected_features if feature not in drop_features]
df_features = df_features[selected_features]
print("\nSelected Features:")
display(df_features.head(3))
print("\n")


# %%
# Split data
X_train, X_validate, X_test, y_train, y_validate, y_test = utils.train_test_validation_split(
    df_features, df_train, validation_year, test_year
)
# %%
# Define model
reg = LazyRegressor(regressors=regr_list, **model_params)
models, predictions = reg.fit(X_train, X_validate, y_train, y_validate)
print(models)


# # Fit model on train set
# model.fit(X_train, y_train)
# # Output info about fitted model
# try:
#     feature_importances = dict(zip(model.feature_names_in_, model.feature_importances_, strict=True))
#     sorted_feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
#     print("\nFeature Importances:")
#     for feature, importance in sorted_feature_importances.items():
#         print(f"{feature}: {importance:.3f}")
# except AttributeError:
#     logging.warning("Feature importances not available for this model.")
# print("")

# # Predict
# df_pred = X_validate.copy()
# df_pred["prediction"] = model.predict(df_pred)

# # Add necessary columns for evaluation
# df_pred["date"] = df_train["date"]
# df_pred["target"] = df_train["target"]
# X_train, df_pred = utils.identify_future_launches(X_train, df_pred)
# df_pred.loc[df_pred.index, "zero_actuals"] = df_pred["zero_actuals"]

# # Evaluate on validation set
# print("\nValidation year:", validation_year)
# cyme_score = helper.compute_metric(df_pred)
# metric_recent, metric_future = helper.compute_metric_terms(df_pred)

# print("CYME:", round(cyme_score, 3), "- Recent:", round(metric_recent, 3), "Future:", round(metric_future, 3))
# print("---")


# # %%
# # Predict on test set
# df_test = X_test.copy()
# df_test["prediction"] = model.predict(df_test)

# # Add necessary columns for evaluation
# df_test["date"] = df_train["date"]
# df_test["target"] = df_train["target"]
# X_train, df_test = utils.identify_future_launches(X_train, df_test)
# df_test.loc[df_test.index, "zero_actuals"] = df_test["zero_actuals"]

# # Evaluate on test set
# print("\nTest year:", test_year)
# cyme_score = helper.compute_metric(df_test)
# metric_recent, metric_future = helper.compute_metric_terms(df_test)

# print("CYME:", round(cyme_score, 3), "- Recent:", round(metric_recent, 3), "Future:", round(metric_future, 3))
# print("---")


# # %%
# # Prepare submission data and file
# if submit:
#     submission = utils.load_data("predict")

#     # Perform all data preparation steps like in the training data (cleaning, filling, feature engineering)
#     submission = utils.add_date_features(submission)
#     if add_ltm_columns:
#         submission = utils.add_ltm_kpis(submission, columns=("target",))

#     submission[categorical_features] = submission[categorical_features].astype(
#         {col: "category" for col in categorical_features}
#     )
#     df_features = utils.replace_minus_one_with_mean(df_features, include_columns=numerical_features)

#     submission["prediction"] = model.predict(submission[selected_features])
#     root = Path.cwd()  # .parent
#     utils.save_submission_file(submission, root=root, user=P["user"])  # NOTE: Uncomment to save submission file
