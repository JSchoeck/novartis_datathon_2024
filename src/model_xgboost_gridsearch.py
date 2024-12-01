# %%
import multiprocessing
from pathlib import Path

import pandas as pd
from IPython.display import display
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

import helper
import utils

print("---")
logging = utils.get_logger(__name__)
P = utils.load_settings()["params"]

# %%
################ Parameters ################
validation_year = 2022
test_year = 2022
add_ltm_columns = False
model_params = {
    "enable_categorical": True,  # good?
    "tree_method": "hist",  # good?
    "max_depth": 6,  # 4-7 is good, more is overfitting with default number of features
    "n_estimators": 200,  # 500 for dev, 200 for submit??
    "max_cat_threshold": 1000,  # good?
    "n_jobs": multiprocessing.cpu_count() - 1,
}
drop_features = [
    # "price_month",
    # "insurance_perc_che",
    "therapeutic_area",
    "country",
    "corporation",
    "drug_id",
]
cv_estimator = XGBRegressor(enable_categorical=True)

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


# %%
df_train = utils.load_data("train")
if submit:
    submission = utils.load_data("predict")

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

# ts_cv = TimeSeriesSplit(gap=0, n_splits=5)
# all_splits = list(
#     ts_cv.split(
#         df_features.drop(columns=["target"]),
#         df_features["target"],
#     )
# )
# print("Splits:", len(all_splits))
# %%
# Define model
param_grid = {"max_depth": [3, 4, 5], "n_estimators": [17, 200, 225], "learning_rate": [0.1]}

# X_validate["date"] = df_train["date"]
# X_validate["target"] = df_train["target"]
X_train, X_validate = utils.identify_future_launches(X_train, X_validate)
non_numerical_columns = X_train.select_dtypes(exclude=["number"]).columns
X_train[non_numerical_columns] = X_train[non_numerical_columns].astype(
    {col: "category" for col in non_numerical_columns}
)

# cyme_scorer = helper.cyme_scorer()
grid_search = GridSearchCV(
    estimator=cv_estimator,
    param_grid=param_grid,
    # scoring=cyme_scorer,
    # cv=ts_cv,
    verbose=1,
    n_jobs=multiprocessing.cpu_count() - 1,
    # error_score="raise",
)
grid_search.fit(X_validate, y_validate)
model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

# Fit model on train set
model.fit(X_train, y_train)
# Output info about fitted model
try:
    feature_importances = dict(zip(model.feature_names_in_, model.feature_importances_, strict=True))
    sorted_feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    print("\nFeature Importances:")
    for feature, importance in sorted_feature_importances.items():
        print(f"{feature}: {importance:.3f}")
except AttributeError:
    logging.warning("Feature importances not available for this model.")
print("")

# Predict
df_pred = X_validate.copy()
df_pred["prediction"] = model.predict(df_pred)

# Add necessary columns for evaluation
df_pred["date"] = df_train["date"]
df_pred["target"] = df_train["target"]
X_train, df_pred = utils.identify_future_launches(X_train, df_pred)
df_pred.loc[df_pred.index, "zero_actuals"] = df_pred["zero_actuals"]

# Evaluate on validation set
print("\nValidation year:", validation_year)
cyme_score = helper.compute_metric(df_pred)
metric_recent, metric_future = helper.compute_metric_terms(df_pred)

print("CYME:", round(cyme_score, 3), "- Recent:", round(metric_recent, 3), "Future:", round(metric_future, 3))
print("---")


# %%
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
#     # Perform all data preparation steps like in the training data (cleaning, filling, feature engineering)
#     submission = utils.add_date_features(submission)  # type: ignore
#     if add_ltm_columns:
#         submission_length = len(submission)
#         df_features["date"] = df_train["date"]
#         submission = utils.add_ltm_kpis(
#             pd.concat(
#                 [df_features.drop(columns=["ltm_target"]), submission[df_features.drop(columns=["ltm_target"]).columns]]
#             ).reset_index(),
#             columns=("target",),
#         )
#         submission = utils.add_ltm_kpis(submission, columns=("target",))
#         submission = submission[-submission_length:]
#         submission[["brand", "cluster_nl", "indication"]] = submission[["brand", "cluster_nl", "indication"]].astype(
#             {col: "category" for col in ["brand", "cluster_nl", "indication"]}
#         )
#     else:
#         submission[categorical_features] = submission[categorical_features].astype(
#             {col: "category" for col in categorical_features}
#         )
#     submission = utils.replace_minus_one_with_mean(submission, include_columns=numerical_features)

#     submission["prediction"] = model.predict(submission[selected_features])
#     root = Path.cwd()  # .parent
#     utils.save_submission_file(submission, root=root, user=P["user"])  # NOTE: Uncomment to save submission file
