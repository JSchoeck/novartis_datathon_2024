# %%
# import warnings


# def custom_warning_handler(message, category, filename, lineno, file=None, line=None):
#     if not issubclass(category, FutureWarning):
#         breakpoint()
#         warnings.showwarning(message, category, filename, lineno, file, line)

# warnings.showwarning = custom_warning_handler

# %%
import multiprocessing
from pathlib import Path

import xgboost as xgb
from IPython.display import display
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor

import helper
import utils

logging = utils.get_logger(__name__)
P = utils.load_settings()["params"]


# %%
df_train = utils.load_data("train")

# %%

# split up indication values  - did not seem to increase (lower) score
# new_indication = []
# for ind in df_train.indication:
#     indications_per_ind = ind.strip("[] ").split(", ")
#     indication_2 = []
#     for indication in indications_per_ind:
#         indication_2.append(indication[1:-1])
#     new_indication.append(indication_2)

# df_train["indication"] = new_indication

# df_train = df_train.explode("indication",ignore_index=True)

# %%
df_features = df_train.copy()

df_features = utils.add_date_features(df_features)

drop_features = [
    # "price_month",
    # "insurance_perc_che",
    "therapeutic_area",
    "country",
    "corporation",
    "drug_id",
]
potential_corr_features = [
    # "cluster_nl",
    # "corporation",
    # "drug_id",
    # "ind_launch_date",
    # "launch_date",
    # "date",
    # "country",
    # "indication"
]
categorical_features = [
    "brand",
    "cluster_nl",
    "corporation",
    "country",
    "drug_id",
    "indication",
    "therapeutic_area",
]

# date_features = ["day", "week_of_year", "month", "year"]

numerical_features = list(df_features.select_dtypes(include=["number"]).drop(columns=["target"]).columns)

df_features[categorical_features] = df_features[categorical_features].astype(
    {col: "category" for col in categorical_features}
)

selected_features = categorical_features + numerical_features
drop_features += potential_corr_features
selected_features = [feature for feature in selected_features if feature not in drop_features]
s_target = df_features.pop("target")
df_features = df_features[selected_features]

print(s_target.dtypes, s_target.shape, "\n")
print(df_features.dtypes, df_features.shape)
display(df_features.head(3))


# %%
# model = XGBRegressor(
#     enable_categorical=True,
#     tree_method="hist",
#     max_depth=6,  # higher number takes much longer top run. 5 or 6 is good for quick checks
#     n_estimators=100,
#     max_cat_threshold=1000,
#     n_jobs=multiprocessing.cpu_count() - 1,
# )


# %%
# Cell with TimeSeriesSplit

# ts_cv = TimeSeriesSplit(gap=0, n_splits=5)
# all_splits = list(
#     ts_cv.split(
#         df_features,
#         s_target,
#     )
# )
# print("Splits:", len(all_splits))

# %%
# scores = []
# cymes = []
# for idx in all_splits:
#     print(df_train["date"].iloc[idx[0][0]], "to", df_train["date"].iloc[idx[0][-1]])
#     # fit
#     X_train, y_train = df_features.iloc[idx[0]], s_target.iloc[idx[0]]
#     model.fit(X_train, y_train)
#     X_test, y_test = df_features.iloc[idx[1]].copy(), s_target.iloc[idx[1]]
#     # predict
#     X_test.loc[:, "prediction"] = model.predict(X_test[selected_features])
#     # NOTE: required for metric calculation are ["cluster_nl", "date", "target", "prediction", "zero_actuals"]
#     # TODO: check if indexing is correct for the following columns ["cluster_nl", "date", "target", "zero_actuals"]
#     if "cluster_nl" not in X_test.columns:
#         X_test.loc[idx[1], "cluster_nl"] = df_train["cluster_nl"].loc[idx[1]]
#     if "date" not in X_test.columns:
#         X_test.loc[idx[1], "date"] = df_train["date"].loc[idx[1]]
#     if "target" not in X_test.columns:
#         X_test.loc[idx[1], "target"] = s_target.loc[idx[1]]
#     # TODO: set
#     X_train, X_test = utils.identify_future_launches(X_train, X_test)
#     df_pred.loc[X_train.index, "zero_actuals"] = X_train["zero_actuals"]
#     df_pred.loc[X_test.index, "zero_actuals"] = X_test["zero_actuals"]

#     # X_test.loc[:, "zero_actuals"] = False
#     # X_test.loc[X_test["date"] < df_features.iloc[idx[1]]["launch_date"], "zero_actuals"] = True
#     print("CYME:", cyme := helper.compute_metric(X_test))
#     cymes.append(cyme)
#     metric_recent, metric_future = helper.compute_metric_terms(X_test)
#     print("CYME Recent Launches:", metric_recent)
#     print("CYME Future Launches:", metric_future, "\n")
# print("---\nMean CYME:", round(sum(cymes) / len(cymes), 3), "\n---")


# %%
# Cell with GridSearchCV

# Define the parameters to search
# param_grid = {
#     "max_depth": [4, 8, 12],
#     "n_estimators": [50, 100, 200],
#     "learning_rate": [0.05, 0.1, 0.2]
# }
# param_grid = {"max_depth": [4, 6], "n_estimators": [50, 100], "learning_rate": [0.1]}

# cyme_scorer = helper.cyme_scorer()
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     # scoring=cyme_scorer,
#     cv=ts_cv,
#     verbose=1,
#     n_jobs=multiprocessing.cpu_count() - 1,
# )
# grid_search.fit(df_features, s_target)
# print("Best parameters found: ", grid_search.best_params_)
# print("Best score: ", grid_search.best_score_)

# %%
# Use the best estimator to make predictions
# model = grid_search.best_estimator_
model = XGBRegressor(
    enable_categorical=True,
    tree_method="hist",
    max_depth=5,  # 4 for dev, 12 for submit
    n_estimators=100,
    max_cat_threshold=1000,
    n_jobs=multiprocessing.cpu_count() - 1,
)

# Split data into train and test set
test_year = 2022

X_train, y_train = df_features[df_features["year"] < test_year], s_target[df_features["year"] < test_year]
X_test, y_test = df_features[df_features["year"] >= test_year], s_target[df_features["year"] >= test_year]

# Fit model on train set
model.fit(X_train, y_train)

# Predict and evaluate
df_pred = X_test.copy()
df_pred["prediction"] = model.predict(df_pred)
df_pred["date"] = df_train["date"]
df_pred["target"] = df_train["target"]
X_train, df_pred = utils.identify_future_launches(X_train, df_pred)
df_pred.loc[df_pred.index, "zero_actuals"] = df_pred["zero_actuals"]

cyme_score = helper.compute_metric(df_pred)
metric_recent, metric_future = helper.compute_metric_terms(df_pred)
feature_importances = dict(zip(model.feature_names_in_, model.feature_importances_, strict=True))
sorted_feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))

print("\nTest year:", test_year)
print("CYME:", cyme_score)
print("CYME Recent Launches:", metric_recent)
print("CYME Future Launches:", metric_future)
print("\nFeature Importances:")
for feature, importance in sorted_feature_importances.items():
    print(f"{feature}: {importance:.3f}")
print("\n")


# %%
submission = utils.load_data("predict")
submission = utils.add_date_features(submission)

submission[categorical_features] = submission[categorical_features].astype(
    {col: "category" for col in categorical_features}
)

submission["prediction"] = model.predict(submission[selected_features])
root = Path.cwd().parent
# utils.save_submission_file(submission, root=root)  # NOTE: Uncomment to save submission file

# %%
