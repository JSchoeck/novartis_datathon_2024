import multiprocessing

import dateutil.parser
import numpy as np
from catboost import CatBoostRegressor

import helper
import utils

P = utils.load_settings()["params"]

# Import data
df_train = utils.load_data("train")
df_submission = utils.load_data("predict")

columns_to_average = ["che_pc_usd", "che_perc_gdp", "public_perc_che", "insurance_perc_che", "price_month"]

df_train = utils.replace_minus_one_with_mean(df_train, columns_to_average)
df_submission = utils.replace_minus_one_with_mean(df_submission, columns_to_average)

df_train = utils.remove_outlier_data(df_train, "prev_perc", 4)

df_train = utils.add_date_features(df_train)
df_submission = utils.add_date_features(df_submission)
target_series = df_train.pop("target")

# ensure dates are saved for later
df_train_dates = df_train["date"]
df_submission_dates = df_submission["date"]

# update date features
date_features = [
    "date",
    # "launch_date",
    # "ind_launch_date"
]

# specify the rest of the features
cat_features = [
    "brand",
    "cluster_nl",
    "country",
    "drug_id",
    "indication",
]

numerical_features = list(df_train.select_dtypes(include=["number"]).columns)
features = cat_features + numerical_features
print(features)
# Ensure category features are treated as such in both traijning and submission data
df_train = df_train[features].astype({col: "category" for col in cat_features})
df_submission = df_submission[features].astype({col: "category" for col in cat_features})

# Set year split
test_year = 2022

# Split data to training and test data
X_train, y_train = df_train[df_train["year"] < test_year], target_series[df_train["year"] < test_year]
X_test = df_train[df_train["year"] >= test_year]

# Set up regressor model
model = CatBoostRegressor(
    depth=12,
    cat_features=cat_features,
    eval_metric=helper.CYMEMetric(),
)

# Fit model
model.fit(X_train, y_train)

# Get predictions
df_pred = X_test.copy()
df_pred["prediction"] = model.predict(X_test)

# Set up Zero Actuals
print(df_train.columns)
df_pred["date"] = df_train_dates
df_pred["target"] = target_series
X_train, df_pred = utils.identify_future_launches(X_train, df_pred)
df_pred.loc[df_pred.index, "zero_actuals"] = df_pred["zero_actuals"]

# Score the prediction
cyme_score = helper.compute_metric(df_pred)
metric_recent, metric_future = helper.compute_metric_terms(df_pred)

print("Test year:", test_year)
print("CYME:", cyme_score)
print("CYME Recent Launches:", metric_recent)
print("CYME Future Launches:", metric_future)

df_submission["prediction"] = model.predict(df_submission)  # NOTE: Uncomment to save submission file
df_submission["date"] = df_submission_dates
utils.save_submission_file(df_submission)  # NOTE: Uncomment to save submission file
