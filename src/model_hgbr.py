from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import helper
import utils

P = utils.load_settings()["params"]

# Import data
df_train = utils.load_data("train")
df_submission = utils.load_data("predict")

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


one_hot_encoder = make_column_transformer(
    (
        OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
        make_column_selector(dtype_include="category"),  # type: ignore
    ),
    remainder="passthrough",
)

pipeline = make_pipeline(
    one_hot_encoder,
    HistGradientBoostingRegressor(
        verbose=True,
        max_features=0.8,  # type: ignore
        max_iter=200,
    ),
    memory="cache",
)

#### Results #########################
# Test year: 2022
# CYME: 0.08803927
# CYME Recent Launches: 0.09033301
# CYME Future Launches: 0.04621431

#### Results #########################
# max_features=0.9, max_iter=200
# Test year: 2022
# CYME: 0.0789728
# CYME Recent Launches: 0.08051437
# CYME Future Launches: 0.05086318

#### Results #########################
# max_features=0.8, max_iter=200
# Test year: 2022
# CYME: 0.07748433
# CYME Recent Launches: 0.07882676
# CYME Future Launches: 0.05300604

# Set year split
test_year = 2022

# Split data to training and test data
X_train, y_train = df_train[df_train["year"] < test_year], target_series[df_train["year"] < test_year]
X_test = df_train[df_train["year"] >= test_year]

# Set up regressor model
model = pipeline

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

df_submission["prediction"] = model.predict(df_submission)
df_submission["date"] = df_submission_dates
utils.save_submission_file(df_submission)  # NOTE: Uncomment to save submission file
