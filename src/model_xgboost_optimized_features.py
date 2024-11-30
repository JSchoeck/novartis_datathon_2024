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
################ Parameters ################
test_year = 2022
enable_categorical = True  # good?
tree_method = "hist"  # good?
max_depth = 4  # 4 for dev, 12-30 for submit??
n_estimators = 200  # 500 for dev, 200 for submit??
max_cat_threshold = 1000  # good?

################# Settings #################
n_jobs = multiprocessing.cpu_count() - 1

# %%
df_train = utils.load_data("train")

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
selected_features = [feature for feature in selected_features if feature not in drop_features]
s_target = df_features.pop("target")
df_features = df_features[selected_features]

print(s_target.dtypes, s_target.shape, "\n")
print(df_features.dtypes, df_features.shape)
display(df_features.head(3))


# %%
# Define model
model = XGBRegressor(
    enable_categorical=enable_categorical,
    tree_method=tree_method,
    max_depth=max_depth,  # 4 for dev, 12 for submit
    n_estimators=n_estimators,
    max_cat_threshold=max_cat_threshold,
    n_jobs=n_jobs,
)

# Split data into train and test set
X_train, y_train = df_features[df_features["year"] < test_year], s_target[df_features["year"] < test_year]
X_test, y_test = df_features[df_features["year"] >= test_year], s_target[df_features["year"] >= test_year]

# Fit model on train set
model.fit(X_train, y_train)

# Predict
df_pred = X_test.copy()
df_pred["prediction"] = model.predict(df_pred)

# Add necessary columns for evaluation
df_pred["date"] = df_train["date"]
df_pred["target"] = df_train["target"]
X_train, df_pred = utils.identify_future_launches(X_train, df_pred)
df_pred.loc[df_pred.index, "zero_actuals"] = df_pred["zero_actuals"]

# Evaluate
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
# Prepare submission data and file
submission = utils.load_data("predict")
submission = utils.add_date_features(submission)

submission[categorical_features] = submission[categorical_features].astype(
    {col: "category" for col in categorical_features}
)

submission["prediction"] = model.predict(submission[selected_features])
root = Path.cwd()  # .parent
utils.save_submission_file(submission, root=root, user=P["user"])  # NOTE: Uncomment to save submission file
