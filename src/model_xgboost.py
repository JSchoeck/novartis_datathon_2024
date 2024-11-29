# %%
import multiprocessing

from IPython.display import display
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

import helper
import utils

P = utils.load_settings()["params"]


# %%
df_train = utils.load_data("train")
df_train.head(3)


# %%
df_features = df_train.copy()

df_features = utils.add_date_features(df_features)

potential_corr_features = ["cluster_nl", "corporation", "drug_id", "ind_launch_date", "launch_date", "date"]
categorical_features = [
    "brand",
    "cluster_nl",
    "corporation",
    "country",
    "drug_id",
    "indication",
    "therapeutic_area",
]
# date_feature = ["launch_date", "date", "ind_launch_date"]
date_features = ["launch_date", "date", "ind_launch_date", "weekofyear", "month", "year"]

numerical_features = list(df_features.select_dtypes(include=["number"]).drop(columns=["target"]).columns)
df_features[categorical_features] = df_features[categorical_features].astype(
    {col: "category" for col in categorical_features}
)

selected_features = date_features + categorical_features + numerical_features
selected_features = [feature for feature in selected_features if feature not in potential_corr_features]
s_target = df_features.pop("target")
selected_features = numerical_features  # BUG: categorical features are not working for XGBRegressor currently
df_features = df_features[selected_features]

print(s_target.dtypes, s_target.shape, "\n")
print(df_features.dtypes, df_features.shape)
display(df_features.head(3))


# %%
ts_cv = TimeSeriesSplit(gap=0, n_splits=5)
all_splits = list(
    ts_cv.split(
        df_features,
        s_target,
    )
)
print("Splits:", len(all_splits))

# %%
xgb_model = XGBRegressor(
    enable_categorical=True,
    tree_method="hist",
    max_depth=2,
    n_estimators=50,
    max_cat_threshold=1000,
    # eval_metric = helper._metric,
    n_jobs=multiprocessing.cpu_count() - 1,
)


# %%
scores = []
cymes = []
for idx in all_splits:
    print(df_train["date"].iloc[idx[0][0]], "to", df_train["date"].iloc[idx[0][-1]])

    X_train, y_train = df_features.iloc[idx[0]], s_target.iloc[idx[0]]
    xgb_model.fit(X_train, y_train)
    X_test, y_test = df_features.iloc[idx[1]].copy(), s_target.iloc[idx[1]]

    # print(score := xgb_model.score(X_test, y_test))
    # scores.append(score)

    X_test.loc[:, "prediction"] = xgb_model.predict(X_test[selected_features])
    # NOTE: required for metric calculation are ["cluster_nl", "date", "target", "prediction", "zero_actuals"]
    # TODO: check if indexing is correct for the following columns ["cluster_nl", "date", "target", "zero_actuals"]
    if "cluster_nl" not in X_test.columns:
        X_test.loc[idx[1], "cluster_nl"] = df_train["cluster_nl"].loc[idx[1]]
    if "date" not in X_test.columns:
        X_test.loc[idx[1], "date"] = df_train["date"].loc[idx[1]]
    if "target" not in X_test.columns:
        X_test.loc[idx[1], "target"] = s_target.loc[idx[1]]
    # TODO: check if zero_actuals is correctly set, or if it should be True for NaN values maybe?
    X_test.loc[:, "zero_actuals"] = False
    X_test.loc[X_test["target"] == 0, "zero_actuals"] = True
    print("CYME:", cyme := helper.compute_metric(X_test), "\n")
    cymes.append(cyme)

# print("Mean score:", round(sum(scores) / len(scores), 3))
print("---\nMean CYME:", round(sum(cymes) / len(cymes), 3), "\n---")


# %%
submission = utils.load_data("predict")
submission = utils.add_date_features(submission)
submission["prediction"] = xgb_model.predict(submission[selected_features])
# utils.save_submission_file(submission) # NOTE: Uncomment to save submission file
