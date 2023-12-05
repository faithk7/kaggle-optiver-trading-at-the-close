import os

import catboost as cbt
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.neighbors import NearestNeighbors

# define hyperparamters
N_FOLDES = 5
model_dict = {
    "lgb": lgb.LGBMRegressor(objective="regression_l1", n_estimators=500),
    "xgb": xgb.XGBRegressor(
        tree_method="hist", objective="reg:absoluteerror", n_estimators=500
    ),
    "cbt": cbt.CatBoostRegressor(objective="MAE", iterations=3000),
}
EARLY_STOPPING = 100

# add stopping rounds supports for the above three models
lgb.early_stopping(EARLY_STOPPING)


# load the train df
df = pd.read_csv("/Users/kaiqu/kaggle-datasets/train.csv")
# load the test df
test_df = pd.read_csv("/Users/kaiqu/kaggle-datasets/example_test_files/test.csv")

# drop the stocks where contains the nonsensible NaN values; we are going to drop those lines, as they do not seem to be making any sense
df = df.dropna(
    subset=["imbalance_size", "target"]
)  # for df, drop where imbalance_size is NaN or targets is NaN

print(df.info())

# augment the train df dataset where it contains the prev day value
df_grouped = df.sort_values(by="date_id").groupby(["stock_id", "seconds_in_bucket"])
df["prev_target"] = df_grouped["target"].shift(1)
avg_target_per_stock = df.groupby("stock_id")["target"].mean()
df["prev_target"] = df.apply(
    lambda row: avg_target_per_stock[row["stock_id"]]
    if pd.isna(row["prev_target"])
    else row["prev_target"],
    axis=1,
)

# Showing the first few rows of the updated DataFrame
print(df.head())
print(df.info())
print(df.isna().sum())


# fit the model with the training dataset
Y = df["target"]
# drop target column from df, row_id, and time_id also seem to be useless
X = df.drop(["target", "row_id", "time_id"], axis=1)
X = X.fillna(0)
index = np.arange(len(X))
models = []
print(Y.info())
print(X.info())


def train(fold, modelname):
    model = model_dict[modelname]
    print(
        "train set", len(X[index % N_FOLDES != fold]), len(Y[index % N_FOLDES != fold])
    )
    print(
        "eval set", len(X[index % N_FOLDES == fold]), len(Y[index % N_FOLDES == fold])
    )
    model.fit(
        # ? is this a good way to split the data?
        X[index % N_FOLDES != fold],
        Y[index % N_FOLDES != fold],
        eval_set=[(X[index % N_FOLDES == fold], Y[index % N_FOLDES == fold])],
        verbose=10,
        early_stopping_rounds=100,  # stop training if the validation score is not improving for 100 rounds
    )
    models.append(model)
    joblib.dump(model, f"./models/{modelname}_{fold}.model")


# if models directory not exist, make a directory called models
if not os.path.exists("./models"):
    os.makedirs("./models")

for fold in range(N_FOLDES):
    # for model_name in model_dict.keys():
    #     train(fold, model_name)
    # train(fold, "lgb") # NOTE: this is not working with the early_stopping rounds
    train(fold, "xgb")
    train(fold, "cbt")

# set up the metrics

# predict the test dataset using the offline API
