from itertools import combinations

import numpy as np
import pandas as pd
from numba import njit, prange

df_train = pd.read_csv("/Users/kaiqu/kaggle-datasets/train.csv")

global_stock_id_feats = {
    "median_size": df_train.groupby("stock_id")["bid_size"].median()
    + df_train.groupby("stock_id")["ask_size"].median(),
    "std_size": df_train.groupby("stock_id")["bid_size"].std()
    + df_train.groupby("stock_id")["ask_size"].std(),
    "ptp_size": df_train.groupby("stock_id")["bid_size"].max()  # ? what is ptp_size?
    - df_train.groupby("stock_id")["bid_size"].min(),
    "median_price": df_train.groupby("stock_id")["bid_price"].median()
    + df_train.groupby("stock_id")["ask_price"].median(),
    "std_price": df_train.groupby("stock_id")["bid_price"].std()
    + df_train.groupby("stock_id")["ask_price"].std(),
    "ptp_price": df_train.groupby("stock_id")["bid_price"].max()  # ? what is ptp_price?
    - df_train.groupby("stock_id")["ask_price"].min(),
}


def imbalance_features(df):
    prices = [
        "reference_price",
        "far_price",
        "near_price",
        "ask_price",
        "bid_price",
        "wap",
    ]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    df["volume"] = df.eval("ask_size + bid_size")
    df["mid_price"] = df.eval("(ask_price + bid_price) / 2")
    df["liquidity_imbalance"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["matched_imbalance"] = df.eval(
        "(imbalance_size-matched_size)/(matched_size+imbalance_size)"
    )
    df["size_imbalance"] = df.eval("bid_size / ask_size")

    for c in combinations(prices, 2):
        df[f"{c[0]}_{c[1]}_imb"] = df.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [["ask_price", "bid_price", "wap", "reference_price"], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, df)
        df[triplet_feature.columns] = triplet_feature.values

    df["imbalance_momentum"] = (
        df.groupby(["stock_id"])["imbalance_size"].diff(periods=1) / df["matched_size"]
    )
    df["price_spread"] = df["ask_price"] - df["bid_price"]
    df["spread_intensity"] = df.groupby(["stock_id"])["price_spread"].diff()
    df["price_pressure"] = df["imbalance_size"] * (df["ask_price"] - df["bid_price"])
    df["market_urgency"] = df["price_spread"] * df["liquidity_imbalance"]
    df["depth_pressure"] = (df["ask_size"] - df["bid_size"]) * (
        df["far_price"] - df["near_price"]
    )

    for func in ["mean", "std", "skew", "kurt"]:
        df[f"all_prices_{func}"] = df[prices].agg(func, axis=1)
        df[f"all_sizes_{func}"] = df[sizes].agg(func, axis=1)

    for col in [
        "matched_size",
        "imbalance_size",
        "reference_price",
        "imbalance_buy_sell_flag",
    ]:
        for window in [1, 2, 3, 10]:
            df[f"{col}_shift_{window}"] = df.groupby("stock_id")[col].shift(window)
            df[f"{col}_ret_{window}"] = df.groupby("stock_id")[col].pct_change(window)

    for col in ["ask_price", "bid_price", "ask_size", "bid_size"]:
        for window in [1, 2, 3, 10]:
            df[f"{col}_diff_{window}"] = df.groupby("stock_id")[col].diff(window)
    return df.replace([np.inf, -np.inf], 0)


def other_features(df):
    df["dow"] = df["date_id"] % 5  # Day of the week
    df["seconds"] = df["seconds_in_bucket"] % 60  # Seconds
    df["minute"] = df["seconds_in_bucket"] // 60  # Minutes
    for key, value in global_stock_id_feats.items():
        df[f"global_{key}"] = df["stock_id"].map(value.to_dict())

    return df


def calculate_triplet_imbalance_numba(price, df):
    df_values = df[price].values
    comb_indices = [
        (price.index(a), price.index(b), price.index(c))
        for a, b, c in combinations(price, 3)
    ]
    features_array = compute_triplet_imbalance(df_values, comb_indices)
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)
    return features


@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))
    for i in prange(num_combinations):
        a, b, c = comb_indices[i]
        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = (
                df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val
            )
            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features


def generate_all_features(df):
    cols = [c for c in df.columns if c not in ["row_id", "time_id", "target"]]
    df = df[cols]
    df = imbalance_features(df)
    df = other_features(df)
    feature_name = [
        i for i in df.columns if i not in ["row_id", "target", "time_id", "date_id"]
    ]

    return df[feature_name]


if __name__ == "__main__":
    df = pd.read_csv("/Users/kaiqu/kaggle-datasets/train.csv")
    df_feat = generate_all_features(df)
    print(df_feat.head())
    print(df_feat.info())
    missing_values_summary = df_feat.isnull().sum()
    print(missing_values_summary)
    for column, missing_count in missing_values_summary.items():
        print(f"Column: {column}, Missing Values: {missing_count}")
