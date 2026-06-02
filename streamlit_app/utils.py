import os
import numpy as np
import pandas as pd
import streamlit as st
import pyarrow.dataset as ds

PARQUET_PATH = "data/kaggle/working/engineered_features_encoded.parquet"
PRED_DIR = "predictions"
ASSET_DIR = "assets"

BASE_COLS = [
    "date", "store_nbr", "item_nbr", "unit_sales",
    "kalman_sales", "wavelet_soft_sales", "wavelet_hard_sales",
]

REQUIRED_PRED_COLS = [
    "date", "store_nbr", "item_nbr", "actual_sales", "predicted_sales"
]

TRACK_FILES = {
    "Track 1 - Baseline": "preds_track1.csv",
    "Track 2 - Wavelet": "preds_track2.csv",
    "Track 3 - TFT": "preds_track3.csv",
}

TRACK_IMAGES = {
    "Track 1 - Baseline": "feature_importance_track1.png",
    "Track 2 - Wavelet": "feature_importance_track2.png",
    "Track 3 - TFT": "feature_importance_track3.png",
}


@st.cache_data(show_spinner="Đang đọc danh sách Store...")
def get_store_list():
    dataset = ds.dataset(PARQUET_PATH, format="parquet")
    table = dataset.to_table(columns=["store_nbr"])
    df = table.to_pandas()
    return sorted(df["store_nbr"].dropna().unique())


@st.cache_data(show_spinner="Đang đọc danh sách Item...")
def get_item_list(store_id):
    dataset = ds.dataset(PARQUET_PATH, format="parquet")
    table = dataset.to_table(
        columns=["store_nbr", "item_nbr"],
        filter=ds.field("store_nbr") == store_id
    )
    df = table.to_pandas()
    return sorted(df["item_nbr"].dropna().unique())


@st.cache_data(show_spinner="Đang load dữ liệu Store/Item...")
def load_store_item_2017(store_id, item_id):
    dataset = ds.dataset(PARQUET_PATH, format="parquet")

    table = dataset.to_table(
        columns=BASE_COLS,
        filter=(
            (ds.field("store_nbr") == store_id) &
            (ds.field("item_nbr") == item_id)
        )
    )

    df = table.to_pandas()
    df["date"] = pd.to_datetime(df["date"])

    df["unit_sales"] = df["unit_sales"].clip(lower=0)

    # Sales không nên âm khi visual hóa
    for col in ["kalman_sales", "wavelet_soft_sales", "wavelet_hard_sales"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    df = df[df["date"].dt.year == 2017].copy()
    return df.sort_values("date")


def create_dummy_prediction(df):
    temp = df.copy()
    np.random.seed(42)

    noise = np.random.normal(loc=1.0, scale=0.08, size=len(temp))

    temp["actual_sales"] = temp["unit_sales"]
    temp["predicted_sales"] = (temp["unit_sales"] * noise).clip(lower=0).round(4)

    return temp[
        ["date", "store_nbr", "item_nbr", "actual_sales", "predicted_sales"]
    ]


@st.cache_data(show_spinner="Đang đọc file dự báo...")
def load_prediction_csv(path):
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    missing = [c for c in REQUIRED_PRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} thiếu cột: {missing}")

    df = df[REQUIRED_PRED_COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["actual_sales"] = df["actual_sales"].clip(lower=0)
    df["predicted_sales"] = df["predicted_sales"].clip(lower=0).round(4)

    return df


@st.cache_data(show_spinner="Đang kiểm tra 3 track model...")
def load_all_predictions():
    predictions = {}

    for track_name, filename in TRACK_FILES.items():
        path = os.path.join(PRED_DIR, filename)
        if os.path.exists(path):
            predictions[track_name] = load_prediction_csv(path)

    return predictions


def filter_prediction(pred_df, store_id, item_id):
    return pred_df[
        (pred_df["store_nbr"] == store_id) &
        (pred_df["item_nbr"] == item_id)
    ].sort_values("date")


def calculate_metrics(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)

    actual = np.clip(actual, 0, None)
    predicted = np.clip(predicted, 0, None)

    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))
    nwrmsle = np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))

    return {
        "NWRMSLE": round(nwrmsle, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
    }


def build_leaderboard(metrics_dict):
    rows = []

    for model_name, m in metrics_dict.items():
        rows.append({
            "Model": model_name,
            "NWRMSLE": m["NWRMSLE"],
            "RMSE": m["RMSE"],
            "MAE": m["MAE"],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("NWRMSLE").reset_index(drop=True)
    df.insert(0, "Rank", [f"#{i+1}" for i in range(len(df))])
    return df