import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="GPU Forecast System", layout="wide")

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "Name": "GPU",
        "Retail Price": "Price"
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["GPU", "Date"]).reset_index(drop=True)

    df["month_index"] = df.groupby("GPU").cumcount()

    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df.groupby("GPU")["Price"].shift(lag)

    df["rolling_mean_3"] = (
        df.groupby("GPU")["Price"]
        .rolling(3).mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_std_3"] = (
        df.groupby("GPU")["Price"]
        .rolling(3).std()
        .reset_index(level=0, drop=True)
    )

    df = df.dropna().reset_index(drop=True)

    return df


# =====================================================
# TRAIN SINGLE GPU (LAZY TRAINING)
# =====================================================
@st.cache_resource
def train_single_gpu(df, gpu):

    features = [
        "month_index",
        "lag_1", "lag_2", "lag_3",
        "rolling_mean_3", "rolling_std_3"
    ]

    df_gpu = df[df["GPU"] == gpu]

    X = df_gpu[features]
    y = df_gpu["Price"]

    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    return model, features


# =====================================================
# NAIVE BASELINE
# =====================================================
def naive_baseline(df):
    df_sorted = df.sort_values(["GPU", "Date"])
    df_sorted["Naive"] = df_sorted.groupby("GPU")["Price"].shift(1)
    df_sorted = df_sorted.dropna()
    return mean_absolute_error(df_sorted["Price"], df_sorted["Naive"])


# =====================================================
# FORECAST FUNCTION
# =====================================================
def forecast_gpu(model, df, features, gpu_name, months):

    df_gpu = df[df["GPU"] == gpu_name].copy()
    df_future = df_gpu.copy()
    predictions = []

    for _ in range(months):

        last_row = df_future.iloc[-1:].copy()
        X_last = last_row[features]

        next_price = model.predict(X_last)[0]
        predictions.append(next_price)

        new_row = last_row.copy()
        new_row["Price"] = next_price
        new_row["month_index"] += 1

        new_row["lag_3"] = last_row["lag_2"].values
        new_row["lag_2"] = last_row["lag_1"].values
        new_row["lag_1"] = next_price

        last_prices = df_future["Price"].tail(2).tolist() + [next_price]
        new_row["rolling_mean_3"] = np.mean(last_prices)
        new_row["rolling_std_3"] = np.std(last_prices)

        df_future = pd.concat([df_future, new_row], ignore_index=True)

    future_dates = pd.date_range(
        start=df_gpu["Date"].max() + pd.DateOffset(months=1),
        periods=months,
        freq="MS"
    )

    return pd.DataFrame({
        "GPU": gpu_name,
        "Date": future_dates,
        "Predicted_Price": predictions
    })


# =====================================================
# UI
# =====================================================
st.title("🚀 GPU Price Forecast System")

df = load_data("gpu_price_history.csv")

st.subheader("📊 Baseline Performance")
naive_mae = naive_baseline(df)
st.metric("Naive MAE (Lag-1)", round(naive_mae, 2))

st.info("Model will train only when you run forecast (faster loading).")

# ================= FORECAST =================
st.subheader("🔮 Forecast")

mode = st.radio("Mode", ["Single GPU", "All GPUs"])
months = st.slider("Forecast Months", 1, 24, 12)

if mode == "Single GPU":
    selected_gpu = st.selectbox("Select GPU", df["GPU"].unique())

if st.button("Run Forecast"):

    if mode == "Single GPU":

        with st.spinner("Training model..."):
            model, features = train_single_gpu(df, selected_gpu)

        result = forecast_gpu(
            model,
            df,
            features,
            selected_gpu,
            months
        )

        st.subheader("Forecast Result")
        st.dataframe(result)
        st.line_chart(result.set_index("Date")["Predicted_Price"])

    else:

        all_results = []
        gpu_list = df["GPU"].unique()
        progress = st.progress(0)

        for i, gpu in enumerate(gpu_list):

            model, features = train_single_gpu(df, gpu)

            result = forecast_gpu(
                model,
                df,
                features,
                gpu,
                months
            )

            all_results.append(result)
            progress.progress((i + 1) / len(gpu_list))

        final_df = pd.concat(all_results)

        st.subheader("All GPU Forecast Result")
        st.dataframe(final_df)

        pivot_df = final_df.pivot(index="Date", columns="GPU", values="Predicted_Price")
        st.line_chart(pivot_df)
