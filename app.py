import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

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

    for lag in [1,2,3]:
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
# TRAIN PER GPU (FAST VERSION)
# =====================================================
@st.cache_resource
def train_models(df):

    models = {}
    metrics = {}

    features = [
        "month_index",
        "lag_1","lag_2","lag_3",
        "rolling_mean_3","rolling_std_3"
    ]

    for gpu in df["GPU"].unique():

        df_gpu = df[df["GPU"] == gpu]

        # holdout split (เร็วกว่า TimeSeriesSplit มาก)
        split = int(len(df_gpu) * 0.8)

        train = df_gpu.iloc[:split]
        test = df_gpu.iloc[split:]

        X_train = train[features]
        y_train = train["Price"]

        X_test = test[features]
        y_test = test["Price"]

        model = xgb.XGBRegressor(
            n_estimators=250,      # ลดลงให้เร็วขึ้น
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        models[gpu] = model
        metrics[gpu] = {"mae": mae, "r2": r2}

    return models, features, metrics

# =====================================================
# NAIVE BASELINE
# =====================================================
def naive_baseline(df):
    df_sorted = df.sort_values(["GPU", "Date"])
    df_sorted["Naive"] = df_sorted.groupby("GPU")["Price"].shift(1)
    df_sorted = df_sorted.dropna()
    return mean_absolute_error(df_sorted["Price"], df_sorted["Naive"])

# =====================================================
# FORECAST
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
models, features, metrics = train_models(df)

naive_mae = naive_baseline(df)

# ===== PERFORMANCE =====
st.subheader("📊 Model Performance")

avg_mae = np.mean([m["mae"] for m in metrics.values()])
avg_r2 = np.mean([m["r2"] for m in metrics.values()])

col1, col2, col3 = st.columns(3)
col1.metric("Average MAE", round(avg_mae,2))
col2.metric("Average R²", round(avg_r2,4))
col3.metric("Naive MAE", round(naive_mae,2))

if avg_mae < naive_mae:
    st.success("Model Outperforms Naive ✅")
else:
    st.warning("Model Worse Than Naive ❌")

# ===== FORECAST =====
st.subheader("🔮 Forecast")

mode = st.radio("Mode", ["Single GPU", "All GPUs"])
months = st.slider("Forecast Months", 1, 24, 12)

if mode == "Single GPU":
    selected_gpu = st.selectbox("Select GPU", df["GPU"].unique())

if st.button("Run Forecast"):

    if mode == "Single GPU":

        result = forecast_gpu(
            models[selected_gpu],
            df,
            features,
            selected_gpu,
            months
        )

        st.dataframe(result)
        st.line_chart(result.set_index("Date")["Predicted_Price"])

    else:

        all_results = []

        for gpu in df["GPU"].unique():
            result = forecast_gpu(models[gpu], df, features, gpu, months)
            all_results.append(result)

        final_df = pd.concat(all_results)

        st.dataframe(final_df)

        pivot_df = final_df.pivot(index="Date", columns="GPU", values="Predicted_Price")
        st.line_chart(pivot_df)
