import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="GPU Forecast System", layout="wide")

# =====================================================
# LOAD & FEATURE ENGINEERING
# =====================================================
@st.cache_data
def load_and_prepare_data(path):

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
# TRAIN MODEL
# =====================================================
@st.cache_resource
def train_model(df):

    le = LabelEncoder()
    df["GPU_Code"] = le.fit_transform(df["GPU"])

    features = [
        "GPU_Code",
        "month_index",
        "lag_1","lag_2","lag_3",
        "rolling_mean_3","rolling_std_3"
    ]

    X = df[features]
    y = df["Price"]

    model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=5,
        reg_alpha=2,
        random_state=42
    )

    model.fit(X, y)

    return model, le, features


# =====================================================
# MODEL EVALUATION (TimeSeries CV)
# =====================================================
@st.cache_resource
def evaluate_model(df, features):

    le = LabelEncoder()
    df["GPU_Code"] = le.fit_transform(df["GPU"])

    X = df[features]
    y = df["Price"]

    tscv = TimeSeriesSplit(n_splits=5)

    mae_list = []
    rmse_list = []
    r2_list = []

    for train_idx, test_idx in tscv.split(X):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=5,
            reg_alpha=2,
            random_state=42
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae_list.append(mean_absolute_error(y_test, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, preds)))
        r2_list.append(r2_score(y_test, preds))

    return np.mean(mae_list), np.mean(rmse_list), np.mean(r2_list)


# =====================================================
# NAIVE BASELINE
# =====================================================
def naive_baseline(df):

    df_sorted = df.sort_values(["GPU", "Date"])
    df_sorted["Naive_Pred"] = df_sorted.groupby("GPU")["Price"].shift(1)
    df_sorted = df_sorted.dropna()

    return mean_absolute_error(df_sorted["Price"], df_sorted["Naive_Pred"])


# =====================================================
# FORECAST FUNCTION
# =====================================================
def forecast_gpu(model, df, le, features, gpu_name, months):

    df_gpu = df[df["GPU"] == gpu_name].copy()
    df_gpu["GPU_Code"] = le.transform(df_gpu["GPU"])

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

    last_date = df_gpu["Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=months,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "GPU": gpu_name,
        "Date": future_dates,
        "Predicted_Price": predictions
    })

    return forecast_df


# =====================================================
# STREAMLIT UI
# =====================================================
st.title("🚀 GPU Price Forecast System")

df = load_and_prepare_data("gpu_price_history.csv")
model, le, features = train_model(df)

mae, rmse, r2 = evaluate_model(df.copy(), features)
naive_mae = naive_baseline(df.copy())

# ================= PERFORMANCE =================
st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", round(mae,2))
col2.metric("RMSE", round(rmse,2))
col3.metric("R²", round(r2,4))
col4.metric("Naive MAE", round(naive_mae,2))

if mae < naive_mae:
    st.success("Model Outperforms Naive Baseline ✅")
else:
    st.warning("Model Worse Than Naive ❌")

# ================= FORECAST =================
st.subheader("🔮 Forecast")

mode = st.radio("Forecast Mode", ["Single GPU", "All GPUs"])

months = st.slider("Forecast Months", 1, 36, 12)

if mode == "Single GPU":
    selected_gpu = st.selectbox("Select GPU", df["GPU"].unique())

if st.button("Run Forecast"):

    if mode == "Single GPU":

        result = forecast_gpu(model, df, le, features, selected_gpu, months)

        st.subheader("Forecast Result")
        st.dataframe(result)

        # Historical Fit
        df_gpu = df[df["GPU"] == selected_gpu].copy()
        df_gpu["GPU_Code"] = le.transform(df_gpu["GPU"])
        df_gpu["Model_Pred"] = model.predict(df_gpu[features])

        chart_df = df_gpu[["Date","Price","Model_Pred"]].set_index("Date")
        st.line_chart(chart_df)

        st.subheader("Forecast Chart")
        st.line_chart(result.set_index("Date")["Predicted_Price"])


    else:  # ALL GPUs

        all_results = []

        progress_bar = st.progress(0)
        gpu_list = df["GPU"].unique()

        for i, gpu in enumerate(gpu_list):
            result = forecast_gpu(model, df, le, features, gpu, months)
            all_results.append(result)

            progress_bar.progress((i + 1) / len(gpu_list))

        final_df = pd.concat(all_results).reset_index(drop=True)

        st.subheader("All GPU Forecast Result")
        st.dataframe(final_df)

        # Pivot for chart
        pivot_df = final_df.pivot(index="Date", columns="GPU", values="Predicted_Price")
        st.line_chart(pivot_df)

        # Download CSV
        csv = final_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            "gpu_forecast.csv",
            "text/csv"
        )
