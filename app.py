import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ===============================
# LOAD & PREP
# ===============================
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

# ===============================
# TRAIN MODEL
# ===============================
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

# ===============================
# FORECAST FUNCTION
# ===============================
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


# ===============================
# STREAMLIT UI
# ===============================
st.title("GPU Price Forecast System")

df = load_and_prepare_data("gpu_price_history.csv")
model, le, features = train_model(df)

gpu_list = df["GPU"].unique()
selected_gpu = st.selectbox("Select GPU", gpu_list)
months = st.slider("Forecast Months", 1, 36, 12)

if st.button("Run Forecast"):

    result = forecast_gpu(model, df, le, features, selected_gpu, months)

    st.subheader("Forecast Result")
    st.dataframe(result)

    st.line_chart(result.set_index("Date")["Predicted_Price"])
