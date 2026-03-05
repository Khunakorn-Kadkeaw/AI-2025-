import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="GPU Price Forecast", layout="wide")

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv("gpu_price_history.csv")

    df = df.rename(columns={
        "Name":"GPU",
        "Retail Price":"Price"
    })

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["GPU","Date"])

    df["month_index"] = df.groupby("GPU").cumcount()

    # lag features
    for lag in [1,2,3]:
        df[f"lag_{lag}"] = df.groupby("GPU")["Price"].shift(lag)

    df["rolling_mean_3"] = (
        df.groupby("GPU")["Price"]
        .rolling(3).mean()
        .reset_index(level=0,drop=True)
    )

    df["rolling_std_3"] = (
        df.groupby("GPU")["Price"]
        .rolling(3).std()
        .reset_index(level=0,drop=True)
    )

    df["momentum"] = df["Price"] - df["lag_3"]

    df = df.dropna().reset_index(drop=True)

    return df


df = load_data()

# =====================================================
# MODEL TRAIN
# =====================================================

features = [
    "month_index",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
    "rolling_std_3",
    "momentum"
]


@st.cache_resource
def train_model(df, gpu):

    df_gpu = df[df["GPU"] == gpu]

    split = int(len(df_gpu)*0.8)

    train = df_gpu.iloc[:split]
    test = df_gpu.iloc[split:]

    X_train = train[features]
    y_train = train["Price"]

    X_test = test[features]
    y_test = test["Price"]

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8
    )

    model.fit(X_train,y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,pred)

    return model, mae


# =====================================================
# FORECAST
# =====================================================

def forecast_gpu(model, df, gpu, months):

    df_gpu = df[df["GPU"] == gpu].copy()

    df_future = df_gpu.copy()

    preds = []

    for i in range(months):

        last = df_future.iloc[-1:].copy()

        X = last[features]

        pred = model.predict(X)[0]

        preds.append(pred)

        new = last.copy()

        new["Price"] = pred

        new["month_index"] += 1

        new["lag_3"] = last["lag_2"].values[0]
        new["lag_2"] = last["lag_1"].values[0]
        new["lag_1"] = pred

        prices = df_future["Price"].tail(3).tolist()

        new["rolling_mean_3"] = np.mean(prices)
        new["rolling_std_3"] = np.std(prices)

        new["momentum"] = pred - prices[0]

        df_future = pd.concat([df_future,new],ignore_index=True)

    future_dates = pd.date_range(
        start=df_gpu["Date"].max() + pd.DateOffset(months=1),
        periods=months,
        freq="MS"
    )

    result = pd.DataFrame({
        "Date":future_dates,
        "Predicted_Price":preds
    })

    return result


# =====================================================
# NAIVE BASELINE
# =====================================================

def naive_baseline(df):

    rows=[]

    for gpu in df["GPU"].unique():

        d = df[df["GPU"]==gpu].copy()

        d["naive"]=d["Price"].shift(1)

        d=d.dropna()

        mae = mean_absolute_error(d["Price"],d["naive"])

        rows.append({
            "GPU":gpu,
            "MAE":round(mae,2)
        })

    return pd.DataFrame(rows)


# =====================================================
# UI
# =====================================================

st.title("🎮 GPU Price Forecast System")

# baseline
st.subheader("Baseline (Naive Forecast)")
st.dataframe(naive_baseline(df))


# ============================================
# USER INPUT
# ============================================

gpu_list = df["GPU"].unique()

selected_gpu = st.selectbox(
    "Select GPU",
    gpu_list
)

time_mode = st.radio(
    "Forecast Unit",
    ["Months","Years"]
)

value = st.slider("Forecast Horizon",1,36,12)

if time_mode == "Years":
    months = value*12
else:
    months = value


# ============================================
# RUN FORECAST
# ============================================

if st.button("Run Forecast"):

    with st.spinner("Training model..."):

        model, mae = train_model(df,selected_gpu)

    forecast = forecast_gpu(model,df,selected_gpu,months)

    hist = df[df["GPU"]==selected_gpu][["Date","Price"]]

    hist = hist.rename(columns={"Price":"Historical"})

    st.subheader("Model Performance")

    st.metric(
        "MAE",
        round(mae,2)
    )

    st.subheader("Forecast Result")

    st.dataframe(forecast)

    # combine chart
    hist_chart = hist.set_index("Date")

    forecast_chart = forecast.set_index("Date")

    chart_df = pd.concat(
        [hist_chart,forecast_chart],
        axis=1
    )

    st.line_chart(chart_df)
