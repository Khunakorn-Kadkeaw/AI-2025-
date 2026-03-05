import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="GPU Price Forecast AI", layout="wide")

# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():
    forecast = pd.read_csv("gpu_forecast (1).csv")
    data_count = pd.read_csv("gpu_data_count.csv")
    accuracy = pd.read_csv("gpu_accuracy.csv")

    forecast["ds"] = pd.to_datetime(forecast["ds"])

    return forecast, data_count, accuracy


forecast_df, count_df, acc_df = load_data()

# ===============================
# TITLE
# ===============================

st.title("GPU Price Forecast System (AI)")

# ===============================
# GPU DATA COUNT
# ===============================

st.subheader("จำนวนข้อมูลของแต่ละ GPU")

st.dataframe(count_df)

# ===============================
# MODEL ACCURACY
# ===============================

st.subheader("Model Accuracy")

st.dataframe(acc_df)

# ===============================
# FORECAST SECTION
# ===============================

st.subheader("GPU Price Forecast")

gpu_list = forecast_df["GPU"].unique()

mode = st.radio(
    "เลือกโหมดการทำนาย",
    ["ทำนาย GPU รุ่นเดียว", "ทำนายทุก GPU"]
)

months = st.slider(
    "ต้องการดูอนาคตกี่เดือน",
    1,
    24,
    12
)

# ===============================
# SINGLE GPU
# ===============================

if mode == "ทำนาย GPU รุ่นเดียว":

    selected_gpu = st.selectbox(
        "เลือกรุ่น GPU",
        gpu_list
    )

    df_gpu = forecast_df[forecast_df["GPU"] == selected_gpu]

    df_future = df_gpu.tail(months)

    st.subheader("Forecast Data")

    st.dataframe(df_future)

    # GRAPH
    fig, ax = plt.subplots()

    ax.plot(df_gpu["ds"], df_gpu["yhat"])

    ax.set_title(selected_gpu)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    st.pyplot(fig)

# ===============================
# ALL GPU
# ===============================

else:

    df_future = forecast_df.groupby("GPU").tail(months)

    st.subheader("Forecast Data (All GPU)")

    st.dataframe(df_future)

    pivot = df_future.pivot(
        index="ds",
        columns="GPU",
        values="yhat"
    )

    st.line_chart(pivot)
