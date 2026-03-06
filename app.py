# import library ที่ใช้สร้าง Web App
import streamlit as st        # ใช้สร้างหน้าเว็บ
import pandas as pd           # ใช้จัดการข้อมูลตาราง
import matplotlib.pyplot as plt  # ใช้วาดกราฟ

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="GPU Price Forecast AI",  # ชื่อ tab บน browser
    layout="wide"                        # ใช้หน้าจอแบบเต็ม
)

# ===============================
# LOAD DATA
# ===============================

# cache_data = ให้ Streamlit โหลดข้อมูลครั้งเดียว
# รอบต่อไปจะใช้ cache ทำให้เว็บเร็วขึ้น
@st.cache_data
def load_data():

    # โหลดไฟล์ forecast ที่ model ทำนายไว้
    forecast = pd.read_csv("gpu_forecast.csv")

    # โหลดจำนวนข้อมูลของ GPU แต่ละรุ่น
    data_count = pd.read_csv("gpu_data_count.csv")

    # โหลดผล accuracy ของ model
    accuracy = pd.read_csv("gpu_accuracy.csv")

    # แปลง column ds ให้เป็น datetime
    forecast["ds"] = pd.to_datetime(forecast["ds"])

    # return ข้อมูลทั้ง 3 ตาราง
    return forecast, data_count, accuracy


# เรียกใช้ function load_data()
forecast_df, count_df, acc_df = load_data()

# ===============================
# TITLE
# ===============================

# แสดง title ของเว็บ
st.title("GPU Price Forecast System (AI)")

# ===============================
# GPU DATA COUNT
# ===============================

# แสดงหัวข้อ
st.subheader("จำนวนข้อมูลของแต่ละ GPU")

# แสดง dataframe จำนวนข้อมูล GPU
# เช่น RTX3060 มีข้อมูลกี่เดือน
st.dataframe(count_df)

# ===============================
# MODEL ACCURACY
# ===============================

# แสดงหัวข้อ accuracy model
st.subheader("Model Accuracy")

# แสดงตาราง MAE / MAPE ของแต่ละ GPU
st.dataframe(acc_df)

# ===============================
# FORECAST SECTION
# ===============================

# ส่วนแสดงผลการทำนายราคา
st.subheader("GPU Price Forecast")

# ดึง list GPU จาก forecast dataframe
gpu_list = forecast_df["GPU"].unique()

# สร้าง radio button ให้ผู้ใช้เลือกโหมด
mode = st.radio(
    "เลือกโหมดการทำนาย",
    ["ทำนาย GPU รุ่นเดียว", "ทำนายทุก GPU"]
)

# slider ให้ผู้ใช้เลือกจำนวนเดือนในอนาคต
months = st.slider(
    "ต้องการดูอนาคตกี่เดือน",
    1,     # ต่ำสุด
    12,    # สูงสุด
    12     # ค่า default
)

# ===============================
# SINGLE GPU
# ===============================

# ถ้าเลือกโหมด GPU รุ่นเดียว
if mode == "ทำนาย GPU รุ่นเดียว":

    # dropdown ให้เลือก GPU รุ่น
    selected_gpu = st.selectbox(
        "เลือกรุ่น GPU",
        gpu_list
    )

    # filter dataframe เฉพาะ GPU ที่เลือก
    df_gpu = forecast_df[forecast_df["GPU"] == selected_gpu]

    # เอาเฉพาะข้อมูลอนาคตตามจำนวนเดือนที่ slider เลือก
    df_future = df_gpu.tail(months)

    # แสดงหัวข้อ
    st.subheader("Forecast Data")

    # แสดงตารางผลการทำนาย
    st.dataframe(df_future)

    # =====================
    # GRAPH
    # =====================

    # สร้าง figure และ axis สำหรับ plot
    fig, ax = plt.subplots()

    # plot ราคา GPU ตามเวลา
    ax.plot(df_gpu["ds"], df_gpu["yhat"])

    # ตั้งชื่อกราฟ
    ax.set_title(selected_gpu)

    # ตั้งชื่อแกน X
    ax.set_xlabel("Date")

    # ตั้งชื่อแกน Y
    ax.set_ylabel("Price")

    # แสดงกราฟบน Streamlit
    st.pyplot(fig)

# ===============================
# ALL GPU
# ===============================

# ถ้าเลือกโหมดดูทุก GPU
else:

    # ดึงข้อมูลอนาคตของ GPU ทุกตัว
    df_future = forecast_df.groupby("GPU").tail(months)

    # แสดงหัวข้อ
    st.subheader("Forecast Data (All GPU)")

    # แสดงตารางข้อมูล
    st.dataframe(df_future)

    # pivot table เพื่อแปลงข้อมูล
    # ให้แต่ละ GPU เป็น column
    pivot = df_future.pivot(
        index="ds",     # วันที่
        columns="GPU",  # GPU แต่ละรุ่น
        values="yhat"   # ราคาที่ทำนาย
    )

    # แสดงกราฟ line chart ของทุก GPU
    st.line_chart(pivot)
