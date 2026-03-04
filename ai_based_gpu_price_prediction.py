# ==========================================
# 1️⃣ Import Libraries
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 2️⃣ Load & Prepare Data
# ==========================================

# โหลดไฟล์ CSV ของคุณ
df = pd.read_csv("gpu_price_history.csv")

# ตั้งชื่อคอลัมน์ให้ชัดเจน
df = df.rename(columns={
    "Name": "GPU",
    "Date": "Date",
    "Retail Price": "Price"
})

# แปลงวันที่
df["Date"] = pd.to_datetime(df["Date"])

# เรียงตาม GPU และเวลา
df = df.sort_values(["GPU", "Date"])

# ==========================================
# 3️⃣ Feature Engineering (Lag & Rolling)
# ==========================================

df["month_index"] = df.groupby("GPU").cumcount()

df["lag_1"] = df.groupby("GPU")["Price"].shift(1)
df["lag_2"] = df.groupby("GPU")["Price"].shift(2)
df["rolling_3"] = df.groupby("GPU")["Price"].rolling(3).mean().reset_index(level=0, drop=True)

df = df.dropna()

# ==========================================
# 4️⃣ Encode GPU Name
# ==========================================

le = LabelEncoder()
df["GPU_Code"] = le.fit_transform(df["GPU"])

# ==========================================
# 5️⃣ Train/Test Split (NO Data Leakage)
# ==========================================

split_date = df["Date"].quantile(0.8)

train = df[df["Date"] <= split_date]
test = df[df["Date"] > split_date]

features = ["GPU_Code", "month_index", "lag_1", "lag_2", "rolling_3"]

X_train = train[features]
y_train = train["Price"]

X_test = test[features]
y_test = test["Price"]

# ==========================================
# 6️⃣ Train Model
# ==========================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# 7️⃣ Evaluate Model
# ==========================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Performance")
print("MAE :", round(mae, 2))
print("RMSE:", round(rmse, 2))

# ==========================================
# 8️⃣ Forecast Function (Production Version)
# ==========================================

def forecast_gpu(gpu_name, months_ahead=6):

    if gpu_name not in df["GPU"].unique():
        print("GPU not found in dataset.")
        return

    gpu_df = df[df["GPU"] == gpu_name].sort_values("Date")

    last_prices = list(gpu_df["Price"].tail(3))
    future_month = gpu_df["month_index"].max()

    predictions = []

    for i in range(months_ahead):

        future_month += 1

        X_future = pd.DataFrame([[
            le.transform([gpu_name])[0],
            future_month,
            last_prices[-1],
            last_prices[-2],
            np.mean(last_prices)
        ]], columns=features)

        pred_price = model.predict(X_future)[0]
        predictions.append(pred_price)

        # update rolling window
        last_prices.append(pred_price)
        last_prices.pop(0)

    # Plot
    plt.figure(figsize=(10,5))

    plt.plot(gpu_df["Date"], gpu_df["Price"], label="Historical Price")

    future_dates = pd.date_range(
        start=gpu_df["Date"].max(),
        periods=months_ahead+1,
        freq="M"
    )[1:]

    plt.plot(future_dates, predictions, linestyle="--", marker="o", label="Forecast")

    plt.title(f"Price Forecast for {gpu_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

    return predictions

# ==========================================
# 9️⃣ Example Usage
# ==========================================

forecast_gpu("GeForce RTX 3060", 6)
