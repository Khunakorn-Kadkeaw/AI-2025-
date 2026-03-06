# =========================================
# INSTALL LIBRARY
# =========================================
!pip install prophet tqdm

# =========================================
# IMPORT LIBRARY
# =========================================
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from tqdm import tqdm

# =========================================
# LOAD DATA
# =========================================
df = pd.read_csv("gpu_price.csv")

# =========================================
# RENAME COLUMN
# =========================================
df = df.rename(columns={
    "Name":"GPU",
    "Retail Price":"Price"
})

# =========================================
# BASIC CLEANING
# =========================================

# แปลงข้อมูลในคอลัมน์ Date ให้เป็นชนิด datetime
# เพื่อให้ pandas เข้าใจว่าเป็นข้อมูลเวลา
# และสามารถนำไปใช้กับงาน Time Series เช่น การเรียงลำดับเวลา
# การ resample เดือน หรือการ forecast ใน Prophet ได้
df["Date"] = pd.to_datetime(df["Date"])

# ลบแถวที่มีค่าข้อมูลหาย (Missing Value หรือ NaN)
# เช่น GPU ไม่มีราคา หรือไม่มีวันที่
# เพราะโมเดล Machine Learning และ Time Series ส่วนใหญ่ไม่สามารถทำงานกับค่า NaN ได้
df = df.dropna()

# ลบข้อมูลที่ราคา (Price) น้อยกว่าหรือเท่ากับ 0
# เนื่องจากราคา GPU จริงไม่ควรเป็น 0
# และในขั้นตอนต่อไปมีการใช้ log(price)
# ถ้า price = 0 จะทำให้เกิดค่า -inf และทำให้โมเดลทำงานผิดพลาดได้
df = df[df["Price"] > 0]

# ลบข้อมูลที่ซ้ำกัน โดยพิจารณาจาก Date และ GPU
# หมายความว่า ใน 1 GPU และ 1 วันที่กำหนด ควรมีข้อมูลราคาเพียง 1 ค่าเท่านั้น
# หากมีข้อมูลซ้ำ จะทำให้การวิเคราะห์ Time Series ผิดพลาด
df = df.drop_duplicates(subset=["Date","GPU"])

# เรียงลำดับข้อมูลตามชื่อ GPU ก่อน และเรียงตามวันที่หลัง
# เพื่อให้ข้อมูลอยู่ในลำดับเวลา (Time Order)
# ซึ่งสำคัญสำหรับการสร้างโมเดล Time Series
df = df.sort_values(["GPU","Date"])

# รีเซ็ต index ของ DataFrame ใหม่
# หลังจากลบข้อมูลและเรียงข้อมูล index เดิมอาจไม่ต่อเนื่อง
# การ reset index ทำให้ index เริ่มใหม่จาก 0 และเรียงลำดับถูกต้อง
df = df.reset_index(drop=True)

# แสดงขนาดของข้อมูลหลังจากทำความสะอาด
# df.shape จะแสดง (จำนวนแถว , จำนวนคอลัมน์)
# เพื่อใช้ตรวจสอบว่าหลัง cleaning แล้วข้อมูลเหลือเท่าไหร่
print("Clean data:",df.shape)

# =========================================
# REMOVE OUTLIER (IQR)
# =========================================

# สร้างฟังก์ชันสำหรับลบค่าผิดปกติ (Outlier)
# โดยใช้วิธี IQR (Interquartile Range)
def remove_outlier(group):

    # หา Quartile ที่ 1 (25%)
    # คือค่าที่แบ่งข้อมูล 25% แรก
    Q1 = group["Price"].quantile(0.25)

    # หา Quartile ที่ 3 (75%)
    # คือค่าที่แบ่งข้อมูล 75%
    Q3 = group["Price"].quantile(0.75)

    # คำนวณ IQR (Interquartile Range)
    # IQR = Q3 - Q1
    # เป็นช่วงของข้อมูลตรงกลาง 50%
    IQR = Q3 - Q1

    # คำนวณขอบล่างของข้อมูลที่ยอมรับได้
    # ค่าที่ต่ำกว่านี้จะถือว่าเป็น Outlier
    lower = Q1 - 1.5 * IQR

    # คำนวณขอบบนของข้อมูลที่ยอมรับได้
    # ค่าที่สูงกว่านี้จะถือว่าเป็น Outlier
    upper = Q3 + 1.5 * IQR

    # เลือกเฉพาะข้อมูลที่อยู่ในช่วง lower และ upper
    # เพื่อกรองค่าที่ผิดปกติออก
    return group[(group["Price"] >= lower) & (group["Price"] <= upper)]

# ใช้ groupby เพื่อแยกข้อมูลตาม GPU แต่ละรุ่น
# เพราะราคาของ GPU แต่ละรุ่นต่างกันมาก
# ถ้าไม่แยก GPU จะทำให้ลบข้อมูลผิด
df = df.groupby("GPU", group_keys=False).apply(remove_outlier)

# แสดงขนาดข้อมูลหลังจากลบ outlier
# ใช้เพื่อตรวจสอบว่ามีข้อมูลถูกลบออกไปเท่าไร
print("After outlier remove:", df.shape)

# =========================================
# RESAMPLE MONTHLY
# =========================================

# สร้าง list ว่างไว้เก็บ dataframe ของแต่ละ GPU
new_df = []

# วนลูป GPU ทีละรุ่น
# df["GPU"].unique() = ดึงชื่อ GPU ที่ไม่ซ้ำกัน เช่น RTX 3060, RTX 3070
for gpu in df["GPU"].unique():

    # เลือกข้อมูลเฉพาะ GPU รุ่นนั้น
    # เช่น ถ้า gpu = RTX 3060 ก็จะได้เฉพาะข้อมูล RTX 3060
    temp = df[df["GPU"]==gpu].copy()

    # ตั้ง column Date เป็น index
    # เพราะการ resample ต้องใช้ datetime index
    temp = temp.set_index("Date")

    # แปลงข้อมูลให้เป็นรายเดือน
    # "MS" = Month Start (ต้นเดือน)
    # ถ้าในเดือนนั้นมีหลายค่า จะเอา mean
    temp = temp[["Price"]].resample("MS").mean()

    # เติมค่าที่หายไป (missing month)
    # interpolate = ใช้ค่ากลางระหว่างเดือนก่อนหน้าและเดือนถัดไป
    # limit=3 = เติมได้สูงสุด 3 เดือนติดกัน
    temp["Price"] = temp["Price"].interpolate(limit=3)

    # ใส่ชื่อ GPU กลับเข้าไป
    # เพราะตอน resample เหลือแค่ Price
    temp["GPU"] = gpu

    # reset index เพื่อให้ Date กลับมาเป็น column
    # แล้วเอา dataframe นี้ไปเก็บใน list
    new_df.append(temp.reset_index())

# รวม dataframe ของทุก GPU เข้าด้วยกัน
df = pd.concat(new_df)

# reset index ใหม่
df = df.reset_index(drop=True)

# แสดงขนาดข้อมูลหลัง resample
print("After resample:",df.shape)

# =========================================
# FILTER GPU WITH LOW DATA
# =========================================

# นับจำนวนข้อมูลของ GPU แต่ละรุ่น
# value_counts() จะนับว่าชื่อ GPU แต่ละตัวปรากฏกี่ครั้งใน column "GPU"
# ตัวอย่างผลลัพธ์ เช่น
# RTX 3060      36
# RTX 3070      30
# GTX 1050       5
gpu_count = df["GPU"].value_counts()

# เลือกเฉพาะ GPU ที่มีข้อมูล >= 12 แถว
# หมายถึง GPU รุ่นนั้นต้องมีข้อมูลอย่างน้อย 12 เดือน
# gpu_count >= 12 จะสร้างเงื่อนไข filter
# .index คือการดึงชื่อ GPU ที่ผ่านเงื่อนไขออกมา
valid_gpu = gpu_count[gpu_count >= 12].index

# กรอง dataframe df ใหม่
# โดยเลือกเฉพาะ GPU ที่อยู่ใน valid_gpu
# isin() ใช้เช็คว่า GPU ในแต่ละแถวอยู่ใน list valid_gpu หรือไม่
df = df[df["GPU"].isin(valid_gpu)]

# แสดงจำนวน GPU ที่เหลืออยู่หลังจากกรองข้อมูล
print("GPU used:",len(valid_gpu))

# บันทึกจำนวนข้อมูลของ GPU แต่ละรุ่นลงไฟล์ CSV
# เพื่อนำไปดูว่ามี GPU รุ่นไหนมีข้อมูลมากหรือน้อย
# ไฟล์นี้จะถูกใช้สำหรับวิเคราะห์คุณภาพ dataset
gpu_count.to_csv("gpu_data_count.csv")

# =========================================
# LOG TRANSFORM
# =========================================

# ใช้ฟังก์ชัน log จาก numpy เพื่อแปลงค่า Price
# การทำ Log Transform คือการแปลงค่าราคาให้เป็น log(price)

# ทำงานโดย:
# นำค่าราคาทุกตัวใน column "Price" มาเข้าสมการ
# log(price)

# ตัวอย่าง
# ราคา 10000 → log ≈ 9.21
# ราคา 5000  → log ≈ 8.51

# เหตุผลที่ต้องทำ:
# 1. ลดความต่างของช่วงราคา GPU ที่ต่างกันมาก
#    เช่น GPU ราคา 3000 กับ 80000 จะต่างกันมากเกินไป

# 2. ลดผลกระทบของ outlier
#    เช่น RTX 4090 ที่ราคาแพงมาก

# 3. ทำให้ข้อมูลมี distribution ที่เหมาะกับโมเดลมากขึ้น
#    โดยเฉพาะโมเดล Time Series อย่าง Prophet

# 4. ทำให้โมเดลเรียนรู้ trend ได้ดีขึ้น

# หมายเหตุสำคัญ
# หลังจากโมเดลทำนายเสร็จ ต้องแปลงค่ากลับด้วย
# np.exp() เพื่อให้กลับมาเป็น "ราคาจริง"

df["Price"] = np.log(df["Price"])

# =========================================
# TRAIN PROPHET
# =========================================
def train_prophet(df_gpu):

    # เปลี่ยนชื่อ column ให้ตรงกับ format ที่ Prophet ต้องใช้
    # Prophet ต้องการ
    # ds = วันที่ (datetime)
    # y  = ค่าที่ต้องการทำนาย (target variable)
    data = df_gpu.rename(columns={
        "Date":"ds",     # เปลี่ยน Date -> ds
        "Price":"y"      # เปลี่ยน Price -> y
    })

    # สร้างโมเดล Prophet
    model = Prophet(

        # ปิด seasonality ทั้งหมด
        # เพราะข้อมูลราคาการ์ดจอส่วนใหญ่ไม่ได้มี pattern รายปี / รายสัปดาห์ชัดเจน
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,

        # ควบคุมความยืดหยุ่นของ trend
        # ค่าใหญ่ = model ปรับ trend ได้เร็ว
        # ค่าเล็ก = model smooth มากขึ้น
        # 0.3 เหมาะกับข้อมูลราคาที่มีการเปลี่ยน trend บ้าง
        changepoint_prior_scale=0.3
    )

    # train model ด้วยข้อมูล GPU รุ่นนั้น
    model.fit(data)

    # ส่ง model กลับไปใช้ forecast ต่อ
    return model

# =========================================
# FUNCTION: FORECAST GPU PRICE
# =========================================

# ฟังก์ชันนี้ใช้สำหรับสร้างการพยากรณ์ราคาการ์ดจอ (GPU)
# โดยใช้โมเดล Prophet ที่ train มาแล้ว

def forecast_gpu(model, months=12):

    # สร้าง DataFrame ของวันที่ในอนาคต
    # periods = จำนวนเดือนที่ต้องการพยากรณ์
    # freq="MS" = ความถี่แบบ Month Start (ต้นเดือน)
    future = model.make_future_dataframe(
        periods=months,
        freq="MS"
    )

    # ใช้โมเดล Prophet ทำการพยากรณ์ค่าราคาในอนาคต
    forecast = model.predict(future)

    # เลือกเฉพาะคอลัมน์ที่ต้องการใช้
    # ds = วันที่
    # yhat = ค่าที่โมเดลพยากรณ์
    # yhat_lower = ขอบล่างของช่วงความเชื่อมั่น
    # yhat_upper = ขอบบนของช่วงความเชื่อมั่น
    forecast = forecast[["ds","yhat","yhat_lower","yhat_upper"]]

    # ป้องกันค่าราคาเป็นลบ
    # ถ้าโมเดลทำนายค่า < 0 จะถูกปรับให้เป็น 0
    forecast["yhat"] = forecast["yhat"].clip(lower=0)

    # คืนค่าผลลัพธ์การพยากรณ์
    return forecast

# =========================================
# FORECAST ALL GPU
# =========================================

# สร้าง list ว่างเพื่อเก็บผล forecast ของ GPU ทุกตัว
results = []

# วนลูป GPU แต่ละรุ่น
# df["GPU"].unique() = ดึงชื่อ GPU ที่ไม่ซ้ำกัน
# tqdm() = ใช้แสดง progress bar ตอนรันโค้ด
for gpu in tqdm(df["GPU"].unique()):

    # สร้าง dataframe เฉพาะข้อมูลของ GPU รุ่นนั้น
    df_gpu = df[df["GPU"]==gpu]

    # ถ้าข้อมูลน้อยกว่า 12 เดือน จะไม่ทำการ forecast
    # เพราะข้อมูลไม่พอสำหรับ time series model
    if len(df_gpu) < 12:
        continue

    try:

        # train โมเดล Prophet ด้วยข้อมูลของ GPU รุ่นนั้น
        model = train_prophet(df_gpu)

        # ทำการ forecast ล่วงหน้า 12 เดือน
        forecast = forecast_gpu(model,12)

        # เพิ่ม column GPU เพื่อระบุว่า forecast นี้เป็นของ GPU รุ่นไหน
        forecast["GPU"] = gpu

        # เก็บผล forecast ลงใน list results
        results.append(forecast)

    # ถ้ามี error เกิดขึ้น (เช่น model train ไม่ได้)
    # จะข้าม GPU ตัวนั้นไป
    except:
        continue

# ถ้าไม่มี GPU ที่สามารถ forecast ได้เลย
if len(results) == 0:

    print("No GPU available for forecast")

# ถ้ามีผล forecast
else:

    # รวม dataframe ของ GPU ทุกตัวเข้าด้วยกัน
    final_forecast = pd.concat(results)

    # บันทึกผล forecast ลงไฟล์ CSV
    final_forecast.to_csv("gpu_forecast.csv",index=False)

    # แสดงข้อความเมื่อบันทึกเสร็จ
    print("Forecast saved")

# =========================================
# MODEL ACCURACY TEST
# =========================================

# สร้าง list ว่างเพื่อเก็บผล accuracy ของแต่ละ GPU
accuracy_results = []

# วนลูป GPU แต่ละรุ่น
# df["GPU"].unique() = ดึงชื่อ GPU ที่ไม่ซ้ำกัน
# tqdm() = แสดง progress bar ตอนรัน
for gpu in tqdm(df["GPU"].unique()):

    # สร้าง dataframe เฉพาะข้อมูลของ GPU รุ่นนั้น
    df_gpu = df[df["GPU"] == gpu]

    # ถ้ามีข้อมูลน้อยกว่า 12 เดือน จะไม่ใช้ประเมิน model
    # เพราะข้อมูล time series น้อยเกินไป
    if len(df_gpu) < 12:
        print("Skip:",gpu,"data too small")
        continue

    # ---------------------------------
    # SPLIT TRAIN / TEST
    # ---------------------------------

    # ใช้ข้อมูลทั้งหมด ยกเว้น 3 เดือนสุดท้าย เป็น train
    train = df_gpu.iloc[:-3]

    # ใช้ 3 เดือนสุดท้าย เป็น test
    # เพื่อดูว่า model ทำนายใกล้เคียงจริงไหม
    test = df_gpu.iloc[-3:]

    try:

        # train prophet model ด้วยข้อมูล train
        model = train_prophet(train)

        # สร้าง future dataframe เพิ่มอีก 3 เดือน
        future = model.make_future_dataframe(periods=3,freq="MS")

        # ให้ model ทำนายค่า
        forecast = model.predict(future)

        # ดึงเฉพาะค่าทำนาย 3 เดือนสุดท้าย
        pred = forecast.tail(3)["yhat"]

        # ป้องกันค่าติดลบ (ราคา GPU ไม่ควรติดลบ)
        pred = pred.clip(lower=0)

        # แปลงค่ากลับจาก log → ราคาเดิม
        pred = np.exp(pred)

        # แปลง test data จาก log → ราคาเดิม
        test_price = np.exp(test["Price"])

        # รวมค่าจริง (true) และค่าที่ model ทำนาย (pred)
        temp_eval = pd.DataFrame({
            "true":test_price.values,
            "pred":pred.values
        }).dropna()   # ลบค่า NaN

        # ถ้าไม่มีข้อมูลเหลือให้ใช้ประเมิน
        if len(temp_eval) == 0:
            print("Skip:",gpu,"no eval data")
            continue

        # ---------------------------------
        # CALCULATE ERROR
        # ---------------------------------

        # MAE = Mean Absolute Error
        # ค่าความคลาดเคลื่อนเฉลี่ยของราคา
        mae = mean_absolute_error(temp_eval["true"],temp_eval["pred"])

        # MAPE = Mean Absolute Percentage Error
        # ค่าความคลาดเคลื่อนเป็น %
        mape = mean_absolute_percentage_error(temp_eval["true"],temp_eval["pred"])

        # เก็บผลลัพธ์ของ GPU รุ่นนี้ลง list
        accuracy_results.append({
            "GPU":gpu,
            "MAE":round(mae,2),
            "MAPE(%)":round(mape*100,2)
        })

    # ถ้า model มี error จะข้าม GPU ตัวนั้น
    except Exception as e:
        print("Model error:",gpu,e)

# =========================================
# SAVE ACCURACY
# =========================================
# สร้าง DataFrame จาก list accuracy_results
# accuracy_results เก็บผลลัพธ์ MAE และ MAPE ของแต่ละ GPU
acc_df = pd.DataFrame(accuracy_results)

# แสดงจำนวนแถวของผลลัพธ์ accuracy
# ใช้ตรวจสอบว่า model ประเมินผลได้กี่ GPU
print("Accuracy rows:",len(acc_df))

# ตรวจสอบว่ามีผลลัพธ์ accuracy หรือไม่
# ถ้า DataFrame มีข้อมูลมากกว่า 0 แถว
if len(acc_df) > 0:

    # เรียงลำดับ GPU ตามค่า MAPE จากน้อย → มาก
    # MAPE น้อย = model ทำนายแม่นกว่า
    acc_df = acc_df.sort_values("MAPE(%)")

    # บันทึกผลลัพธ์ accuracy ลงไฟล์ csv
    # index=False เพื่อไม่ให้ pandas บันทึก index column
    acc_df.to_csv("gpu_accuracy.csv",index=False)

    # แจ้งว่าไฟล์ถูกสร้างแล้ว
    print("gpu_accuracy.csv saved")

    # แสดงข้อความหัวข้อ
    print("\nTop 10 Model")

    # แสดง GPU ที่ model ทำนายได้แม่นที่สุด 10 อันดับแรก
    # head(10) = แสดง 10 แถวแรกของ DataFrame
    print(acc_df.head(10))

# ถ้าไม่มีข้อมูล accuracy เลย
else:

    # แสดงข้อความแจ้งเตือน
    # แปลว่าไม่มี GPU ที่สามารถประเมิน model ได้
    print("No accuracy result generated")
