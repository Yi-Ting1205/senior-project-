# import os
# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model

# # 禁用 TensorFlow 的 GPU 使用
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# app = Flask(__name__)

# # 載入模型
# try:
#     model = load_model("gait_model_5.h5")
# except Exception as e:
#     print(f"模型加載失敗: {str(e)}")
#     model = None

# # 正常參考區間（來自批次結果）
# STANCE_LOWER, STANCE_UPPER = 0.5004, 0.7169
# SWING_LOWER, SWING_UPPER = 0.2831, 0.4996
# GAIT_CYCLE_LOWER, GAIT_CYCLE_UPPER = 0.9424, 1.3673
# CADENCE_LOWER, CADENCE_UPPER = 42.6880, 62.1078
# STANCE_VAR_LOWER, STANCE_VAR_UPPER = 0.0045, 0.1005
# SWING_VAR_LOWER, SWING_VAR_UPPER = 0.0045, 0.1005

# def generate_comment(params):
#     """根據步態參數生成評語"""
#     stance = params.get("Stance Phase", 0)
#     swing = params.get("Swing Phase", 0)
#     gait_cycle = params.get("Gait Cycle Time (s)", 0)
#     cadence = params.get("Cadence (steps/min)", 0)
#     stance_var = params.get("Stance Variability", 0)
#     swing_var = params.get("Swing Variability", 0)

#     normal = (
#         STANCE_LOWER <= stance <= STANCE_UPPER and
#         SWING_LOWER <= swing <= SWING_UPPER and
#         GAIT_CYCLE_LOWER <= gait_cycle <= GAIT_CYCLE_UPPER and
#         CADENCE_LOWER <= cadence <= CADENCE_UPPER and
#         STANCE_VAR_LOWER <= stance_var <= STANCE_VAR_UPPER and
#         SWING_VAR_LOWER <= swing_var <= SWING_VAR_UPPER
#     )

#     mild_abnormal = (
#         (STANCE_LOWER - 0.025 <= stance < STANCE_LOWER or STANCE_UPPER < stance <= STANCE_UPPER + 0.025) or
#         (SWING_LOWER - 0.025 <= swing < SWING_LOWER or SWING_UPPER < swing <= SWING_UPPER + 0.025) or
#         (GAIT_CYCLE_LOWER - 0.1 <= gait_cycle < GAIT_CYCLE_LOWER or GAIT_CYCLE_UPPER < gait_cycle <= GAIT_CYCLE_UPPER + 0.1) or
#         (CADENCE_LOWER - 5 <= cadence < CADENCE_LOWER or CADENCE_UPPER < cadence <= CADENCE_UPPER + 5) or
#         (STANCE_VAR_UPPER < stance_var <= STANCE_VAR_UPPER + 0.01) or
#         (SWING_VAR_UPPER < swing_var <= SWING_VAR_UPPER + 0.01)
#     )

#     if normal:
#         return "步態正常：參數落在正常範圍內，步態穩定協調。"
#     elif mild_abnormal:
#         comments = ["步態輕微異常："]
#         if stance < STANCE_LOWER:
#             comments.append("站立期比例偏低。")
#         elif stance > STANCE_UPPER:
#             comments.append("站立期比例偏高。")
#         if swing < SWING_LOWER:
#             comments.append("擺動期比例偏低。")
#         elif swing > SWING_UPPER:
#             comments.append("擺動期比例偏高。")
#         if stance_var > STANCE_VAR_UPPER:
#             comments.append("站立期變異性偏高。")
#         if swing_var > SWING_VAR_UPPER:
#             comments.append("擺動期變異性偏高。")
#         if cadence < CADENCE_LOWER:
#             comments.append("步頻偏低。")
#         elif cadence > CADENCE_UPPER:
#             comments.append("步頻偏高。")
#         return "\n".join(comments)
#     else:
#         comments = ["步態顯著異常："]
#         if stance < STANCE_LOWER - 0.025 or stance > STANCE_UPPER + 0.025:
#             comments.append("站立期顯著異常，建議進一步評估。")
#         if swing < SWING_LOWER - 0.025 or swing > SWING_UPPER + 0.025:
#             comments.append("擺動期顯著異常，建議進一步評估。")
#         if stance_var > STANCE_VAR_UPPER + 0.01 or swing_var > SWING_VAR_UPPER + 0.01:
#             comments.append("步態變異性過高，增加跌倒風險。")
#         if cadence < CADENCE_LOWER - 5 or cadence > CADENCE_UPPER + 5:
#             comments.append("步頻顯著異常。")
#         return "\n".join(comments)

# def predict_events(model, test_time, test_gyro, window_size=60, distance=40):
#     """使用模型預測 HS 和 TO 事件"""
#     if model is None:
#         return []
        
#     x_pred = []
#     pred_events = []

#     # 準備滑動窗口數據
#     for i in range(window_size, len(test_gyro) - window_size):
#         window = test_gyro[i - window_size : i + window_size + 1].reshape(-1, 1) 
#         x_pred.append(window)

#     x_pred = np.array(x_pred)
#     y_pred = model.predict(x_pred, verbose=0)
#     pred_labels = np.argmax(y_pred, axis=1)
    
#     # 事件檢測參數
#     last_event_idx = {"HS": -distance, "TO": -distance} 
#     hs_distance_threshold = 30  # HS事件最小間隔
#     to_indices = []  # 儲存TO事件索引
#     in_to_segment = False

#     # 檢測TO事件 (模型標籤為0)
#     for i in range(1, len(pred_labels)):
#         if pred_labels[i] == 0 and not in_to_segment:
#             start = i
#             in_to_segment = True
#         elif pred_labels[i] != 0 and in_to_segment:
#             end = i - 1
#             if end - start >= 5:  # 最小持續時間
#                 seg = test_gyro[start + window_size : end + window_size + 1]
#                 local_min_idx = np.argmin(seg)  # 找局部最小值
#                 global_idx = start + window_size + local_min_idx
#                 if global_idx - last_event_idx["TO"] >= distance:
#                     event_time = test_time[global_idx]
#                     pred_events.append((event_time, "TO"))
#                     to_indices.append(global_idx)
#                     last_event_idx["TO"] = global_idx
#             in_to_segment = False

#     # 處理最後一個可能未結束的TO段
#     if in_to_segment:
#         end = len(pred_labels) - 1
#         if end - start >= 5:
#             seg = test_gyro[start + window_size : end + window_size + 1]
#             local_min_idx = np.argmin(seg)
#             global_idx = start + window_size + local_min_idx
#             if global_idx - last_event_idx["TO"] >= distance:
#                 event_time = test_time[global_idx]
#                 pred_events.append((event_time, "TO"))
#                 to_indices.append(global_idx)
#                 last_event_idx["TO"] = global_idx

#     # 檢測HS事件 (在TO事件之間找局部最大值)
#     last_hs_idx = -distance
#     for i in range(len(to_indices) - 1):
#         start_idx = to_indices[i]
#         end_idx = to_indices[i+1]
#         if end_idx - start_idx <= 5:  # 忽略過短區間
#             continue
#         seg = test_gyro[start_idx:end_idx+1]
#         local_max_idx = np.argmax(seg)  # 找局部最大值
#         hs_global_idx = start_idx + local_max_idx
#         if hs_global_idx - last_hs_idx >= hs_distance_threshold:
#             event_time = test_time[hs_global_idx]
#             pred_events.append((event_time, "HS"))
#             last_hs_idx = hs_global_idx
    
#     return pred_events

# def calculate_gait_phases(pred_events):
#     """直接從預測事件計算站立期和擺動期"""
#     try:
#         # 分離HS和TO事件
#         hs_times = [t for t, e in pred_events if e == "HS"]
#         to_times = [t for t, e in pred_events if e == "TO"]

#         if len(hs_times) < 2 or len(to_times) < 1:
#             return None

#         stance_phases = []
#         swing_phases = []

#         # 計算每個步態周期的相位
#         for i in range(len(hs_times) - 1):
#             hs1 = hs_times[i]
#             hs2 = hs_times[i + 1]
            
#             # 找出當前周期內的TO事件
#             cycle_to = [to for to in to_times if hs1 < to < hs2]
#             if not cycle_to:
#                 continue
                
#             to = cycle_to[0]  # 取第一個TO事件
#             gait_cycle = hs2 - hs1
#             stance_duration = to - hs1
#             swing_duration = hs2 - to

#             if gait_cycle <= 0:
#                 continue

#             stance_phases.append(stance_duration / gait_cycle)
#             swing_phases.append(swing_duration / gait_cycle)

#         if not stance_phases or not swing_phases:
#             return None

#         # 計算平均比例
#         avg_stance = np.mean(stance_phases)
#         avg_swing = np.mean(swing_phases)
        
#         return {
#             "Stance Phase": float(avg_stance),
#             "Swing Phase": float(avg_swing)
#         }
#     except Exception as e:
#         print(f"計算步態相位錯誤: {str(e)}")
#         return None

# @app.route("/predict", methods=["POST"])
# def predict():
#     if model is None:
#         return jsonify({"error": "模型未加載"}), 500
    
#     if "file" not in request.files:
#         return jsonify({"error": "未上傳文件"}), 400
    
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "空文件名"}), 400
    
#     if file and file.filename.endswith(".csv"):
#         try:
#             # 讀取CSV文件
#             df = pd.read_csv(file)
            
#             # 檢查必要字段
#             required_columns = ["Time", "Gyroscope_Z"]
#             if not all(col in df.columns for col in required_columns):
#                 return jsonify({
#                     "error": "CSV必須包含 'Time' 和 'Gyroscope_Z' 字段",
#                     "analysis": {
#                         "Stance Phase": 0,
#                         "Swing Phase": 0
#                     }
#                 }), 400

#             # 取得時間和陀螺儀數據
#             test_time = df["Time"].values
#             test_gyro = df["Gyroscope_Z"].values

#             # 預測步態事件
#             pred_events = predict_events(model, test_time, test_gyro)
            
#             # 計算步態相位
#             result = calculate_gait_phases(pred_events)
            
#             if result is None:
#                 return jsonify({
#                     "error": "無法計算步態相位 - 數據不足或無有效事件",
#                     "analysis": {
#                         "Stance Phase": 0,
#                         "Swing Phase": 0
#                     }
#                 }), 400

#             # 生成評語
#             comment = generate_comment(result)

#             return jsonify({
#                 "analysis": result,
#                 "comment": comment,  # 新增評語字段
#                 "message": "分析成功",
#                 "status": "success",
#                 "events": [{"time": float(t), "event": e} for t, e in pred_events]
#             })
            
#         except Exception as e:
#             return jsonify({
#                 "error": f"處理錯誤: {str(e)}",
#                 "analysis": {
#                     "Stance Phase": 0,
#                     "Swing Phase": 0
#                 }
#             }), 500
#     else:
#         return jsonify({"error": "僅支持CSV文件"}), 400

# @app.route("/health", methods=["GET"])
# def health_check():
#     """健康檢查端點"""
#     return jsonify({"status": "healthy", "model_loaded": model is not None})

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     app.run(host="0.0.0.0", port=port, debug=False)
# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from datetime import datetime
import sqlite3
import io
import os

app = FastAPI()

# 禁用 GPU（Render 免費版）
tf.config.set_visible_devices([], 'GPU')

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 初始化資料庫 ---
def init_db():
    conn = sqlite3.connect('gait_results.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS flatfoot_results
                 (timestamp TEXT, probability REAL, diagnosis TEXT, user_id TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 載入 V02 模型 ---
v02_model_path = "V02_Infer.keras"
if os.path.exists(v02_model_path):
    try:
        v02_model = load_model(v02_model_path, compile=False)
        print(f"✅ 成功載入 V02 模型: {v02_model_path}")
    except Exception as e:
        print(f"❌ 載入 V02 模型時發生錯誤: {e}")
        v02_model = None
else:
    print(f"❌ 找不到 V02 模型檔案: {v02_model_path}")
    v02_model = None

# --- API 根目錄 ---
@app.get("/")
def read_root():
    return {"status": "API is running"}

# --- 預測 HS/TO (dummy 範例) ---
def predict(model, test_time, test_gyro, window_size=60, distance=40):
    x_pred = []
    pred_events = []

    for i in range(window_size, len(test_gyro) - window_size):
        window = test_gyro[i - window_size: i + window_size + 1].reshape(-1, 1)
        x_pred.append(window)

    x_pred = np.array(x_pred)
    y_pred = model.predict(x_pred, verbose=0)
    pred_labels = np.argmax(y_pred, axis=1)

    last_event_idx = {"HS": -distance, "TO": -distance}
    hs_distance_threshold = 30
    to_indices = []
    in_to_segment = False

    for i in range(1, len(pred_labels)):
        if pred_labels[i] == 0 and not in_to_segment:
            start = i
            in_to_segment = True
        elif pred_labels[i] != 0 and in_to_segment:
            end = i - 1
            if end - start >= 5:
                seg = test_gyro[start + window_size: end + window_size + 1]
                local_min_idx = np.argmin(seg)
                global_idx = start + window_size + local_min_idx
                if global_idx - last_event_idx["TO"] >= distance:
                    event_time = test_time[global_idx]
                    pred_events.append((event_time, "TO"))
                    to_indices.append(global_idx)
                    last_event_idx["TO"] = global_idx
            in_to_segment = False

    if in_to_segment:
        end = len(pred_labels) - 1
        if end - start >= 5:
            seg = test_gyro[start + window_size: end + window_size + 1]
            local_min_idx = np.argmin(seg)
            global_idx = start + window_size + local_min_idx
            if global_idx - last_event_idx["TO"] >= distance:
                event_time = test_time[global_idx]
                pred_events.append((event_time, "TO"))
                to_indices.append(global_idx)
                last_event_idx["TO"] = global_idx

    last_hs_idx = -distance
    for i in range(len(to_indices) - 1):
        start_idx = to_indices[i]
        end_idx = to_indices[i + 1]
        if end_idx - start_idx <= 5:
            continue
        seg = test_gyro[start_idx:end_idx + 1]
        local_max_idx = np.argmax(seg)
        hs_global_idx = start_idx + local_max_idx
        if hs_global_idx - last_hs_idx >= hs_distance_threshold:
            event_time = test_time[hs_global_idx]
            pred_events.append((event_time, "HS"))
            last_hs_idx = hs_global_idx

    return pred_events


# --- API: 上傳 JSON 分析 ---
@app.post("/analyze_flatfoot/")
async def analyze_flatfoot(data: dict):
    """
    JSON 格式:
    {
        "Gyroscope_X": [...],
        "Gyroscope_Y": [...],
        "Gyroscope_Z": [...],
        "Acceleration_X": [...],
        "Acceleration_Y": [...],
        "Acceleration_Z": [...]
    }
    """
    try:
        # 1. 讀 JSON
        required_cols = ["Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
                         "Acceleration_X","Acceleration_Y","Acceleration_Z"]
        for col in required_cols:
            if col not in data:
                return JSONResponse(status_code=400, content={"error": f"缺少欄位: {col}"})

        gx = np.array(data["Gyroscope_X"], dtype=np.float32)
        gy = np.array(data["Gyroscope_Y"], dtype=np.float32)
        gz = np.array(data["Gyroscope_Z"], dtype=np.float32)
        ax = np.array(data["Acceleration_X"], dtype=np.float32)
        ay = np.array(data["Acceleration_Y"], dtype=np.float32)
        az = np.array(data["Acceleration_Z"], dtype=np.float32)

        # 2. HS/TO 推論
        hs_to_events = predict_hs_to(gz)

        # 3. V02 模型分析
        if v02_model is not None:
            # 將六軸數據做簡單堆疊 (N,6)
            features = np.stack([gx, gy, gz, ax, ay, az], axis=1)
            # 模型輸入: (1,N,6) 假設 V02 可以接受整段序列
            X_input = features[None, ...]
            probs = v02_model.predict(X_input, verbose=0)  # (1,2)
            prob_flat = float(probs[0,1])
            diagnosis = "高風險" if prob_flat >= 0.6 else "正常"
        else:
            prob_flat = None
            diagnosis = "模型未載入"

        # 4. 存入資料庫
        conn = sqlite3.connect('gait_results.db')
        c = conn.cursor()
        c.execute("INSERT INTO flatfoot_results VALUES (?,?,?,?)",
                  (datetime.now().isoformat(), prob_flat, diagnosis, "current_user"))
        conn.commit()
        conn.close()

        # 5. 回傳結果
        return {
            "status": "success",
            "hs_to_events": [{"idx": i, "event": e} for i,e in hs_to_events],
            "probability": prob_flat,
            "diagnosis": diagnosis,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- API: 取得歷史結果 ---
@app.get("/get_flatfoot_results/")
async def get_results(user_id: str = "current_user"):
    conn = sqlite3.connect('gait_results.db')
    c = conn.cursor()
    c.execute("SELECT * FROM flatfoot_results WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
    results = [{"timestamp": r[0], "probability": r[1], "diagnosis": r[2]} for r in c.fetchall()]
    conn.close()
    return {"status": "success", "results": results}

# --- 啟動 ---
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("Server:app", host="0.0.0.0", port=port, workers=1, timeout_keep_alive=60)



# 成功但沒有用到v02
# from fastapi.middleware.cors import CORSMiddleware
# from tensorflow.keras.models import load_model
# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# import pandas as pd
# import numpy as np
# import io
# from datetime import datetime
# import sqlite3
# import os
# import tensorflow as tf

# app = FastAPI()  # 啟動 FastAPI 應用

# # 禁用 GPU 加速（Render 免費版沒有 GPU）
# tf.config.set_visible_devices([], 'GPU')


# @app.get("/")
# def read_root():
#     return {"status": "API is running"}


# # 初始化資料庫（只需執行一次）
# def init_db():
#     conn = sqlite3.connect('gait_results.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS flatfoot_results
#                  (timestamp TEXT, probability REAL, diagnosis TEXT, user_id TEXT)''')
#     conn.commit()
#     conn.close()


# init_db()  # 確保資料庫表已建立

# # 加入 CORS 中介軟體
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 載入模型（確保 gait_model_5.h5 檔案存在）
# model = load_model("gait_model_5.h5")


# # === 預測邏輯 ===
# def predict(model, test_time, test_gyro, window_size=60, distance=40):
#     x_pred = []
#     pred_events = []

#     for i in range(window_size, len(test_gyro) - window_size):
#         window = test_gyro[i - window_size: i + window_size + 1].reshape(-1, 1)
#         x_pred.append(window)

#     x_pred = np.array(x_pred)
#     y_pred = model.predict(x_pred, verbose=0)
#     pred_labels = np.argmax(y_pred, axis=1)

#     last_event_idx = {"HS": -distance, "TO": -distance}
#     hs_distance_threshold = 30
#     to_indices = []
#     in_to_segment = False

#     for i in range(1, len(pred_labels)):
#         if pred_labels[i] == 0 and not in_to_segment:
#             start = i
#             in_to_segment = True
#         elif pred_labels[i] != 0 and in_to_segment:
#             end = i - 1
#             if end - start >= 5:
#                 seg = test_gyro[start + window_size: end + window_size + 1]
#                 local_min_idx = np.argmin(seg)
#                 global_idx = start + window_size + local_min_idx
#                 if global_idx - last_event_idx["TO"] >= distance:
#                     event_time = test_time[global_idx]
#                     pred_events.append((event_time, "TO"))
#                     to_indices.append(global_idx)
#                     last_event_idx["TO"] = global_idx
#             in_to_segment = False

#     if in_to_segment:
#         end = len(pred_labels) - 1
#         if end - start >= 5:
#             seg = test_gyro[start + window_size: end + window_size + 1]
#             local_min_idx = np.argmin(seg)
#             global_idx = start + window_size + local_min_idx
#             if global_idx - last_event_idx["TO"] >= distance:
#                 event_time = test_time[global_idx]
#                 pred_events.append((event_time, "TO"))
#                 to_indices.append(global_idx)
#                 last_event_idx["TO"] = global_idx

#     last_hs_idx = -distance
#     for i in range(len(to_indices) - 1):
#         start_idx = to_indices[i]
#         end_idx = to_indices[i + 1]
#         if end_idx - start_idx <= 5:
#             continue
#         seg = test_gyro[start_idx:end_idx + 1]
#         local_max_idx = np.argmax(seg)
#         hs_global_idx = start_idx + local_max_idx
#         if hs_global_idx - last_hs_idx >= hs_distance_threshold:
#             event_time = test_time[hs_global_idx]
#             pred_events.append((event_time, "HS"))
#             last_hs_idx = hs_global_idx

#     return pred_events


# # === API 端點 ===
# @app.post("/predict/")
# async def predict_from_csv(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         df = pd.read_csv(io.BytesIO(contents))

#         # 檢查欄位
#         if "Gyroscope_Z" not in df.columns or "Time" not in df.columns:
#             return JSONResponse(status_code=400, content={"error": "缺少 'Time' 或 'Gyroscope_Z' 欄位"})

#         test_time = df["Time"].values
#         test_gyro = df["Gyroscope_Z"].values

#         # 預測
#         results = predict(model, test_time, test_gyro)

#         return {"status": "success", "predictions": [{"time": t, "event": e} for t, e in results]}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})


# @app.post("/analyze_flatfoot/")
# async def analyze_flatfoot(file: UploadFile = File(...)):
#     try:
#         # 1. 讀取 CSV
#         contents = await file.read()
#         df = pd.read_csv(io.BytesIO(contents))

#         # 2. 模型推論（這裡用假數值，實際請替換）
#         prob = 0.65
#         diagnosis = "高風險" if prob >= 0.6 else "正常"

#         # 3. 儲存結果到資料庫
#         conn = sqlite3.connect('gait_results.db')
#         c = conn.cursor()
#         c.execute("INSERT INTO flatfoot_results VALUES (?,?,?,?)",
#                   (datetime.now().isoformat(), prob, diagnosis, "current_user"))
#         conn.commit()
#         conn.close()

#         return {
#             "status": "success",
#             "probability": prob,
#             "diagnosis": diagnosis,
#             "timestamp": datetime.now().isoformat()
#         }
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})


# @app.get("/get_flatfoot_results/")
# async def get_results(user_id: str = "current_user"):
#     conn = sqlite3.connect('gait_results.db')
#     c = conn.cursor()
#     c.execute("SELECT * FROM flatfoot_results WHERE user_id=? ORDER BY timestamp DESC", (user_id,))
#     results = [{"timestamp": r[0], "probability": r[1], "diagnosis": r[2]} for r in c.fetchall()]
#     conn.close()
#     return {"status": "success", "results": results}


# # === 啟動程式 ===
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run(
#         "Server:app",  # 注意這裡 S 要大寫，跟檔案名一致
#         host="0.0.0.0",
#         port=port,
#         workers=1,  # Render 免費版只支援單 worker
#         timeout_keep_alive=60
#     )

# import os  # 新增：必须导入 os 模块
# from flask import Flask, request, jsonify
# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import load_model
# from io import StringIO

# # 禁用 TensorFlow 的 GPU 使用（Render 只支持 CPU）
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# app = Flask(__name__)

# # 载入模型（确保文件路径正确）
# try:
#     model = load_model("gait_model_5.h5")
# except Exception as e:
#     print(f"模型加载失败: {str(e)}")
#     model = None

# def predict_events(model, test_time, test_gyro, window_size=60, distance=40):
#     """使用模型预测 HS 和 TO 事件"""
#     if model is None:
#         return []
        
#     x_pred = []
#     pred_events = []

#     # 准备滑动窗口数据
#     for i in range(window_size, len(test_gyro) - window_size):
#         window = test_gyro[i - window_size : i + window_size + 1].reshape(-1, 1) 
#         x_pred.append(window)

#     x_pred = np.array(x_pred)
#     y_pred = model.predict(x_pred, verbose=0)
#     pred_labels = np.argmax(y_pred, axis=1)
    
#     # 事件检测参数
#     last_event_idx = {"HS": -distance, "TO": -distance} 
#     hs_distance_threshold = 30  # HS事件最小间隔
#     to_indices = []  # 储存TO事件索引
#     in_to_segment = False

#     # 检测TO事件 (模型标签为0)
#     for i in range(1, len(pred_labels)):
#         if pred_labels[i] == 0 and not in_to_segment:
#             start = i
#             in_to_segment = True
#         elif pred_labels[i] != 0 and in_to_segment:
#             end = i - 1
#             if end - start >= 5:  # 最小持续时间
#                 seg = test_gyro[start + window_size : end + window_size + 1]
#                 local_min_idx = np.argmin(seg)  # 找局部最小值
#                 global_idx = start + window_size + local_min_idx
#                 if global_idx - last_event_idx["TO"] >= distance:
#                     event_time = test_time[global_idx]
#                     pred_events.append((event_time, "TO"))
#                     to_indices.append(global_idx)
#                     last_event_idx["TO"] = global_idx
#             in_to_segment = False

#     # 处理最后一个可能未结束的TO段
#     if in_to_segment:
#         end = len(pred_labels) - 1
#         if end - start >= 5:
#             seg = test_gyro[start + window_size : end + window_size + 1]
#             local_min_idx = np.argmin(seg)
#             global_idx = start + window_size + local_min_idx
#             if global_idx - last_event_idx["TO"] >= distance:
#                 event_time = test_time[global_idx]
#                 pred_events.append((event_time, "TO"))
#                 to_indices.append(global_idx)
#                 last_event_idx["TO"] = global_idx

#     # 检测HS事件 (在TO事件之间找局部最大值)
#     last_hs_idx = -distance
#     for i in range(len(to_indices) - 1):
#         start_idx = to_indices[i]
#         end_idx = to_indices[i+1]
#         if end_idx - start_idx <= 5:  # 忽略过短区间
#             continue
#         seg = test_gyro[start_idx:end_idx+1]
#         local_max_idx = np.argmax(seg)  # 找局部最大值
#         hs_global_idx = start_idx + local_max_idx
#         if hs_global_idx - last_hs_idx >= hs_distance_threshold:
#             event_time = test_time[hs_global_idx]
#             pred_events.append((event_time, "HS"))
#             last_hs_idx = hs_global_idx
    
#     return pred_events

# def calculate_gait_phases(pred_events):
#     """直接从预测事件计算站立期和摆动期"""
#     try:
#         # 分离HS和TO事件
#         hs_times = [t for t, e in pred_events if e == "HS"]
#         to_times = [t for t, e in pred_events if e == "TO"]

#         if len(hs_times) < 2 or len(to_times) < 1:
#             return None

#         stance_phases = []
#         swing_phases = []

#         # 计算每个步态周期的相位
#         for i in range(len(hs_times) - 1):
#             hs1 = hs_times[i]
#             hs2 = hs_times[i + 1]
            
#             # 找出当前周期内的TO事件
#             cycle_to = [to for to in to_times if hs1 < to < hs2]
#             if not cycle_to:
#                 continue
                
#             to = cycle_to[0]  # 取第一个TO事件
#             gait_cycle = hs2 - hs1
#             stance_duration = to - hs1
#             swing_duration = hs2 - to

#             if gait_cycle <= 0:
#                 continue

#             stance_phases.append(stance_duration / gait_cycle)
#             swing_phases.append(swing_duration / gait_cycle)

#         if not stance_phases or not swing_phases:
#             return None

#         # 计算平均比例
#         avg_stance = np.mean(stance_phases)
#         avg_swing = np.mean(swing_phases)
        
#         return {
#             "Stance Phase": float(avg_stance),
#             "Swing Phase": float(avg_swing)
#         }
#     except Exception as e:
#         print(f"计算步态相位错误: {str(e)}")
#         return None

# @app.route("/predict", methods=["POST"])  # 修正：移除URL末尾的斜杠
# def predict():
#     if model is None:
#         return jsonify({"error": "模型未加载"}), 500
    
#     if "file" not in request.files:
#         return jsonify({"error": "未上传文件"}), 400
    
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "空文件名"}), 400
    
#     if file and file.filename.endswith(".csv"):
#         try:
#             # 读取CSV文件
#             df = pd.read_csv(file)
            
#             # 检查必要字段
#             required_columns = ["Time", "Gyroscope_Z"]
#             if not all(col in df.columns for col in required_columns):
#                 return jsonify({
#                     "error": "CSV必须包含 'Time' 和 'Gyroscope_Z' 字段",
#                     "analysis": {
#                         "Stance Phase": 0,
#                         "Swing Phase": 0
#                     }
#                 }), 400

#             # 取得时间和陀螺仪数据
#             test_time = df["Time"].values
#             test_gyro = df["Gyroscope_Z"].values

#             # 预测步态事件
#             pred_events = predict_events(model, test_time, test_gyro)
            
#             # 计算步态相位
#             result = calculate_gait_phases(pred_events)
            
#             if result is None:
#                 return jsonify({
#                     "error": "无法计算步态相位 - 数据不足或无有效事件",
#                     "analysis": {
#                         "Stance Phase": 0,
#                         "Swing Phase": 0
#                     }
#                 }), 400

#             return jsonify({
#                 "analysis": result,
#                 "message": "分析成功",
#                 "status": "success",
#                 "events": [{"time": float(t), "event": e} for t, e in pred_events]  # 确保时间是float类型
#             })
            
#         except Exception as e:
#             return jsonify({
#                 "error": f"处理错误: {str(e)}",
#                 "analysis": {
#                     "Stance Phase": 0,
#                     "Swing Phase": 0
#                 }
#             }), 500
#     else:
#         return jsonify({"error": "仅支持CSV文件"}), 400

# @app.route("/health", methods=["GET"])
# def health_check():
#     """健康检查端点"""
#     return jsonify({"status": "healthy", "model_loaded": model is not None})

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))  # Render 默认使用PORT环境变量
#     app.run(host="0.0.0.0", port=port, debug=False)
