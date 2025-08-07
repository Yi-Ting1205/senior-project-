from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from io import StringIO

app = Flask(__name__)

# 載入模型
model = load_model("gait_model_5.h5")

def predict_events(model, test_time, test_gyro, window_size=60, distance=40):
    """使用模型預測 HS 和 TO 事件"""
    x_pred = []
    pred_events = []

    # 準備滑動窗口數據
    for i in range(window_size, len(test_gyro) - window_size):
        window = test_gyro[i - window_size : i + window_size + 1].reshape(-1, 1) 
        x_pred.append(window)

    x_pred = np.array(x_pred)
    y_pred = model.predict(x_pred, verbose=0)
    pred_labels = np.argmax(y_pred, axis=1)
    
    # 事件檢測參數
    last_event_idx = {"HS": -distance, "TO": -distance} 
    hs_distance_threshold = 30  # HS事件最小間隔
    to_indices = []  # 儲存TO事件索引
    in_to_segment = False

    # 檢測TO事件 (模型標籤為0)
    for i in range(1, len(pred_labels)):
        if pred_labels[i] == 0 and not in_to_segment:
            start = i
            in_to_segment = True
        elif pred_labels[i] != 0 and in_to_segment:
            end = i - 1
            if end - start >= 5:  # 最小持續時間
                seg = test_gyro[start + window_size : end + window_size + 1]
                local_min_idx = np.argmin(seg)  # 找局部最小值
                global_idx = start + window_size + local_min_idx
                if global_idx - last_event_idx["TO"] >= distance:
                    event_time = test_time[global_idx]
                    pred_events.append((event_time, "TO"))
                    to_indices.append(global_idx)
                    last_event_idx["TO"] = global_idx
            in_to_segment = False

    # 處理最後一個可能未結束的TO段
    if in_to_segment:
        end = len(pred_labels) - 1
        if end - start >= 5:
            seg = test_gyro[start + window_size : end + window_size + 1]
            local_min_idx = np.argmin(seg)
            global_idx = start + window_size + local_min_idx
            if global_idx - last_event_idx["TO"] >= distance:
                event_time = test_time[global_idx]
                pred_events.append((event_time, "TO"))
                to_indices.append(global_idx)
                last_event_idx["TO"] = global_idx

    # 檢測HS事件 (在TO事件之間找局部最大值)
    last_hs_idx = -distance
    for i in range(len(to_indices) - 1):
        start_idx = to_indices[i]
        end_idx = to_indices[i+1]
        if end_idx - start_idx <= 5:  # 忽略過短區間
            continue
        seg = test_gyro[start_idx:end_idx+1]
        local_max_idx = np.argmax(seg)  # 找局部最大值
        hs_global_idx = start_idx + local_max_idx
        if hs_global_idx - last_hs_idx >= hs_distance_threshold:
            event_time = test_time[hs_global_idx]
            pred_events.append((event_time, "HS"))
            last_hs_idx = hs_global_idx
    
    return pred_events

def calculate_gait_phases(pred_events):
    """直接從預測事件計算站立期和擺動期"""
    try:
        # 分離HS和TO事件
        hs_times = [t for t, e in pred_events if e == "HS"]
        to_times = [t for t, e in pred_events if e == "TO"]

        if len(hs_times) < 2 or len(to_times) < 1:
            return None

        stance_phases = []
        swing_phases = []

        # 計算每個步態周期的相位
        for i in range(len(hs_times) - 1):
            hs1 = hs_times[i]
            hs2 = hs_times[i + 1]
            
            # 找出當前周期內的TO事件
            cycle_to = [to for to in to_times if hs1 < to < hs2]
            if not cycle_to:
                continue
                
            to = cycle_to[0]  # 取第一個TO事件
            gait_cycle = hs2 - hs1
            stance_duration = to - hs1
            swing_duration = hs2 - to

            if gait_cycle <= 0:
                continue

            stance_phases.append(stance_duration / gait_cycle)
            swing_phases.append(swing_duration / gait_cycle)

        if not stance_phases or not swing_phases:
            return None

        # 計算平均比例
        avg_stance = np.mean(stance_phases)
        avg_swing = np.mean(swing_phases)
        
        return {
            "Stance Phase": float(avg_stance),
            "Swing Phase": float(avg_swing)
        }
    except Exception as e:
        print(f"計算步態相位錯誤: {str(e)}")
        return None

@app.route("/predict/", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    if file and file.filename.endswith(".csv"):
        try:
            # 讀取CSV文件
            df = pd.read_csv(file)
            
            # 檢查必要欄位
            if "Gyroscope_Z" not in df.columns or "Time" not in df.columns:
                return jsonify({
                    "error": "CSV必須包含 'Time' 和 'Gyroscope_Z' 欄位",
                    "analysis": {
                        "Stance Phase": 0,
                        "Swing Phase": 0
                    }
                }), 400

            # 取得時間和陀螺儀數據
            test_time = df["Time"].values
            test_gyro = df["Gyroscope_Z"].values

            # 預測步態事件
            pred_events = predict_events(model, test_time, test_gyro)
            
            # 計算步態相位
            result = calculate_gait_phases(pred_events)
            
            if result is None:
                return jsonify({
                    "error": "無法計算步態相位 - 數據不足或無有效事件",
                    "analysis": {
                        "Stance Phase": 0,
                        "Swing Phase": 0
                    }
                }), 400

            return jsonify({
                "analysis": result,
                "message": "分析成功",
                "status": "success",
                "events": [{"time": t, "event": e} for t, e in pred_events]  # 可選: 返回事件詳細信息
            })
            
        except Exception as e:
            return jsonify({
                "error": f"處理錯誤: {str(e)}",
                "analysis": {
                    "Stance Phase": 0,
                    "Swing Phase": 0
                }
            }), 500
    else:
        return jsonify({"error": "僅支持CSV文件"}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
    
# from flask import Flask, request, jsonify
# import pandas as pd
# from io import StringIO

# app = Flask(__name__)

# def process_gait_file(df):
#     try:
#         # 檢查必要欄位
#         if "Pred_result" not in df.columns or "Time" not in df.columns:
#             return None

#         hs_all = df[df["Pred_result"] == "HS"]["Time"].dropna().values
#         to_all = df[df["Pred_result"] == "TO"]["Time"].dropna().values

#         if len(hs_all) < 2 or len(to_all) == 0:
#             return None

#         stance_list = []
#         swing_list = []

#         for i in range(len(hs_all) - 1):
#             hs1 = hs_all[i]
#             hs2 = hs_all[i + 1]
#             to_candidates = [to for to in to_all if hs1 < to < hs2]
#             if not to_candidates:
#                 continue
#             to = to_candidates[0]

#             gait_cycle = hs2 - hs1
#             stance = to - hs1
#             swing = hs2 - to

#             if gait_cycle <= 0:
#                 continue

#             stance_list.append(stance / gait_cycle)
#             swing_list.append(swing / gait_cycle)

#         if not stance_list or not swing_list:
#             return None

#         return {
#             "stancePhase": sum(stance_list) / len(stance_list),
#             "swingPhase": sum(swing_list) / len(swing_list)
#         }
#     except Exception as e:
#         print(f"處理錯誤: {str(e)}")
#         return None

# @app.route("/predict/", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400
    
#     if file and file.filename.endswith(".csv"):
#         try:
#             df = pd.read_csv(file)
#             result = process_gait_file(df)
            
#             if result is None:
#                 return jsonify({
#                     "error": "Invalid or insufficient data",
#                     "analysis": {
#                         "stancePhase": 0,
#                         "swingPhase": 0
#                     }
#                 }), 400

#             return jsonify({
#                 "analysis": result,
#                 "message": "Analysis successful",
#                 "status": "success"
#             })
#         except Exception as e:
#             return jsonify({
#                 "error": str(e),
#                 "analysis": {
#                     "stancePhase": 0,
#                     "swingPhase": 0
#                 }
#             }), 500
#     else:
#         return jsonify({"error": "Only CSV files allowed"}), 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000, debug=True)
    
# from flask import Flask, request, jsonify
# import pandas as pd

# app = Flask(__name__)

# @app.route("/predict/", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
    
#     file = request.files["file"]
#     if file.filename == "":
#         return jsonify({"error": "Empty filename"}), 400
    
#     if file and file.filename.endswith(".csv"):
#         # 读取 CSV 文件（这里仅做示例，你可以替换成你的机器学习模型）
#         df = pd.read_csv(file)
#         # 模拟返回预测结果
#         result = {
#             "status": "success",
#             "prediction": "Your data was processed!",
#             "data_sample": df.head().to_dict()
#         }
#         return jsonify(result)
#     else:
#         return jsonify({"error": "Only CSV files allowed"}), 400

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8000)
