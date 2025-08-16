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


from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import os
import logging
from io import StringIO

# 初始化 Flask 應用
app = Flask(__name__)

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 禁用 GPU（Render 只支持 CPU）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 載入模型（放在全局範圍避免重複加載）
try:
    # 步態分析模型
    gait_model = load_model("gait_model_5.h5")
    # 足弓分析模型（新增）
    arch_model = load_model("arch_model.h5") 
    logger.info("模型加載成功")
except Exception as e:
    logger.error(f"模型加載失敗: {str(e)}")
    gait_model = None
    arch_model = None

# 特徵提取函數（用於足弓分析）
def extract_arch_features(df):
    """
    從原始數據提取足弓分析特徵
    返回: 特徵向量 (n_features,)
    """
    features = []
    
    # 1. 陀螺儀Z軸特徵
    gyro_z = df["Gyroscope_Z"].values
    features.extend([
        np.mean(gyro_z),        # 平均值
        np.std(gyro_z),         # 標準差
        np.max(gyro_z),         # 最大值
        np.min(gyro_z),         # 最小值
        np.median(gyro_z),      # 中位數
        np.percentile(gyro_z, 25),  # 25百分位
        np.percentile(gyro_z, 75),  # 75百分位
    ])
    
    # 2. 加速度計特徵（X/Y/Z）
    for col in ["Acceleration_X", "Acceleration_Y", "Acceleration_Z"]:
        acc = df[col].values
        features.extend([
            np.mean(acc),
            np.std(acc),
            np.max(acc) - np.min(acc),  # 峰峰值
        ])
    
    # 3. 時頻域特徵（示例）
    fft = np.abs(np.fft.fft(gyro_z))
    features.extend([
        np.mean(fft[:len(fft)//2]),  # 低頻能量
        np.mean(fft[len(fft)//2:]),   # 高頻能量
    ])
    
    return np.array(features)


def predict_events(model, test_time, test_gyro, window_size=60, distance=40):
    """使用模型预测 HS 和 TO 事件"""
    if model is None:
        return []
        
    x_pred = []
    pred_events = []

    # 准备滑动窗口数据
    for i in range(window_size, len(test_gyro) - window_size):
        window = test_gyro[i - window_size : i + window_size + 1].reshape(-1, 1) 
        x_pred.append(window)

    x_pred = np.array(x_pred)
    y_pred = model.predict(x_pred, verbose=0)
    pred_labels = np.argmax(y_pred, axis=1)
    
    # 事件检测参数
    last_event_idx = {"HS": -distance, "TO": -distance} 
    hs_distance_threshold = 30  # HS事件最小间隔
    to_indices = []  # 储存TO事件索引
    in_to_segment = False

    # 检测TO事件 (模型标签为0)
    for i in range(1, len(pred_labels)):
        if pred_labels[i] == 0 and not in_to_segment:
            start = i
            in_to_segment = True
        elif pred_labels[i] != 0 and in_to_segment:
            end = i - 1
            if end - start >= 5:  # 最小持续时间
                seg = test_gyro[start + window_size : end + window_size + 1]
                local_min_idx = np.argmin(seg)  # 找局部最小值
                global_idx = start + window_size + local_min_idx
                if global_idx - last_event_idx["TO"] >= distance:
                    event_time = test_time[global_idx]
                    pred_events.append((event_time, "TO"))
                    to_indices.append(global_idx)
                    last_event_idx["TO"] = global_idx
            in_to_segment = False

    # 处理最后一个可能未结束的TO段
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

    # 检测HS事件 (在TO事件之间找局部最大值)
    last_hs_idx = -distance
    for i in range(len(to_indices) - 1):
        start_idx = to_indices[i]
        end_idx = to_indices[i+1]
        if end_idx - start_idx <= 5:  # 忽略过短区间
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
    """直接从预测事件计算站立期和摆动期"""
    try:
        # 分离HS和TO事件
        hs_times = [t for t, e in pred_events if e == "HS"]
        to_times = [t for t, e in pred_events if e == "TO"]

        if len(hs_times) < 2 or len(to_times) < 1:
            return None

        stance_phases = []
        swing_phases = []

        # 计算每个步态周期的相位
        for i in range(len(hs_times) - 1):
            hs1 = hs_times[i]
            hs2 = hs_times[i + 1]
            
            # 找出当前周期内的TO事件
            cycle_to = [to for to in to_times if hs1 < to < hs2]
            if not cycle_to:
                continue
                
            to = cycle_to[0]  # 取第一个TO事件
            gait_cycle = hs2 - hs1
            stance_duration = to - hs1
            swing_duration = hs2 - to

            if gait_cycle <= 0:
                continue

            stance_phases.append(stance_duration / gait_cycle)
            swing_phases.append(swing_duration / gait_cycle)

        if not stance_phases or not swing_phases:
            return None

        # 计算平均比例
        avg_stance = np.mean(stance_phases)
        avg_swing = np.mean(swing_phases)
        
        return {
            "Stance Phase": float(avg_stance),
            "Swing Phase": float(avg_swing)
        }
    except Exception as e:
        print(f"计算步态相位错误: {str(e)}")
        return None
# 新增的足弓分析函數
def analyze_foot_arch(model, df):
    """
    分析足弓類型
    返回: (normal_prob, flat_prob)
    """
    if model is None:
        return (0.5, 0.5)  # 默認值
    
    try:
        # 1. 特徵提取
        features = extract_arch_features(df)
        
        # 2. 標準化（假設模型需要標準化數據）
        # 這裡應該使用與訓練時相同的標準化參數
        # features = (features - mean_train) / std_train
        
        # 3. 預測
        probs = model.predict(features.reshape(1, -1), verbose=0)[0]
        normal_prob = float(probs[0])
        flat_prob = float(probs[1])
        
        return (normal_prob, flat_prob)
    except Exception as e:
        logger.error(f"足弓分析失敗: {str(e)}")
        return (0.5, 0.5)  # 失敗時返回中性概率

# 統一的API響應格式
def create_api_response(gait_result, arch_result, events):
    """構建標準化API響應"""
    normal_prob, flat_prob = arch_result
    return {
        "gait_analysis": gait_result or {
            "Stance Phase": 0.6,
            "Swing Phase": 0.4
        },
        "arch_analysis": {
            "normalProb": normal_prob,
            "flatProb": flat_prob,
            "diagnosis": "normal" if normal_prob > 0.5 else "flat"
        },
        "events": [{"time": float(t), "event": e} for t, e in events],
        "status": "success",
        "message": "分析完成"
    }

# 文件上傳端點（主要修改部分）
@app.route("/predict", methods=["POST"])
def predict():
    # 基本檢查
    if gait_model is None or arch_model is None:
        return jsonify({
            "status": "error",
            "message": "服務器模型未加載"
        }), 500
    
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "未上傳文件"
        }), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "空文件名"
        }), 400
    
    # 只接受CSV文件
    if not (file and file.filename.endswith(".csv")):
        return jsonify({
            "status": "error",
            "message": "僅支持CSV文件"
        }), 400
    
    try:
        # 1. 讀取CSV數據
        df = pd.read_csv(file)
        logger.info(f"成功讀取CSV，行數: {len(df)}")
        
        # 2. 檢查必要欄位
        required_columns = ["Time", "Gyroscope_Z", 
                          "Acceleration_X", "Acceleration_Y", "Acceleration_Z"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"CSV缺少必要欄位: {', '.join(missing_cols)}"
            }), 400
        
        # 3. 步態分析
        test_time = df["Time"].values
        test_gyro = df["Gyroscope_Z"].values
        
        pred_events = predict_events(gait_model, test_time, test_gyro)
        gait_result = calculate_gait_phases(pred_events)
        
        # 4. 足弓分析（新增）
        arch_result = analyze_foot_arch(arch_model, df)
        
        # 5. 構建響應
        response = create_api_response(gait_result, arch_result, pred_events)
        logger.info("分析成功完成")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"處理失敗: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"服務器錯誤: {str(e)}"
        }), 500

# 健康檢查端點
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "gait_model": gait_model is not None,
            "arch_model": arch_model is not None
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)


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
