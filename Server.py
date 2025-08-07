from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
from typing import Dict, Any

app = Flask(__name__)

def process_gait_file(df: pd.DataFrame) -> Dict[str, Any]:
    """
    步態分析核心邏輯
    返回格式範例:
    {
        "Stance Phase": 0.62,
        "Swing Phase": 0.38,
        "RawData": {
            "HS_Times": [0.1, 0.5, 0.9],
            "TO_Times": [0.3, 0.7]
        }
    }
    """
    # 檢查必要欄位
    required_columns = ["Pred_result", "Time"]
    if not all(col in df.columns for col in required_columns):
        return None

    # 提取 HS (Heel Strike) 和 TO (Toe Off) 時間點
    hs_all = df[df["Pred_result"] == "HS"]["Time"].dropna().values
    to_all = df[df["Pred_result"] == "TO"]["Time"].dropna().values

    # 基本數據驗證
    if len(hs_all) < 2 or len(to_all) == 0:
        return None

    stance_list = []
    swing_list = []

    # 計算每個步態週期
    for i in range(len(hs_all) - 1):
        hs1 = hs_all[i]
        hs2 = hs_all[i + 1]
        
        # 找出介於兩個 HS 之間的 TO 事件
        to_candidates = [to for to in to_all if hs1 < to < hs2]
        if not to_candidates:
            continue
            
        to = to_candidates[0]  # 取第一個符合的 TO 事件
        gait_cycle = hs2 - hs1
        
        # 計算站立期和擺動期
        stance = to - hs1
        swing = hs2 - to

        # 排除異常值
        if gait_cycle <= 0:
            continue

        stance_list.append(stance / gait_cycle)
        swing_list.append(swing / gait_cycle)

    # 檢查有效數據
    if not stance_list or not swing_list:
        return None

    # 計算平均值
    stance_mean = sum(stance_list) / len(stance_list)
    swing_mean = sum(swing_list) / len(swing_list)

    return {
        "Stance Phase": stance_mean,
        "Swing Phase": swing_mean,
        "RawData": {
            "HS_Times": hs_all.tolist(),
            "TO_Times": to_all.tolist()
        }
    }

@app.route("/predict/", methods=["POST"])
def predict():
    """
    處理上傳的 CSV 文件並進行步態分析
    成功返回格式:
    {
        "analysis": {
            "Stance Phase": 0.62,
            "Swing Phase": 0.38
        },
        "message": "Analysis successful",
        "status": "success"
    }
    """
    # 檢查文件上傳
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded"
        }), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Empty filename"
        }), 400
    
    # 檢查文件類型
    if not (file and file.filename.lower().endswith('.csv')):
        return jsonify({
            "status": "error",
            "message": "Only CSV files allowed"
        }), 400

    try:
        # 讀取 CSV 文件
        contents = file.read().decode('utf-8')
        data = StringIO(contents)
        df = pd.read_csv(data)

        # 進行步態分析
        result = process_gait_file(df)
        if result is None:
            return jsonify({
                "status": "error",
                "message": "Invalid or insufficient data"
            }), 400

        # 成功響應
        return jsonify({
            "status": "success",
            "message": "Analysis successful",
            "analysis": {
                "Stance Phase": result["Stance Phase"],
                "Swing Phase": result["Swing Phase"]
            },
            "raw_data": result["RawData"]  # 可選，用於調試
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Processing error: {str(e)}"
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    
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
