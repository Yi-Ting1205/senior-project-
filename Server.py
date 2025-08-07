from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO

app = Flask(__name__)

def process_gait_file(df):
    try:
        # 檢查必要欄位
        if "Pred_result" not in df.columns or "Time" not in df.columns:
            return None

        hs_all = df[df["Pred_result"] == "HS"]["Time"].dropna().values
        to_all = df[df["Pred_result"] == "TO"]["Time"].dropna().values

        if len(hs_all) < 2 or len(to_all) == 0:
            return None

        stance_list = []
        swing_list = []

        for i in range(len(hs_all) - 1):
            hs1 = hs_all[i]
            hs2 = hs_all[i + 1]
            to_candidates = [to for to in to_all if hs1 < to < hs2]
            if not to_candidates:
                continue
            to = to_candidates[0]

            gait_cycle = hs2 - hs1
            stance = to - hs1
            swing = hs2 - to

            if gait_cycle <= 0:
                continue

            stance_list.append(stance / gait_cycle)
            swing_list.append(swing / gait_cycle)

        if not stance_list or not swing_list:
            return None

        return {
            "stancePhase": sum(stance_list) / len(stance_list),
            "swingPhase": sum(swing_list) / len(swing_list)
        }
    except Exception as e:
        print(f"處理錯誤: {str(e)}")
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
            df = pd.read_csv(file)
            result = process_gait_file(df)
            
            if result is None:
                return jsonify({
                    "error": "Invalid or insufficient data",
                    "analysis": {
                        "stancePhase": 0,
                        "swingPhase": 0
                    }
                }), 400

            return jsonify({
                "analysis": result,
                "message": "Analysis successful",
                "status": "success"
            })
        except Exception as e:
            return jsonify({
                "error": str(e),
                "analysis": {
                    "stancePhase": 0,
                    "swingPhase": 0
                }
            }), 500
    else:
        return jsonify({"error": "Only CSV files allowed"}), 400

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
