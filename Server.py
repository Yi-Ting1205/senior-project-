from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO, BytesIO
import base64
from typing import Dict, Any

app = FastAPI()

def process_gait_file(df: pd.DataFrame) -> Dict[str, Any]:
    # 步態分析邏輯（保持不變）
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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        s = str(contents, 'utf-8')
        data = StringIO(s)
        df = pd.read_csv(data)

        result = process_gait_file(df)
        if result is None:
            raise HTTPException(status_code=400, detail="Invalid or insufficient data")

        return JSONResponse(content={
            "analysis": {
                "Stance Phase": result["Stance Phase"],
                "Swing Phase": result["Swing Phase"]
            },
            "message": "Analysis successful"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
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
