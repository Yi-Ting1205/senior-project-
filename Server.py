from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route("/predict/", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    if file and file.filename.endswith(".csv"):
        # 读取 CSV 文件（这里仅做示例，你可以替换成你的机器学习模型）
        df = pd.read_csv(file)
        # 模拟返回预测结果
        result = {
            "status": "success",
            "prediction": "Your data was processed!",
            "data_sample": df.head().to_dict()
        }
        return jsonify(result)
    else:
        return jsonify({"error": "Only CSV files allowed"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)