import uvicorn
import pandas as pd
import numpy as np
import tensorflow as tf
import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import traceback

app = FastAPI(title="Arch Detection API", version="1.0")

# --- CORS 設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- 載入模型 (帶錯誤處理) --------
gait_model = None
arch_model = None
model_load_status = {"gait_model": False, "arch_model": False}

try:
    # 檢查模型文件是否存在
    if os.path.exists("gait_model_5.h5"):
        gait_model = tf.keras.models.load_model("gait_model_5.h5", compile=False)
        model_load_status["gait_model"] = True
        print("✅ Gait 模型載入成功")
    else:
        print("❌ 找不到 gait_model_5.h5")
        
    # 檢查 arch 模型文件
    arch_model_path = "V02_Infer.keras"  # 修改為正確路徑
    if os.path.exists(arch_model_path):
        arch_model = tf.keras.models.load_model(arch_model_path, compile=False)
        model_load_status["arch_model"] = True
        print("✅ Arch 模型載入成功")
    else:
        print(f"❌ 找不到 {arch_model_path}")
        # 列出文件幫助調試
        print("當前目錄文件:", os.listdir('.'))
        
except Exception as e:
    print(f"❌ 模型載入錯誤: {e}")
    traceback.print_exc()

# -------- HS/TO 推論函式 --------
def detect_hs_to(df: pd.DataFrame):
    """
    用 gait_model 產生 HS/TO event
    """
    try:
        if "Gyroscope_Z" not in df.columns or "Time" not in df.columns:
            raise ValueError("缺少 'Time' 或 'Gyroscope_Z' 欄位")

        test_time = df["Time"].values
        test_gyro = df["Gyroscope_Z"].values

        # 檢查數據長度
        if len(test_gyro) < 121:  # window_size * 2 + 1
            print(f"數據長度不足: {len(test_gyro)}")
            df["Pred_result"] = ""
            return df

        # 以下模擬第二份程式碼的推論
        x_pred = []
        window_size = 60
        distance = 40
        
        for i in range(window_size, len(test_gyro) - window_size):
            window = test_gyro[i - window_size : i + window_size + 1].reshape(-1, 1)
            x_pred.append(window)
            
        x_pred = np.array(x_pred)
        if len(x_pred) == 0:
            df["Pred_result"] = ""
            return df  # 無法推論
            
        if gait_model is None:
            print("⚠️ Gait 模型未載入，使用模擬數據")
            df["Pred_result"] = ""
            return df
            
        y_pred = gait_model.predict(x_pred, verbose=0)
        pred_labels = np.argmax(y_pred, axis=1)

        # 計算 TO / HS index
        last_event_idx = {"HS": -distance, "TO": -distance}
        hs_distance_threshold = 30
        pred_events = []
        to_indices = []
        in_to_segment = False

        for i in range(1, len(pred_labels)):
            if pred_labels[i] == 0 and not in_to_segment:
                start = i
                in_to_segment = True
            elif pred_labels[i] != 0 and in_to_segment:
                end = i - 1
                if end - start >= 5:
                    seg = test_gyro[start + window_size : end + window_size + 1]
                    local_min_idx = np.argmin(seg)
                    global_idx = start + window_size + local_min_idx
                    if global_idx - last_event_idx["TO"] >= distance:
                        pred_events.append((global_idx, "TO"))
                        to_indices.append(global_idx)
                        last_event_idx["TO"] = global_idx
                in_to_segment = False

        if in_to_segment:
            end = len(pred_labels) - 1
            if end - start >= 5:
                seg = test_gyro[start + window_size : end + window_size + 1]
                local_min_idx = np.argmin(seg)
                global_idx = start + window_size + local_min_idx
                if global_idx - last_event_idx["TO"] >= distance:
                    pred_events.append((global_idx, "TO"))
                    to_indices.append(global_idx)
                    last_event_idx["TO"] = global_idx

        last_hs_idx = -distance
        for i in range(len(to_indices) - 1):
            start_idx = to_indices[i]
            end_idx = to_indices[i+1]
            if end_idx - start_idx <= 5:
                continue
            seg = test_gyro[start_idx:end_idx+1]
            local_max_idx = np.argmax(seg)
            hs_global_idx = start_idx + local_max_idx
            if hs_global_idx - last_hs_idx >= hs_distance_threshold:
                pred_events.append((hs_global_idx, "HS"))
                last_hs_idx = hs_global_idx

        # 將 Pred_result 回寫到 df
        df["Pred_result"] = ""
        for idx, event in pred_events:
            if idx < len(df):
                df.at[idx, "Pred_result"] = event
                
        print(f"檢測到 {len(pred_events)} 個步態事件")
        return df

    except Exception as e:
        print(f"HS/TO 檢測錯誤: {e}")
        df["Pred_result"] = ""
        return df

# -------- PAA + 切步 + 視窗化 --------
def paa_fast(seg, M=100):
    L, F = seg.shape
    idx = (np.linspace(0, L, M + 1)).astype(int)
    out = np.add.reduceat(seg, idx[:-1], axis=0)
    w = np.maximum(np.diff(idx)[:, None], 1)
    return out / w

def build_windows(df):
    FEAT_COLS = ["Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
                 "Acceleration_X","Acceleration_Y","Acceleration_Z"]

    # 檢查必要欄位
    for col in FEAT_COLS + ["Pred_result"]:
        if col not in df.columns:
            print(f"❌ 缺少欄位: {col}")
            return np.empty((0, 500, 6), dtype=np.float32)

    ev = df["Pred_result"].astype(str).str.upper().fillna("")
    to_idx = df.index[ev.eq("TO")].to_numpy()
    paa_list = []
    g = df[FEAT_COLS].to_numpy(dtype=np.float32)

    print(f"檢測到 {len(to_idx)} 個 TO 標記")

    for i in range(len(to_idx) - 1):
        a, b = int(to_idx[i]), int(to_idx[i+1])
        if b <= a or b - a < 10:  # 確保有足夠數據點
            continue
        step = g[a:b]
        paa = paa_fast(step, 100)
        paa_list.append(paa)

    WIN_STEPS = 5
    stride = 2
    
    if len(paa_list) < WIN_STEPS:
        print(f"❌ 不足的步態周期: {len(paa_list)} < {WIN_STEPS}")
        return np.empty((0, 500, 6), dtype=np.float32)
    
    seqs = []
    for s in range(0, len(paa_list) - WIN_STEPS + 1, stride):
        seq = np.concatenate(paa_list[s:s+WIN_STEPS], axis=0)
        seqs.append(seq.astype(np.float32))

    print(f"生成 {len(seqs)} 個分析窗口")
    return np.stack(seqs, axis=0) if seqs else np.empty((0, 500, 6), dtype=np.float32)

# -------- 狀態檢查端點 --------
@app.get("/")
async def root():
    return {
        "status": "API is running",
        "models_loaded": model_load_status,
        "service": "Arch Detection API"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models": model_load_status,
        "timestamp": pd.Timestamp.now().isoformat()
    }

# -------- API Endpoint --------
@app.post("/arch/")
async def arch_inference(file: UploadFile = File(...)):
    try:
        print(f"收到檔案: {file.filename}")
        
        # 讀取 CSV
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        print(f"數據形狀: {df.shape}")

        # 1) gait model 推論 HS/TO
        df = detect_hs_to(df)

        # 2) 切步 & 視窗化
        X = build_windows(df)
        if len(X) == 0:
            return JSONResponse(
                status_code=400, 
                content={"status": "error", "message": "無法形成有效的 5-step 視窗"}
            )

        # 3) 扁平足模型推論
        if arch_model is None:
            return JSONResponse(
                status_code=500,
                content={"status": "error", "message": "Arch 模型未載入"}
            )
            
        probs = arch_model.predict(X, batch_size=64, verbose=0)
        preds = probs.argmax(axis=1)
        maj = int(np.bincount(preds).argmax())
        p_normal = float(probs[:,0].mean())
        p_flat = float(probs[:,1].mean())

        # 4) 回傳結果 (兼容 Swift 格式)
        result_data = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "probability": p_flat,
            "diagnosis": "高風險" if maj == 1 else "正常"
        }

        return {
            "status": "success",
            "results": [result_data],
            "analysis_details": {
                "file": file.filename,
                "windows_count": len(X),
                "p_normal": p_normal,
                "p_flat": p_flat,
                "arch_type": "flatfoot" if maj == 1 else "normal"
            }
        }

    except Exception as e:
        error_msg = f"分析錯誤: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": error_msg}
        )

# --- 啟動 ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"啟動服務在端口 {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1, timeout_keep_alive=60)
