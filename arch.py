from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os, warnings, requests
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import io
import tensorflow as tf
from tensorflow.keras import layers, models

"""
V02 Inference API — 足弓分析服務（整合步態事件預測）
==================================================
- 接收 Swift App 上傳的 CSV 檔案
- 先呼叫第二份服務進行步態事件預測（HS/TO）
- 使用預測結果進行足弓分析
- 輸出足弓類型預測給 Swift App
"""

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- 配置參數 ---------------------------
GAIT_PREDICTION_URL = "http://localhost:8000/predict/"  # 第二份服務的 URL
PAA_IDX = 100
WIN_STEPS = 5
STRIDE_PAA_WIN = 2
FEAT_COLS = ["Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
             "Acceleration_X","Acceleration_Y","Acceleration_Z"]

# --------------------------- 模型加載 ---------------------------
def load_arch_model():
    """加載足弓分析模型"""
    try:
        model_path = r"C:\GITHUB\GaitHelper\weight\V02_Infer.keras"
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ 足弓分析模型加載成功")
        return model
    except Exception as e:
        print(f"❌ 模型加載失敗: {e}")
        return None

# 全局模型變量
arch_model = load_arch_model()

# --------------------------- 輔助函數 ---------------------------
def call_gait_prediction_service(csv_data: bytes, filename: str) -> pd.DataFrame:
    """
    呼叫第二份步態事件預測服務
    返回添加了 Pred_result 欄位的 DataFrame
    """
    try:
        # 準備上傳檔案
        files = {'file': (filename, csv_data, 'text/csv')}
        
        print(f"📡 呼叫步態事件預測服務: {GAIT_PREDICTION_URL}")
        response = requests.post(GAIT_PREDICTION_URL, files=files, timeout=30)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"步態預測服務錯誤: {response.text}"
            )
        
        result = response.json()
        
        if result.get("status") != "success":
            raise HTTPException(
                status_code=500,
                detail=f"步態預測失敗: {result.get('error', 'Unknown error')}"
            )
        
        # 讀取原始 CSV
        df = pd.read_csv(io.BytesIO(csv_data))
        
        # 添加 Pred_result 欄位基於預測結果
        predictions = result.get("predictions", [])
        df["Pred_result"] = ""  # 初始化為空字串
        
        # 將預測事件映射到對應的時間點
        time_values = df["Time"].values if "Time" in df.columns else df.index.values
        
        for pred in predictions:
            pred_time = pred.get("time")
            pred_event = pred.get("event")
            
            if pred_time is not None and pred_event in ["HS", "TO"]:
                # 找到最接近的時間點索引
                time_diff = np.abs(time_values - pred_time)
                closest_idx = np.argmin(time_diff)
                
                # 檢查時間差異是否在合理範圍內
                if time_diff[closest_idx] < 0.1:  # 100ms 容錯
                    df.at[closest_idx, "Pred_result"] = pred_event
        
        print(f"✅ 步態事件預測完成，檢測到 {len(predictions)} 個事件")
        return df
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"無法連接步態預測服務: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"步態預測處理錯誤: {str(e)}"
        )

def paa_fast(seg, M=100):
    """PAA 壓縮"""
    L, F = seg.shape
    idx = (np.linspace(0, L, M+1)).astype(int)
    out = np.add.reduceat(seg, idx[:-1], axis=0)
    w   = np.maximum(np.diff(idx)[:, None], 1)
    return out / w

def five_step_windows(paa_list, win=5, stride=STRIDE_PAA_WIN):
    """生成5步視窗"""
    n = len(paa_list)
    if n < win:
        return np.empty((0, win*PAA_IDX, len(FEAT_COLS)), dtype=np.float32)
    seqs = []
    for s in range(0, n - win + 1, stride):
        seq = np.concatenate(paa_list[s:s+win], axis=0)
        seqs.append(seq.astype(np.float32))
    return np.stack(seqs, axis=0) if seqs else np.empty((0, win*PAA_IDX, len(FEAT_COLS)), dtype=np.float32)

def process_csv_for_arch_prediction(df):
    """
    使用步態事件預測結果進行足弓分析
    """
    # 檢查必要欄位
    missing_feat = [col for col in FEAT_COLS if col not in df.columns]
    if missing_feat:
        raise ValueError(f"缺少必要欄位: {missing_feat}")
    
    # 檢查是否有 TO 事件
    if "Pred_result" not in df.columns:
        raise ValueError("缺少 Pred_result 欄位，請先進行步態事件預測")
    
    # TO→TO 切步
    ev = df["Pred_result"].astype(str).str.upper().fillna("")
    to_idx = df.index[ev.eq("TO")].to_numpy()
    
    print(f"🔍 檢測到 {len(to_idx)} 個 TO 事件")
    
    paa_list = []
    step_spans = []
    sensor_data = df[FEAT_COLS].to_numpy(dtype=np.float32)

    for i in range(len(to_idx) - 1):
        a, b = int(to_idx[i]), int(to_idx[i+1])
        if b <= a:
            continue
        step = sensor_data[a:b]
        
        # 確保步長足夠進行 PAA
        if len(step) < 10:  # 最小步長限制
            continue
            
        paa = paa_fast(step, PAA_IDX)
        paa_list.append(paa)
        step_spans.append((a, b))

    # 生成5步視窗
    X_seq = five_step_windows(paa_list, win=WIN_STEPS, stride=STRIDE_PAA_WIN)
    
    return X_seq, len(paa_list), len(X_seq), len(to_idx)

# --------------------------- API 端點 ---------------------------
@app.post("/analyze-arch/")
async def analyze_arch(file: UploadFile = File(...)):
    """
    足弓分析端點（整合步態事件預測）
    """
    try:
        if arch_model is None:
            return JSONResponse(
                status_code=500, 
                content={"error": "足弓分析模型未加載，服務不可用"}
            )
        
        # 讀取上傳的 CSV
        contents = await file.read()
        
        print(f"📊 接收檔案: {file.filename}, 大小: {len(contents)} bytes")
        
        # 步驟1: 呼叫步態事件預測服務
        df_with_events = call_gait_prediction_service(contents, file.filename)
        
        # 步驟2: 進行足弓分析
        X_seq, n_steps, n_windows, n_to_events = process_csv_for_arch_prediction(df_with_events)
        
        if n_windows == 0:
            return JSONResponse(
                status_code=400, 
                content={
                    "error": "無法生成足夠的分析視窗",
                    "details": {
                        "to_events_detected": n_to_events,
                        "steps_formed": n_steps,
                        "windows_generated": n_windows
                    }
                }
            )
        
        print(f"🔍 步態分析: {n_to_events} TO事件 → {n_steps}步 → {n_windows}視窗")
        
        # 步驟3: 模型預測
        probs = arch_model.predict(X_seq, batch_size=256, verbose=0)
        
        # 計算整體預測結果
        flat_prob_mean = float(probs[:, 1].mean())
        normal_prob_mean = float(probs[:, 0].mean())
        
        # 多數決
        preds = probs.argmax(axis=1)
        final_prediction = int(np.median(preds))
        
        arch_type = "flat" if final_prediction == 1 else "normal"
        confidence = flat_prob_mean if final_prediction == 1 else normal_prob_mean
        
        # 構建響應
        result = {
            "status": "success",
            "gait_analysis": {
                "to_events_detected": n_to_events,
                "steps_formed": n_steps,
                "windows_analyzed": n_windows
            },
            "arch_analysis": {
                "arch_type": arch_type,
                "confidence": round(confidence, 4),
                "normal_probability": round(normal_prob_mean, 4),
                "flat_probability": round(flat_prob_mean, 4),
                "prediction_code": final_prediction
            },
            "details": {
                "total_data_points": len(df_with_events),
                "processing_steps": "步態事件預測 → TO切步 → PAA壓縮 → 5步視窗 → 足弓分類"
            }
        }
        
        print(f"✅ 足弓分析完成: {arch_type} (置信度: {confidence:.3f})")
        return result
        
    except HTTPException as e:
        print(f"❌ HTTP錯誤: {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content={"error": e.detail}
        )
    except Exception as e:
        print(f"❌ 分析錯誤: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"足弓分析過程中發生錯誤: {str(e)}"}
        )

@app.post("/analyze-arch-direct/")
async def analyze_arch_direct(file: UploadFile = File(...)):
    """
    直接足弓分析端點（不使用步態事件預測，用於測試）
    """
    try:
        if arch_model is None:
            return JSONResponse(
                status_code=500, 
                content={"error": "足弓分析模型未加載，服務不可用"}
            )
        
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # 使用自動 TO 檢測（備用方案）
        from scipy.signal import find_peaks
        
        if "Gyroscope_Z" in df.columns:
            gyro_z = df["Gyroscope_Z"].values
            to_indices = find_peaks(-gyro_z, height=2.0, distance=30)[0]
            df["Pred_result"] = ""
            df.loc[to_indices, "Pred_result"] = "TO"
        
        X_seq, n_steps, n_windows, n_to_events = process_csv_for_arch_prediction(df)
        
        if n_windows == 0:
            return JSONResponse(
                status_code=400, 
                content={"error": "無法生成足夠的分析視窗"}
            )
        
        probs = arch_model.predict(X_seq, batch_size=256, verbose=0)
        flat_prob_mean = float(probs[:, 1].mean())
        normal_prob_mean = float(probs[:, 0].mean())
        final_prediction = int(np.median(probs.argmax(axis=1)))
        
        arch_type = "flat" if final_prediction == 1 else "normal"
        confidence = flat_prob_mean if final_prediction == 1 else normal_prob_mean
        
        return {
            "status": "success",
            "arch_type": arch_type,
            "confidence": round(confidence, 4),
            "normal_probability": round(normal_prob_mean, 4),
            "flat_probability": round(flat_prob_mean, 4),
            "steps_detected": n_steps,
            "windows_analyzed": n_windows,
            "method": "direct_auto_to_detection"
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"直接分析錯誤: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    gait_service_available = False
    try:
        response = requests.get(GAIT_PREDICTION_URL.replace("/predict/", "/"), timeout=5)
        gait_service_available = response.status_code == 200
    except:
        pass
    
    return {
        "status": "healthy",
        "arch_model_loaded": arch_model is not None,
        "gait_prediction_service_available": gait_service_available,
        "service": "gait_arch_analysis_integrated"
    }

# --------------------------- 啟動 ---------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # 使用不同端口避免衝突