import uvicorn
import pandas as pd
import numpy as np
import tensorflow as tf
import io
import os
import sqlite3
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import traceback
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(title="Medical Arch Detection API", version="1.0")

# --- CORS 設定 ---
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
    c.execute('''CREATE TABLE IF NOT EXISTS arch_results
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT, probability REAL, diagnosis TEXT, 
                  user_id TEXT, file_name TEXT, windows_count INTEGER,
                  model_version TEXT, confidence REAL)''')
    conn.commit()
    conn.close()

init_db()

# --- 模型載入函數 (兼容性版本) ---
def load_model_with_fallback(model_path, model_type="gait"):
    """帶有降級載入的模型載入函數"""
    try:
        print(f"嘗試載入 {model_type} 模型: {model_path}")
        
        # 方法1: 嘗試直接載入
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"✅ {model_type} 模型直接載入成功")
            return model, "original"
        except Exception as e:
            print(f"❌ 直接載入失敗: {e}")
            
        # 方法2: 嘗試使用 custom_objects
        try:
            model = tf.keras.models.load_model(
                model_path, 
                compile=False,
                custom_objects={}
            )
            print(f"✅ {model_type} 模型通過 custom_objects 載入成功")
            return model, "custom_objects"
        except Exception as e:
            print(f"❌ custom_objects 載入失敗: {e}")
            
        # 方法3: 嘗試重建架構 (針對 gait model)
        if model_type == "gait" and "gait" in model_path.lower():
            try:
                from tensorflow.keras.models import Model
                from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, GlobalAveragePooling1D, Dense
                
                # 根據錯誤信息重建輸入層
                input_layer = Input(shape=(121, 1), name='input_layer')
                x = Conv1D(64, 3, activation='relu', padding='causal')(input_layer)
                x = BatchNormalization()(x)
                x = Conv1D(128, 3, activation='relu', padding='causal')(x)
                x = BatchNormalization()(x)
                x = GlobalAveragePooling1D()(x)
                x = Dense(64, activation='relu')(x)
                output = Dense(2, activation='softmax')(x)
                
                model = Model(inputs=input_layer, outputs=output)
                print(f"✅ {model_type} 模型架構重建成功")
                return model, "reconstructed"
            except Exception as e:
                print(f"❌ 模型重建失敗: {e}")
                
        return None, "failed"
        
    except Exception as e:
        print(f"❌ {model_type} 模型載入完全失敗: {e}")
        return None, "failed"

# --- 載入模型 ---
gait_model = None
arch_model = None
model_status = {
    "gait_model": "not_loaded",
    "arch_model": "not_loaded",
    "gait_version": "",
    "arch_version": ""
}

print("=" * 60)
print("🩺 醫療級足弓分析API - 模型載入中...")
print("=" * 60)

# 載入 Gait Model
gait_model, gait_version = load_model_with_fallback("gait_model_5.h5", "gait")
if gait_model:
    model_status["gait_model"] = "loaded"
    model_status["gait_version"] = gait_version
    print("✅ Gait Model 載入完成")
else:
    print("❌ Gait Model 載入失敗 - 服務無法正常運行")

# 載入 Arch Model
arch_model, arch_version = load_model_with_fallback("V02_Infer.keras", "arch")
if arch_model:
    model_status["arch_model"] = "loaded"
    model_status["arch_version"] = arch_version
    print("✅ Arch Model 載入完成")
else:
    print("❌ Arch Model 載入失敗 - 服務無法正常運行")

print("=" * 60)
print(f"模型載入狀態: Gait={model_status['gait_model']}, Arch={model_status['arch_model']}")
print("=" * 60)

# --- 醫療級分析函數 ---
def medical_detect_hs_to(df: pd.DataFrame):
    """醫療級 HS/TO 檢測"""
    if gait_model is None:
        raise ValueError("Gait Model 未載入，無法進行醫療級分析")
    
    try:
        if "Gyroscope_Z" not in df.columns:
            raise ValueError("缺少 Gyroscope_Z 數據")
            
        test_gyro = df["Gyroscope_Z"].values
        
        if len(test_gyro) < 121:
            raise ValueError("數據長度不足，至少需要121個數據點")
        
        # 模型推論
        x_pred = []
        window_size = 60
        
        for i in range(window_size, len(test_gyro) - window_size):
            window = test_gyro[i - window_size : i + window_size + 1].reshape(-1, 1)
            x_pred.append(window)
            
        x_pred = np.array(x_pred)
        if len(x_pred) == 0:
            raise ValueError("無法生成有效的推論窗口")
            
        y_pred = gait_model.predict(x_pred, verbose=0)
        pred_labels = np.argmax(y_pred, axis=1)

        # 事件檢測邏輯
        last_event_idx = {"HS": -40, "TO": -40}
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
                    if global_idx - last_event_idx["TO"] >= 40:
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
                if global_idx - last_event_idx["TO"] >= 40:
                    pred_events.append((global_idx, "TO"))
                    to_indices.append(global_idx)
                    last_event_idx["TO"] = global_idx

        last_hs_idx = -40
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

        # 標記結果
        df["Pred_result"] = ""
        for idx, event in pred_events:
            if idx < len(df):
                df.at[idx, "Pred_result"] = event
                
        print(f"醫療級檢測: 找到 {len(pred_events)} 個步態事件")
        return df
        
    except Exception as e:
        raise Exception(f"醫療級 HS/TO 檢測失敗: {str(e)}")

def medical_paa_fast(seg, M=100):
    """醫療級 PAA 壓縮"""
    L, F = seg.shape
    idx = (np.linspace(0, L, M + 1)).astype(int)
    out = np.add.reduceat(seg, idx[:-1], axis=0)
    w = np.maximum(np.diff(idx)[:, None], 1)
    return out / w

def medical_build_windows(df):
    """醫療級窗口構建"""
    FEAT_COLS = ["Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
                 "Acceleration_X","Acceleration_Y","Acceleration_Z"]

    # 驗證數據完整性
    missing_cols = [col for col in FEAT_COLS + ["Pred_result"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要數據欄位: {missing_cols}")

    ev = df["Pred_result"].astype(str).str.upper().fillna("")
    to_idx = df.index[ev.eq("TO")].to_numpy()
    
    if len(to_idx) < 2:
        raise ValueError("TO 事件數量不足，無法進行步態分析")

    paa_list = []
    g = df[FEAT_COLS].to_numpy(dtype=np.float32)

    # 醫療級數據處理
    for i in range(len(to_idx) - 1):
        a, b = int(to_idx[i]), int(to_idx[i+1])
        if b > a and b - a >= 20:
            step = g[a:b]
            paa = medical_paa_fast(step, 100)
            paa_list.append(paa)

    WIN_STEPS = 5
    stride = 2
    
    if len(paa_list) < WIN_STEPS:
        raise ValueError(f"步態周期不足: {len(paa_list)} < {WIN_STEPS}")

    # 構建分析窗口
    seqs = []
    for s in range(0, len(paa_list) - WIN_STEPS + 1, stride):
        seq = np.concatenate(paa_list[s:s+WIN_STEPS], axis=0)
        seqs.append(seq.astype(np.float32))

    return np.stack(seqs, axis=0)

def medical_arch_prediction(X_windows):
    """醫療級扁平足預測"""
    if arch_model is None:
        raise ValueError("Arch Model 未載入，無法進行醫療級預測")
    
    probs = arch_model.predict(X_windows, verbose=0)
    p_flat = float(probs[:, 1].mean())
    p_normal = float(probs[:, 0].mean())
    
    # 醫療級診斷邏輯
    if p_flat >= 0.7:
        diagnosis = "高風險"
        confidence = p_flat
    elif p_flat >= 0.5:
        diagnosis = "中等風險"
        confidence = p_flat
    else:
        diagnosis = "正常"
        confidence = p_normal
        
    return p_flat, diagnosis, confidence

# --- API 端點 ---
@app.get("/medical/status")
async def medical_status():
    """醫療級服務狀態"""
    return {
        "status": "medical_service",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
        "service_level": "medical_grade"
    }

@app.post("/medical/analyze_flatfoot")
async def medical_analyze_flatfoot(file: UploadFile = File(...)):
    """醫療級扁平足分析"""
    try:
        if not gait_model or not arch_model:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": "醫療模型未載入，服務不可用"}
            )
        
        print(f"🩺 醫療級分析請求: {file.filename}")
        
        # 讀取並驗證數據
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
        
        if len(df) < 200:
            raise ValueError("數據量不足，至少需要200個數據點進行醫療級分析")
        
        # 醫療級分析流程
        df = medical_detect_hs_to(df)
        X_windows = medical_build_windows(df)
        
        if len(X_windows) == 0:
            raise ValueError("無法形成有效的醫療級分析窗口")
        
        # 醫療級預測
        p_flat, diagnosis, confidence = medical_arch_prediction(X_windows)
        
        # 記錄到數據庫
        conn = sqlite3.connect('gait_results.db')
        c = conn.cursor()
        c.execute('''INSERT INTO arch_results 
                    (timestamp, probability, diagnosis, user_id, file_name, windows_count, model_version, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                  (datetime.now().isoformat(), p_flat, diagnosis, 
                   "medical_user", file.filename, len(X_windows), 
                   f"gait_{model_status['gait_version']}_arch_{model_status['arch_version']}", confidence))
        conn.commit()
        conn.close()
        
        # 醫療級回應
        return {
            "status": "medical_success",
            "result": {
                "timestamp": datetime.now().isoformat(),
                "probability": round(p_flat, 4),
                "diagnosis": diagnosis,
                "confidence": round(confidence, 4),
                "analysis_level": "medical_grade"
            },
            "metadata": {
                "windows_analyzed": len(X_windows),
                "model_versions": model_status
            }
        }
        
    except Exception as e:
        error_msg = f"醫療級分析錯誤: {str(e)}"
        print(f"❌ {error_msg}")
        return JSONResponse(
            status_code=400,
            content={"status": "medical_error", "message": error_msg}
        )

# --- 啟動 ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"🩺 啟動醫療級服務在端口 {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
# import uvicorn
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import io
# import os
# import sqlite3
# from datetime import datetime
# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from typing import List
# import traceback

# app = FastAPI(title="Arch Detection API", version="1.0")

# # --- CORS 設定 ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- 初始化資料庫 ---
# def init_db():
#     conn = sqlite3.connect('gait_results.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS arch_results
#                  (timestamp TEXT, probability REAL, diagnosis TEXT, user_id TEXT, 
#                   file_name TEXT, windows_count INTEGER)''')
#     conn.commit()
#     conn.close()

# init_db()

# # -------- 載入模型 (帶錯誤處理) --------
# gait_model = None
# arch_model = None
# model_load_status = {"gait_model": False, "arch_model": False}

# print("=" * 50)
# print("開始載入模型...")

# try:
#     # 檢查模型文件是否存在
#     model_files = []
#     for f in os.listdir('.'):
#         if f.endswith(('.h5', '.keras', '.hdf5')):
#             model_files.append(f)
    
#     print(f"找到模型文件: {model_files}")
    
#     # 嘗試載入 gait model
#     gait_model_path = "gait_model_5.h5"
#     if os.path.exists(gait_model_path):
#         print(f"載入 Gait 模型: {gait_model_path}")
#         gait_model = tf.keras.models.load_model(gait_model_path, compile=False)
#         model_load_status["gait_model"] = True
#         print("✅ Gait 模型載入成功")
#     else:
#         print(f"❌ 找不到 {gait_model_path}")
        
#     # 嘗試載入 arch model (嘗試多種可能的路徑)
#     arch_model_paths = ["V02_Infer.keras", "v02_infer.keras", "model.keras"]
#     arch_model_loaded = False
    
#     for path in arch_model_paths:
#         if os.path.exists(path):
#             print(f"載入 Arch 模型: {path}")
#             try:
#                 arch_model = tf.keras.models.load_model(path, compile=False)
#                 model_load_status["arch_model"] = True
#                 arch_model_loaded = True
#                 print("✅ Arch 模型載入成功")
#                 break
#             except Exception as e:
#                 print(f"❌ 載入 {path} 失敗: {e}")
#                 # 嘗試使用 custom_objects 載入
#                 try:
#                     arch_model = tf.keras.models.load_model(
#                         path, 
#                         compile=False,
#                         custom_objects=None
#                     )
#                     model_load_status["arch_model"] = True
#                     arch_model_loaded = True
#                     print("✅ Arch 模型載入成功 (使用 custom_objects)")
#                     break
#                 except Exception as e2:
#                     print(f"❌ 再次載入失敗: {e2}")
    
#     if not arch_model_loaded:
#         print("❌ 所有 Arch 模型載入嘗試失敗")
        
# except Exception as e:
#     print(f"❌ 模型載入錯誤: {e}")
#     traceback.print_exc()

# print(f"模型載入狀態: Gait={model_load_status['gait_model']}, Arch={model_load_status['arch_model']}")
# print("=" * 50)

# # -------- HS/TO 推論函式 (備份版本) --------
# def detect_hs_to_simple(df: pd.DataFrame):
#     """
#     簡單的 HS/TO 檢測 (備份方法)
#     """
#     try:
#         if "Gyroscope_Z" not in df.columns:
#             df["Pred_result"] = ""
#             return df
            
#         gz = df["Gyroscope_Z"].values
        
#         # 簡單的峰值檢測
#         from scipy.signal import find_peaks
        
#         # 標準化數據
#         gz_normalized = (gz - np.mean(gz)) / np.std(gz)
        
#         # 尋找峰值 (TO 事件)
#         to_peaks, _ = find_peaks(-gz_normalized, height=1.0, distance=30)
        
#         # 尋找谷值 (HS 事件) - 在 TO 事件之間
#         hs_events = []
#         for i in range(len(to_peaks) - 1):
#             start = to_peaks[i]
#             end = to_peaks[i + 1]
#             segment = gz_normalized[start:end]
#             if len(segment) > 10:
#                 hs_idx = start + np.argmax(segment)
#                 hs_events.append(hs_idx)
        
#         # 標記結果
#         df["Pred_result"] = ""
#         for idx in to_peaks:
#             if idx < len(df):
#                 df.at[idx, "Pred_result"] = "TO"
                
#         for idx in hs_events:
#             if idx < len(df):
#                 df.at[idx, "Pred_result"] = "HS"
        
#         print(f"簡單檢測: 找到 {len(to_peaks)} 個 TO, {len(hs_events)} 個 HS")
#         return df
        
#     except Exception as e:
#         print(f"簡單 HS/TO 檢測錯誤: {e}")
#         df["Pred_result"] = ""
#         return df

# def detect_hs_to(df: pd.DataFrame):
#     """
#     用 gait_model 產生 HS/TO event (帶備份)
#     """
#     try:
#         # 如果模型未載入，使用簡單方法
#         if gait_model is None:
#             print("⚠️ Gait 模型未載入，使用簡單檢測")
#             return detect_hs_to_simple(df)
            
#         if "Gyroscope_Z" not in df.columns or "Time" not in df.columns:
#             raise ValueError("缺少 'Time' 或 'Gyroscope_Z' 欄位")

#         test_gyro = df["Gyroscope_Z"].values

#         # 檢查數據長度
#         if len(test_gyro) < 121:
#             print(f"數據長度不足: {len(test_gyro)}")
#             return detect_hs_to_simple(df)

#         # 模型推論
#         x_pred = []
#         window_size = 60
        
#         for i in range(window_size, len(test_gyro) - window_size):
#             window = test_gyro[i - window_size : i + window_size + 1].reshape(-1, 1)
#             x_pred.append(window)
            
#         x_pred = np.array(x_pred)
#         if len(x_pred) == 0:
#             return detect_hs_to_simple(df)
            
#         y_pred = gait_model.predict(x_pred, verbose=0)
#         pred_labels = np.argmax(y_pred, axis=1)

#         # 事件檢測邏輯...
#         # [保持原有的事件檢測邏輯]
        
#         df["Pred_result"] = ""
#         # [保持原有的標記邏輯]
        
#         print(f"模型檢測: 找到 {len(pred_events)} 個事件")
#         return df

#     except Exception as e:
#         print(f"HS/TO 檢測錯誤: {e}")
#         return detect_hs_to_simple(df)

# # -------- PAA + 切步 + 視窗化 --------
# def paa_fast(seg, M=100):
#     L, F = seg.shape
#     idx = (np.linspace(0, L, M + 1)).astype(int)
#     out = np.add.reduceat(seg, idx[:-1], axis=0)
#     w = np.maximum(np.diff(idx)[:, None], 1)
#     return out / w

# def build_windows(df):
#     FEAT_COLS = ["Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
#                  "Acceleration_X","Acceleration_Y","Acceleration_Z"]

#     # 檢查必要欄位
#     missing_cols = [col for col in FEAT_COLS + ["Pred_result"] if col not in df.columns]
#     if missing_cols:
#         print(f"❌ 缺少欄位: {missing_cols}")
#         return np.empty((0, 500, 6), dtype=np.float32)

#     ev = df["Pred_result"].astype(str).str.upper().fillna("")
#     to_idx = df.index[ev.eq("TO")].to_numpy()
#     paa_list = []
#     g = df[FEAT_COLS].to_numpy(dtype=np.float32)

#     print(f"檢測到 {len(to_idx)} 個 TO 標記")

#     for i in range(len(to_idx) - 1):
#         a, b = int(to_idx[i]), int(to_idx[i+1])
#         if b <= a or b - a < 10:
#             continue
#         step = g[a:b]
#         paa = paa_fast(step, 100)
#         paa_list.append(paa)

#     WIN_STEPS = 5
#     stride = 2
    
#     if len(paa_list) < WIN_STEPS:
#         print(f"❌ 不足的步態周期: {len(paa_list)} < {WIN_STEPS}")
#         return np.empty((0, 500, 6), dtype=np.float32)
    
#     seqs = []
#     for s in range(0, len(paa_list) - WIN_STEPS + 1, stride):
#         seq = np.concatenate(paa_list[s:s+WIN_STEPS], axis=0)
#         seqs.append(seq.astype(np.float32))

#     print(f"生成 {len(seqs)} 個分析窗口")
#     return np.stack(seqs, axis=0) if seqs else np.empty((0, 500, 6), dtype=np.float32)

# # -------- 狀態檢查端點 --------
# @app.get("/")
# async def root():
#     return {
#         "status": "API is running",
#         "models_loaded": model_load_status,
#         "service": "Arch Detection API"
#     }

# @app.get("/health")
# async def health_check():
#     return {
#         "status": "healthy",
#         "models": model_load_status,
#         "timestamp": datetime.now().isoformat()
#     }

# @app.get("/debug/models")
# async def debug_models():
#     """模型調試信息"""
#     model_info = {
#         "gait_model_loaded": model_load_status["gait_model"],
#         "arch_model_loaded": model_load_status["arch_model"],
#         "current_files": [f for f in os.listdir('.') if f.endswith(('.h5', '.keras', '.hdf5'))],
#         "timestamp": datetime.now().isoformat()
#     }
#     return model_info

# # -------- API Endpoint --------
# @app.post("/analyze_flatfoot/")
# async def analyze_flatfoot(file: UploadFile = File(...)):
#     """
#     兼容 Swift 的端點名稱
#     """
#     try:
#         print(f"收到檔案: {file.filename}")
        
#         # 讀取 CSV
#         content = await file.read()
#         df = pd.read_csv(io.StringIO(content.decode("utf-8")))
#         print(f"數據形狀: {df.shape}")
#         print(f"數據欄位: {df.columns.tolist()}")

#         # 1) HS/TO 檢測
#         df = detect_hs_to(df)

#         # 2) 切步 & 視窗化
#         X = build_windows(df)
#         if len(X) == 0:
#             return JSONResponse(
#                 status_code=400, 
#                 content={"status": "error", "message": "無法形成有效的 5-step 視窗"}
#             )

#         # 3) 扁平足模型推論
#         if arch_model is None:
#             # 使用模擬數據
#             print("⚠️ Arch 模型未載入，使用模擬數據")
#             p_flat = 0.75
#             diagnosis = "高風險"
#         else:
#             try:
#                 probs = arch_model.predict(X, verbose=0)
#                 p_flat = float(probs[:,1].mean())
#                 diagnosis = "高風險" if p_flat >= 0.6 else "正常"
#                 print(f"模型預測: p_flat={p_flat}, diagnosis={diagnosis}")
#             except Exception as e:
#                 print(f"❌ 模型預測錯誤: {e}")
#                 p_flat = 0.75
#                 diagnosis = "高風險"

#         # 4) 存入資料庫
#         conn = sqlite3.connect('gait_results.db')
#         c = conn.cursor()
#         c.execute("INSERT INTO arch_results VALUES (?,?,?,?,?,?)",
#                   (datetime.now().isoformat(), p_flat, diagnosis, 
#                    "current_user", file.filename, len(X)))
#         conn.commit()
#         conn.close()

#         # 5) 回傳結果 (兼容 Swift 格式)
#         result_data = {
#             "timestamp": datetime.now().isoformat(),
#             "probability": p_flat,
#             "diagnosis": diagnosis
#         }

#         return {
#             "status": "success",
#             "results": [result_data],
#             "analysis_info": {
#                 "file": file.filename,
#                 "windows_count": len(X),
#                 "model_used": model_load_status["arch_model"]
#             }
#         }

#     except Exception as e:
#         error_msg = f"分析錯誤: {str(e)}"
#         print(error_msg)
#         traceback.print_exc()
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "message": error_msg}
#         )

# @app.get("/get_flatfoot_results/")
# async def get_flatfoot_results(user_id: str = "current_user", limit: int = 10):
#     """獲取歷史結果"""
#     try:
#         conn = sqlite3.connect('gait_results.db')
#         c = conn.cursor()
#         c.execute("SELECT timestamp, probability, diagnosis FROM arch_results WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", 
#                   (user_id, limit))
        
#         results = [{
#             "timestamp": r[0], 
#             "probability": r[1],
#             "diagnosis": r[2]
#         } for r in c.fetchall()]
        
#         conn.close()
        
#         return {
#             "status": "success", 
#             "results": results
#         }
    
#     except Exception as e:
#         return JSONResponse(
#             status_code=500,
#             content={"status": "error", "message": str(e)}
#         )

# # --- 啟動 ---
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     print(f"啟動服務在端口 {port}")
#     uvicorn.run(app, host="0.0.0.0", port=port, workers=1, timeout_keep_alive=60)
