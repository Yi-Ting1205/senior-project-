
# 這份ｏｋ但版本衝突
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from datetime import datetime
import sqlite3
import os
import pandas as pd
import io
from typing import List, Tuple
import traceback

# --- 單一 FastAPI 實例 ---
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

# --- 參數設定 ---
PAA_IDX = 100
WIN_STEPS = 5
STRIDE_PAA_WIN = 2
FEAT_COLS = ["Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z",
             "Acceleration_X", "Acceleration_Y", "Acceleration_Z"]

# --- 初始化資料庫 ---
def init_db():
    conn = sqlite3.connect('gait_results.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS gait_events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT, event_type TEXT, event_time REAL,
                  file_name TEXT, user_id TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS flatfoot_results
                 (timestamp TEXT, probability REAL, diagnosis TEXT, user_id TEXT, 
                  n_windows INTEGER, p_normal_mean REAL, model_source TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 載入兩個模型 ---
gait_model = None
flatfoot_model = None
gait_model_loaded = False
flatfoot_model_loaded = False

print("=" * 60)
print("開始載入模型...")


# ... [前面的導入和其他程式碼保持不變] ...

print("=" * 60)
print("開始載入模型...")

# 載入步態檢測模型 (gait_model_5.keras)
try:
    if os.path.exists("gait_model_5.keras"):
        print("嘗試載入 gait_model_5.keras...")
        # 嘗試不同的載入方式
        try:
            # 方式1: 使用 tf.keras.models.load_model
            gait_model = tf.keras.models.load_model("gait_model_5.keras", compile=False)
            print("✅ Gait 模型通過 tf.keras.models.load_model 載入成功")
        except Exception as e1:
            print(f"方式1失敗: {e1}")
            try:
                # 方式2: 使用 load_model 並指定 custom_objects
                gait_model = load_model("gait_model_5.keras", compile=False)
                print("✅ Gait 模型通過 keras.load_model 載入成功")
            except Exception as e2:
                print(f"方式2失敗: {e2}")
                try:
                    # 方式3: 嘗試使用安全模式載入
                    gait_model = tf.keras.models.load_model("gait_model_5.keras", compile=False, safe_mode=False)
                    print("✅ Gait 模型通過安全模式載入成功")
                except Exception as e3:
                    print(f"方式3失敗: {e3}")
                    raise e3
        
        gait_model_loaded = True
        print(f"Gait 模型輸入形狀: {gait_model.input_shape}")
        print(f"Gait 模型輸出形狀: {gait_model.output_shape}")
    else:
        print("❌ 找不到 gait_model_5.keras")
        # 檢查是否有 .h5 檔案可以轉換
        if os.path.exists("gait_model_5.h5"):
            print("發現 gait_model_5.h5，嘗試轉換...")
            try:
                convert_h5_to_keras("gait_model_5.h5", "gait_model_5.keras")
                gait_model = tf.keras.models.load_model("gait_model_5.keras", compile=False)
                gait_model_loaded = True
                print("✅ H5 轉換並載入成功")
            except Exception as conv_e:
                print(f"❌ 轉換失敗: {conv_e}")
except Exception as e:
    print(f"❌ 載入 Gait 模型失敗: {e}")
    print("詳細錯誤信息:")
    traceback.print_exc()

# 載入扁平足模型 (V02_Infer.keras)
try:
    if os.path.exists("V02_Infer.keras"):
        flatfoot_model = load_model("V02_Infer.keras", compile=False)
        flatfoot_model_loaded = True
        print("✅ Flatfoot 模型載入成功")
    else:
        print("❌ 找不到 V02_Infer.keras")
except Exception as e:
    print(f"❌ 載入 Flatfoot 模型失敗: {e}")

print(f"模型載入狀態: Gait={gait_model_loaded}, Flatfoot={flatfoot_model_loaded}")
print("=" * 60)

# 添加模型轉換函數
def convert_h5_to_keras(h5_path, keras_path):
    """將 H5 模型轉換為 Keras 格式"""
    print(f"正在轉換 {h5_path} 到 {keras_path}...")
    try:
        # 嘗試不同的載入方式
        try:
            model = tf.keras.models.load_model(h5_path, compile=False)
        except:
            # 如果直接載入失敗，嘗試使用 h5py 手動重建
            import h5py
            with h5py.File(h5_path, 'r') as f:
                model_config = f.attrs.get('model_config')
                if model_config is None:
                    raise ValueError("無法讀取模型配置")
                
                model_config = model_config.decode('utf-8') if isinstance(model_config, bytes) else model_config
                model = tf.keras.models.model_from_json(model_config)
                
                # 載入權重
                if 'model_weights' in f:
                    model.load_weights(h5_path)
        
        # 儲存為 Keras 格式
        model.save(keras_path, save_format='keras')
        print(f"✅ 轉換成功: {keras_path}")
        
    except Exception as e:
        print(f"❌ 轉換失敗: {e}")
        raise e

# ... [後面的程式碼保持不變] ...
# # 載入步態檢測模型 (gait_model_5.h5)
# try:
#     if os.path.exists("gait_model_5.keras"):
#         gait_model = load_model("gait_model_5.keras", compile=False)
#         gait_model_loaded = True
#         print("✅ Gait 模型載入成功")
#     else:
#         print("❌ 找不到 gait_model_5.keras")
# except Exception as e:
#     print(f"❌ 載入 Gait 模型失敗: {e}")

# # 載入扁平足模型 (V02_Infer.keras)
# try:
#     if os.path.exists("V02_Infer.keras"):
#         flatfoot_model = load_model("V02_Infer.keras", compile=False)
#         flatfoot_model_loaded = True
#         print("✅ Flatfoot 模型載入成功")
#     else:
#         print("❌ 找不到 V02_Infer.keras")
# except Exception as e:
    # print(f"❌ 載入 Flatfoot 模型失敗: {e}")

print(f"模型載入狀態: Gait={gait_model_loaded}, Flatfoot={flatfoot_model_loaded}")
print("=" * 60)

# --- 步態事件檢測函數 ---
def predict_gait_events(model, test_time, test_gyro, window_size=60, distance=40):
    """使用 gait_model_5.h5 檢測步態事件"""
    if not gait_model_loaded:
        return []
    
    x_pred = []
    pred_events = []

    for i in range(window_size, len(test_gyro) - window_size):
        window = test_gyro[i - window_size : i + window_size + 1].reshape(-1, 1) 
        x_pred.append(window)

    x_pred = np.array(x_pred)
    y_pred = model.predict(x_pred, verbose=0)
    pred_labels = np.argmax(y_pred, axis=1)
    
    last_event_idx = {"HS": -distance, "TO": -distance} 
    hs_distance_threshold = 30
    to_indices = []
    in_to_segment = False

    # 檢測 TO 事件
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
                    event_time = test_time[global_idx]
                    pred_events.append((event_time, "TO"))
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
                event_time = test_time[global_idx]
                pred_events.append((event_time, "TO"))
                to_indices.append(global_idx)
                last_event_idx["TO"] = global_idx

    # 檢測 HS 事件
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
            event_time = test_time[hs_global_idx]
            pred_events.append((event_time, "HS"))
            last_hs_idx = hs_global_idx
    
    return pred_events

# --- 扁平足分析輔助函數 ---
def paa_fast(seg: np.ndarray, M: int = 100) -> np.ndarray:
    """PAA 時間序列壓縮"""
    L, F = seg.shape
    idx = (np.linspace(0, L, M + 1)).astype(int)
    out = np.add.reduceat(seg, idx[:-1], axis=0)
    w = np.maximum(np.diff(idx)[:, None], 1)
    return out / w

def five_step_windows(paa_list: List[np.ndarray], win: int = 5, stride: int = 2) -> np.ndarray:
    """5步窗口拼接"""
    n = len(paa_list)
    if n < win:
        return np.empty((0, win * PAA_IDX, len(FEAT_COLS)), dtype=np.float32)
    
    seqs = []
    for s in range(0, n - win + 1, stride):
        seq = np.concatenate(paa_list[s:s + win], axis=0)
        seqs.append(seq.astype(np.float32))
    
    return np.stack(seqs, axis=0) if seqs else np.empty((0, win * PAA_IDX, len(FEAT_COLS)), dtype=np.float32)

def preprocess_flatfoot_data_from_csv(csv_content: str) -> Tuple[np.ndarray, int]:
    """從 CSV 內容預處理扁平足數據"""
    try:
        # 讀取 CSV
        df = pd.read_csv(io.StringIO(csv_content))
        
        # 檢查必要欄位
        required_cols = ["Time", "Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z",
                        "Acceleration_X", "Acceleration_Y", "Acceleration_Z"]
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 缺少欄位: {missing_cols}")
            return np.empty((0, WIN_STEPS * PAA_IDX, 6)), 0
        
        # 提取數據
        time_data = df["Time"].values
        gx = df["Gyroscope_X"].values.astype(np.float32)
        gy = df["Gyroscope_Y"].values.astype(np.float32)
        gz = df["Gyroscope_Z"].values.astype(np.float32)
        ax = df["Acceleration_X"].values.astype(np.float32)
        ay = df["Acceleration_Y"].values.astype(np.float32)
        az = df["Acceleration_Z"].values.astype(np.float32)
        
        print(f"數據長度: {len(time_data)} 個樣本")
        
        # 合併為特徵矩陣 (N, 6)
        features = np.column_stack([gx, gy, gz, ax, ay, az])
        
        # 使用 gait 模型檢測 TO 事件
        if gait_model_loaded:
            gait_events = predict_gait_events(gait_model, time_data, gz)
            to_indices = [int(event[0]) for event in gait_events if event[1] == "TO"]
            print(f"Gait 模型檢測到 {len(to_indices)} 個 TO 事件")
        else:
            print("❌ Gait 模型未載入，無法檢測 TO 事件")
            return np.empty((0, WIN_STEPS * PAA_IDX, 6)), 0
        
        if len(to_indices) < 2:
            print("❌ 不足的 TO 事件進行切步")
            return np.empty((0, WIN_STEPS * PAA_IDX, 6)), 0
        
        # TO->TO 切步
        paa_list = []
        for i in range(len(to_indices) - 1):
            start_idx = to_indices[i]
            end_idx = to_indices[i + 1]
            
            if end_idx <= start_idx or end_idx - start_idx < 10:
                continue
                
            step_data = features[start_idx:end_idx]
            if len(step_data) > 10:
                paa_step = paa_fast(step_data, PAA_IDX)
                paa_list.append(paa_step)
        
        print(f"成功切分 {len(paa_list)} 個步態周期")
        
        if len(paa_list) < WIN_STEPS:
            print(f"❌ 步態周期不足 {len(paa_list)} < {WIN_STEPS}")
            return np.empty((0, WIN_STEPS * PAA_IDX, 6)), 0
        
        # 5步窗口拼接
        X_windows = five_step_windows(paa_list, WIN_STEPS, STRIDE_PAA_WIN)
        print(f"生成 {len(X_windows)} 個分析窗口")
        
        return X_windows, len(paa_list)
        
    except Exception as e:
        print(f"預處理錯誤: {e}")
        traceback.print_exc()
        return np.empty((0, WIN_STEPS * PAA_IDX, 6)), 0

def predict_flatfoot(X_windows):
    """使用 flatfoot_model 進行預測"""
    if not flatfoot_model_loaded or len(X_windows) == 0:
        return None, None, None, "model_not_loaded"
    
    try:
        print(f"進行扁平足預測，輸入形狀: {X_windows.shape}")
        probs = flatfoot_model.predict(X_windows, verbose=0)
        print(f"預測成功，輸出形狀: {probs.shape}")
        
        p_flat_mean = float(probs[:, 1].mean())
        p_normal_mean = float(probs[:, 0].mean())
        pred_classes = probs.argmax(axis=1)
        majority_vote = int(np.bincount(pred_classes).argmax())
        
        if majority_vote == 1:
            diagnosis = "高風險" if p_flat_mean >= 0.7 else "中等風險"
        else:
            diagnosis = "正常"
            
        return p_flat_mean, p_normal_mean, diagnosis, "real_model"
            
    except Exception as e:
        print(f"❌ 預測失敗: {e}")
        traceback.print_exc()
        return None, None, None, "prediction_failed"

# --- API 端點 ---
@app.get("/")
def read_root():
    return {
        "status": "Gait & Flatfoot Analysis API", 
        "gait_model_loaded": gait_model_loaded,
        "flatfoot_model_loaded": flatfoot_model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...)):
    """檢測步態事件 (HS/TO) - 使用 gait_model_5.h5"""
    try:
        if not gait_model_loaded:
            return JSONResponse(status_code=503, content={"error": "Gait 模型未載入"})
        
        contents = await file.read()
        csv_content = contents.decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_content))

        if "Gyroscope_Z" not in df.columns or "Time" not in df.columns:
            return JSONResponse(status_code=400, content={"error": "缺少 'Time' 或 'Gyroscope_Z' 欄位"})

        test_time = df["Time"].values
        test_gyro = df["Gyroscope_Z"].values

        results = predict_gait_events(gait_model, test_time, test_gyro)

        # 存入資料庫
        conn = sqlite3.connect('gait_results.db')
        c = conn.cursor()
        for time, event in results:
            c.execute("INSERT INTO gait_events (timestamp, event_type, event_time, file_name, user_id) VALUES (?, ?, ?, ?, ?)",
                      (datetime.now().isoformat(), event, float(time), file.filename, "current_user"))
        conn.commit()
        conn.close()

        return {
            "status": "success", 
            "predictions": [{"time": float(t), "event": e} for t, e in results],
            "file": file.filename
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/analyze_flatfoot/")
async def analyze_flatfoot(file: UploadFile = File(...)):
    """分析扁平足風險 - 使用 CSV 文件"""
    try:
        print("收到扁平足分析請求")
        
        if not flatfoot_model_loaded:
            return JSONResponse(status_code=503, content={"error": "Flatfoot 模型未載入"})
        
        # 讀取 CSV 內容
        contents = await file.read()
        csv_content = contents.decode('utf-8')
        
        # 預處理數據
        X_windows, n_steps = preprocess_flatfoot_data_from_csv(csv_content)
        
        if len(X_windows) == 0:
            error_msg = "無法從數據中提取有效的步態窗口"
            return JSONResponse(status_code=400, content={"error": error_msg})

        # 進行預測
        p_flat_mean, p_normal_mean, diagnosis, model_source = predict_flatfoot(X_windows)
        
        if p_flat_mean is None:
            return JSONResponse(status_code=500, content={"error": "預測失敗"})

        # 存入資料庫
        conn = sqlite3.connect('gait_results.db')
        c = conn.cursor()
        c.execute("INSERT INTO flatfoot_results VALUES (?,?,?,?,?,?,?)",
                  (datetime.now().isoformat(), p_flat_mean, diagnosis, 
                   "current_user", len(X_windows), p_normal_mean, model_source))
        conn.commit()
        conn.close()

        # 返回結果
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "probability": round(p_flat_mean, 4),
            "diagnosis": diagnosis,
            "confidence": round(p_normal_mean, 4)
        }
        
        response = {
            "status": "success",
            "results": [result_data],
            "analysis_info": {
                "detected_steps": n_steps,
                "valid_windows": len(X_windows),
                "model_source": model_source
            }
        }
        
        return response

    except Exception as e:
        error_msg = f"分析過程中發生錯誤: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": error_msg})

@app.get("/get_gait_results/")
async def get_gait_results(limit: int = 10):
    """獲取步態事件結果"""
    try:
        conn = sqlite3.connect('gait_results.db')
        c = conn.cursor()
        c.execute("SELECT timestamp, event_type, event_time, file_name FROM gait_events ORDER BY timestamp DESC LIMIT ?", (limit,))
        
        results = [{
            "timestamp": r[0], 
            "event_type": r[1],
            "event_time": r[2],
            "file_name": r[3]
        } for r in c.fetchall()]
        
        conn.close()
        
        return {"status": "success", "results": results}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/get_flatfoot_results/")
async def get_flatfoot_results(limit: int = 10):
    """獲取扁平足分析結果"""
    try:
        conn = sqlite3.connect('gait_results.db')
        c = conn.cursor()
        c.execute("SELECT timestamp, probability, diagnosis, n_windows FROM flatfoot_results ORDER BY timestamp DESC LIMIT ?", (limit,))
        
        results = [{
            "timestamp": r[0], 
            "probability": r[1],
            "diagnosis": r[2],
            "windows_analyzed": r[3]
        } for r in c.fetchall()]
        
        conn.close()
        
        return {"status": "success", "results": results}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "gait_model_loaded": gait_model_loaded,
        "flatfoot_model_loaded": flatfoot_model_loaded,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("Server:app", host="0.0.0.0", port=port, workers=1, timeout_keep_alive=60)


# 分隔線
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
