import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# 常數設定
PAA_IDX = 100
WIN_STEPS = 5
STRIDE_PAA_WIN = 2
FEAT_COLS = ["Gyroscope_X", "Gyroscope_Y", "Gyroscope_Z",
             "Acceleration_X", "Acceleration_Y", "Acceleration_Z"]
LAT_DIM = 512
SpatialDropout = 0.2
DenseDropout = 0.3

# 模型載入函數
def load_model():
    """載入預訓練模型"""
    weight_dir = os.path.join(os.path.dirname(__file__), 'weights')
    
    # 構建模型
    enc_model = make_enc_tcn_res_L(LAT_DIM, SpatialDropout, WIN_STEPS * PAA_IDX, len(FEAT_COLS))
    head_model = head_dense(LAT_DIM, 2, DenseDropout)
    
    # 載入權重
    enc_weight_path = os.path.join(weight_dir, 'enc_best.weights.h5')
    head_weight_path = os.path.join(weight_dir, 'arch_head_best.weights.h5')
    
    enc_model.load_weights(enc_weight_path)
    head_model.load_weights(head_weight_path)
    
    # 建立推理模型
    inp = layers.Input((WIN_STEPS * PAA_IDX, len(FEAT_COLS)))
    features = enc_model(inp)
    probs = head_model(features)
    
    return models.Model(inp, probs)

# 模型架構函數（保持不變）
def _tcn_block(x, f, d, drop):
    h = layers.Conv1D(f, 3, padding="causal", dilation_rate=d, activation="relu")(x)
    h = layers.BatchNormalization()(h)
    h = layers.SpatialDropout1D(drop)(h)
    if x.shape[-1] != f:
        x = layers.Conv1D(f, 1, padding="same")(x)
    return layers.Add()([x, h])

def make_enc_tcn_res_L(lat, drop, T, F):
    inp = layers.Input((T, F))
    x = inp
    skips = []
    for d in (1, 2, 4, 8, 16, 32):
        x = _tcn_block(x, 128, d, drop)
        skips.append(x)
    x = layers.Activation("relu")(layers.Add()(skips))
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(lat, activation="relu")(x)
    return models.Model(inp, out, name="Enc_TCN_Res_L")

def head_dense(lat, n, p):
    inp = layers.Input((lat,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dropout(p)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(p)(x)
    out = layers.Dense(n, activation="softmax")(x)
    return models.Model(inp, out, name="ArchHead")

# 資料處理函數
def paa_fast(seg, M=100):
    L, F = seg.shape
    idx = (np.linspace(0, L, M+1)).astype(int)
    out = np.add.reduceat(seg, idx[:-1], axis=0)
    w = np.maximum(np.diff(idx)[:, None], 1)
    return out / w

def five_step_windows(paa_list, win=5, stride=STRIDE_PAA_WIN):
    n = len(paa_list)
    if n < win:
        return np.empty((0, win*PAA_IDX, len(FEAT_COLS)), dtype=np.float32)
    seqs = []
    for s in range(0, n - win + 1, stride):
        seq = np.concatenate(paa_list[s:s+win], axis=0)
        seqs.append(seq.astype(np.float32))
    return np.stack(seqs, axis=0) if seqs else np.empty((0, win*PAA_IDX, len(FEAT_COLS)), dtype=np.float32)

def detect_to_events(gyro_z_data, threshold=2.0):
    """
    從陀螺儀 Z 軸檢測 TO 事件
    """
    to_indices = []
    for i in range(1, len(gyro_z_data) - 1):
        if gyro_z_data[i] > threshold and gyro_z_data[i] > gyro_z_data[i-1] and gyro_z_data[i] > gyro_z_data[i+1]:
            to_indices.append(i)
    return to_indices

def build_windows_from_sensor_data(sensor_data):
    """
    從感測器數據建立視窗
    """
    # 創建 DataFrame 類似的結構
    accel_x = sensor_data.get('acceleration_x', [])
    accel_y = sensor_data.get('acceleration_y', [])
    accel_z = sensor_data.get('acceleration_z', [])
    gyro_x = sensor_data.get('gyroscope_x', [])
    gyro_y = sensor_data.get('gyroscope_y', [])
    gyro_z = sensor_data.get('gyroscope_z', [])
    
    n_samples = min(len(accel_x), len(accel_y), len(accel_z), 
                   len(gyro_x), len(gyro_y), len(gyro_z))
    
    if n_samples == 0:
        return np.empty((0, WIN_STEPS * PAA_IDX, len(FEAT_COLS)), dtype=np.float32), []
    
    # 檢測 TO 事件
    to_indices = detect_to_events(gyro_z)
    
    if len(to_indices) < 2:
        # 如果沒有足夠的 TO 事件，返回空結果
        return np.empty((0, WIN_STEPS * PAA_IDX, len(FEAT_COLS)), dtype=np.float32), []
    
    # 建立特徵矩陣
    features = np.column_stack([
        gyro_x[:n_samples], gyro_y[:n_samples], gyro_z[:n_samples],
        accel_x[:n_samples], accel_y[:n_samples], accel_z[:n_samples]
    ])
    
    paa_list = []
    step_spans = []
    
    # 根據 TO 事件切分步態
    for i in range(len(to_indices) - 1):
        start_idx = to_indices[i]
        end_idx = to_indices[i + 1]
        
        if end_idx <= start_idx:
            continue
            
        step_data = features[start_idx:end_idx]
        if len(step_data) < 10:  # 最小步長限制
            continue
            
        paa = paa_fast(step_data, PAA_IDX)
        paa_list.append(paa)
        step_spans.append((start_idx, end_idx))
    
    # 建立 5 步視窗
    X_seq = five_step_windows(paa_list, win=WIN_STEPS, stride=STRIDE_PAA_WIN)
    
    # 建立元數據
    meta = []
    if len(paa_list) >= WIN_STEPS:
        for s in range(0, len(paa_list) - WIN_STEPS + 1, STRIDE_PAA_WIN):
            start_idx, _ = step_spans[s]
            _, end_idx = step_spans[s + WIN_STEPS - 1]
            meta.append((s, start_idx, end_idx))
    
    return X_seq, meta

# 主要處理函數
def process_sensor_data(sensor_data, model):
    """處理感測器數據並回傳預測結果"""
    X_seq, meta = build_windows_from_sensor_data(sensor_data)
    
    if len(X_seq) == 0:
        return {
            "n_windows": 0,
            "predictions": [],
            "summary": {
                "p_normal_mean": None,
                "p_flat_mean": None,
                "pred_class_vote": -1
            }
        }
    
    # 進行預測
    probs = model.predict(X_seq, batch_size=256, verbose=0)
    preds = probs.argmax(axis=1)
    
    # 整理視窗層級預測
    window_predictions = []
    for i, (seq_idx, st, ed) in enumerate(meta):
        window_predictions.append({
            "window_index": int(seq_idx),
            "start_index": int(st),
            "end_index": int(ed),
            "p_normal": float(probs[i, 0]),
            "p_flat": float(probs[i, 1]),
            "prediction": int(preds[i]),
            "confidence": float(max(probs[i, 0], probs[i, 1]))
        })
    
    # 計算摘要統計
    p0_mean = float(probs[:, 0].mean())
    p1_mean = float(probs[:, 1].mean())
    maj = int(np.bincount(preds).argmax()) if len(preds) > 0 else -1
    
    return {
        "n_windows": len(X_seq),
        "predictions": window_predictions,
        "summary": {
            "p_normal_mean": p0_mean,
            "p_flat_mean": p1_mean,
            "pred_class_vote": maj,
            "confidence": float(max(p0_mean, p1_mean))
        }
    }