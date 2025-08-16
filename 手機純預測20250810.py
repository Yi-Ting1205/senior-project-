# -*- coding: utf-8 -*-
"""
V02 Inference from CSV — NO labels, NO training
===============================================
- Input: CSV files in DATA_DIR, each containing:
    Pred_result, Gyroscope_X, Gyroscope_Y, Gyroscope_Z,
    Acceleration_X, Acceleration_Y, Acceleration_Z
  (Pred_result 包含 'TO' 標記，用於 TO→TO 切段)

- Pipeline (identical to V02):
    TO→TO 切步 -> 每步 PAA=100 -> 5 步拼接成序列(500,6) -> norm="none"
    模型: enc_tcn_res_L + head_dense(..., 2)
    權重: 讀取 enc_best.weights.h5 與 arch_head_best.weights.h5

- Output:
    1) window_pred.csv : 每個 5-step 視窗的機率與預測
    2) file_pred.csv   : 每檔多數決與平均機率

!!! 僅推論；不包含任何訓練、標記處理或評估指標 !!!
"""

import os, glob, re, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models


# ------------------------ 0. 參數（請自行調整） ------------------------
# 資料夾：放 CSV 檔的目錄
DATA_DIR = r"C:\GITHUB\GaitHelper\CellphoneData(predict)"

   # <-- 修改成你的 CSV 目錄

# 權重目錄：必須含下列兩個檔案（取自 V02 訓練結果）
#   enc_best.weights.h5
#   arch_head_best.weights.h5
WEIGHT_DIR = r"C:\GITHUB\GaitHelper\weight"  # <-- 修改成你的 V02 權重目錄

# 輸出目錄（預設同權重目錄下的 "inference_out"）
OUT_DIR   = os.path.join(DATA_DIR, "inference_out")

# 固定比照訓練設定
PAA_IDX = 100
WIN_STEPS = 5
STRIDE_PAA_WIN = 2
FEAT_COLS = ["Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
             "Acceleration_X","Acceleration_Y","Acceleration_Z"]
EVENT_COL = "Pred_result"  # 需含 'TO' 記號
CLASSES   = ["normal(0)","flat(1)"]  # 僅作欄位命名用
# --------------------------------------------------------------------

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception:
        pass

# --------------------------- 1. helper funcs --------------------------
def banner(m):
    print("\n" + "="*70 + f"\n{m}\n" + "="*70)

def paa_fast(seg, M=100):
    """
    seg: (L, F) — 單步資料
    回傳: (M, F) — PAA 壓縮後
    """
    L, F = seg.shape
    idx = (np.linspace(0, L, M+1)).astype(int)
    # reduceat: 把 [idx[i]:idx[i+1]) 的區間做加總
    out = np.add.reduceat(seg, idx[:-1], axis=0)
    w   = np.maximum(np.diff(idx)[:, None], 1)
    return out / w

def five_step_windows(paa_list, win=5, stride=STRIDE_PAA_WIN):
    """
    paa_list: list of (100,6) 的步資料
    回傳: (N, 500, 6) 的視窗
    """
    n = len(paa_list)
    if n < win:
        return np.empty((0, win*PAA_IDX, len(FEAT_COLS)), dtype=np.float32)
    seqs = []
    for s in range(0, n - win + 1, stride):
        seq = np.concatenate(paa_list[s:s+win], axis=0)  # (win*100, 6)
        seqs.append(seq.astype(np.float32))
    return np.stack(seqs, axis=0) if seqs else np.empty((0, win*PAA_IDX, len(FEAT_COLS)), dtype=np.float32)

def load_csv_and_build_windows(csv_path):
    """
    讀單一 CSV -> 依 TO→TO 切步 -> 每步 PAA=100 -> 5步視窗 (500,6)
    norm = none（完全比照 V02）
    回傳:
      X_seq: (N, 500, 6)
      meta : 每個視窗對應的 (file, seq_idx, step_start, step_end)
    """
    df = pd.read_csv(csv_path)
    # 檢查欄位
    missing = [c for c in [EVENT_COL] + FEAT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{os.path.basename(csv_path)}] 缺少欄位: {missing}")

    # 找 TO 索引（與訓練版一致）
    ev = df[EVENT_COL].astype(str).str.upper().fillna("")
    to_idx = df.index[ev.eq("TO")].to_numpy()

    # 至少需要兩個 TO 才能形成一個步段
    paa_list = []
    step_spans = []  # 記錄每步的 (start_idx, end_idx)
    g = df[FEAT_COLS].to_numpy(dtype=np.float32)

    for i in range(len(to_idx) - 1):
        a, b = int(to_idx[i]), int(to_idx[i+1])
        if b <= a:
            continue
        step = g[a:b]                    # (L, 6)
        paa = paa_fast(step, PAA_IDX)    # (100, 6)
        paa_list.append(paa)
        step_spans.append((a, b))

    # 5步視窗
    X_seq = five_step_windows(paa_list, win=WIN_STEPS, stride=STRIDE_PAA_WIN)

    # 視窗 meta（僅輔助追蹤）
    meta = []
    if len(paa_list) >= WIN_STEPS:
        for s in range(0, len(paa_list) - WIN_STEPS + 1, STRIDE_PAA_WIN):
            # 以第一步與最後一步的原始索引範圍做摘要
            a0, _ = step_spans[s]
            _, b1 = step_spans[s + WIN_STEPS - 1]
            meta.append((os.path.basename(csv_path), len(meta), a0, b1))  # file, seq_idx, step_start, step_end

    return X_seq, meta

# --------------------------- 2. V02 模型結構 ---------------------------
def _tcn_block(x, f, d, drop):
    h = layers.Conv1D(f, 3, padding="causal", dilation_rate=d, activation="relu")(x)
    h = layers.BatchNormalization()(h)
    h = layers.SpatialDropout1D(drop)(h)
    if x.shape[-1] != f:
        x = layers.Conv1D(f, 1, padding="same")(x)
    return layers.Add()([x, h])

def make_enc_tcn_res_L(lat, drop, T, F):
    """
    與訓練版一致：
      dilation: 1,2,4,8,16,32，每層 128 filters
      GAP -> Dense(lat, relu)
    """
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
    """
    與訓練版一致：
      Dense(64, relu) -> Dropout(p) -> Dense(32, relu) -> Dropout(p) -> Dense(n, softmax)
    """
    inp = layers.Input((lat,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dropout(p)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(p)(x)
    out = layers.Dense(n, activation="softmax")(x)
    return models.Model(inp, out, name="ArchHead")

# --------------------------- 3. 推論主流程 ----------------------------
def main():
    banner("V02 CSV Inference — start")

    # 1) 彙整資料
    csv_list = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not csv_list:
        raise FileNotFoundError(f"找不到 CSV 檔，請確認 DATA_DIR: {DATA_DIR}")

    X_all = []
    meta_all = []
    file_ranges = []  # 每檔對應在 X_all 的範圍（start, end）
    total_seqs = 0

    for fp in csv_list:
        X_seq, meta = load_csv_and_build_windows(fp)
        n = len(X_seq)
        if n > 0:
            X_all.append(X_seq)
            meta_all.extend(meta)
            file_ranges.append((os.path.basename(fp), total_seqs, total_seqs + n))
            total_seqs += n
        else:
            # 即使沒有視窗，仍記錄範圍（空）
            file_ranges.append((os.path.basename(fp), total_seqs, total_seqs))

    if total_seqs == 0:
        banner("所有 CSV 檔都無足夠 TO→TO 步段，無法形成 5-step 視窗。結束。")
        return

    X_all = np.concatenate(X_all, axis=0)  # (N, 500, 6)
    T, F = X_all.shape[1], X_all.shape[2]
    assert T == WIN_STEPS * PAA_IDX and F == len(FEAT_COLS), "輸入維度不符 V02 設計"

    # 2) 構建 V02 模型並載入權重
    LAT_DIM = 512
    SpatialDropout = 0.2
    DenseDropout = 0.3

    infer_model = models.load_model(
        r"C:\GITHUB\GaitHelper\weight\V02_Infer.keras", compile=False
    )
    

    # 3) 推論
    banner(f"推論中…  共 {len(X_all)} 個 5-step 視窗")
    probs = infer_model.predict(X_all, batch_size=256, verbose=1)  # (N, 2)
    preds = probs.argmax(axis=1)  # 0/1

    # 4) 輸出
    os.makedirs(OUT_DIR, exist_ok=True)

    # 4-1) window_pred.csv
    win_rows = []
    for i, (fname, seq_idx, st, ed) in enumerate(meta_all):
        p0, p1 = float(probs[i, 0]), float(probs[i, 1])
        win_rows.append({
            "file": fname,
            "seq_idx": int(seq_idx),
            "step_start_idx": int(st),
            "step_end_idx": int(ed),
            "p_normal": p0,
            "p_flat": p1,
            "pred_class": int(1 if p1 >= p0 else 0)  # 與 argmax 等價，明列寫法
        })
    df_win = pd.DataFrame(win_rows)
    win_csv = os.path.join(OUT_DIR, "window_pred.csv")
    df_win.to_csv(win_csv, index=False)

    # 4-2) file_pred.csv（多數決 + 平均機率）
    file_rows = []
    for fname, a, b in file_ranges:
        if b > a:
            probs_slice = probs[a:b]
            preds_slice = preds[a:b].tolist()
            # 多數決（平手時取出現順序的最大）
            maj = max(set(preds_slice), key=preds_slice.count)
            p0_mean = float(probs_slice[:, 0].mean())
            p1_mean = float(probs_slice[:, 1].mean())
            file_rows.append({
                "file": fname,
                "n_windows": int(b - a),
                "p_normal_mean": p0_mean,
                "p_flat_mean": p1_mean,
                "pred_class_vote": int(maj)
            })
        else:
            file_rows.append({
                "file": fname,
                "n_windows": 0,
                "p_normal_mean": np.nan,
                "p_flat_mean": np.nan,
                "pred_class_vote": -1  # -1 代表無法判定（沒有視窗）
            })
    df_file = pd.DataFrame(file_rows)
    file_csv = os.path.join(OUT_DIR, "file_pred.csv")
    df_file.to_csv(file_csv, index=False)

    banner("完成 ✅")
    print(f"- 視窗預測：{win_csv}")
    print(f"- 檔案多數決：{file_csv}")

if __name__ == "__main__":
    main()
