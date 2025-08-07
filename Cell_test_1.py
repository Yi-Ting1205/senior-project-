import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
from config1 import get_cfg
import matplotlib.pyplot as plt
from itertools import zip_longest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Dense, Flatten, Bidirectional, LSTM, TimeDistributed
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from scipy.signal import find_peaks


cell_frequency = 60


def normalize(gyro):
    mean = np.mean(gyro)
    std = np.std(gyro)
    if std == 0:
        return np.zeros_like(gyro)
    normalize_gyro = (gyro - mean) / std
    return normalize_gyro

def predict(model, test_time, test_gyro, window_size=60, distance=40):
    x_pred = []
    pred_events = []

    for i in range(window_size, len(test_gyro) - window_size):
        window = test_gyro[i - window_size : i + window_size + 1 ].reshape(-1, 1) 
        x_pred.append(window)

    x_pred = np.array(x_pred)
    y_pred = model.predict( x_pred, verbose=0)
    pred_labels = np.argmax( y_pred, axis=1 )
    last_event_idx = {"HS": -distance, "TO": -distance} 
    hs_distance_threshold = 30
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


def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pred(df, test_gyro):
    time = df["Time"].values
    pred_events = df["Pred_result"].values
    pred_events = [str(e) if pd.notna(e) else "" for e in pred_events]

    plt.figure(figsize=(14, 5))
    plt.plot(time, test_gyro, label="Gyroscope_Z", color='black')

    hs_indices = [i for i, e in enumerate(pred_events) if e == "HS"]
    to_indices = [i for i, e in enumerate(pred_events) if e == "TO"]

    plt.scatter(time[hs_indices], test_gyro[hs_indices], color='blue', label='HS', marker='o')
    plt.scatter(time[to_indices], test_gyro[to_indices], color='red', label='TO', marker='x')

    plt.title("Original gz with event")
    plt.xlabel("Time")
    plt.ylabel("Gyroscope Z")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, labels=['TO', 'HS']):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, linecolor='gray', cbar=False)
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_event_labels(filter_gyro, event_labels, filter_time=None, cell_indices=None):
    plt.figure(figsize=(61, 5))

    if filter_time is None:
        x_axis = np.arange(len(filter_gyro)) 
    else:
        x_axis = filter_time

    plt.plot(x_axis, filter_gyro, label="Filtered Gyro", color='black', linewidth=1)
    idx_to = np.where(event_labels == 0)[0]
    idx_hs = np.where(event_labels == 1)[0]

    plt.scatter(x_axis[idx_to], filter_gyro[idx_to], color='red', s=10, label='Down')
    plt.scatter(x_axis[idx_hs], filter_gyro[idx_hs], color='blue', s=10, label='Up')

    if cell_indices is not None:
        cell_to = [idx for idx, lbl in cell_indices if lbl == 0]
        cell_hs = [idx for idx, lbl in cell_indices if lbl == 1]

        plt.scatter(x_axis[cell_to], filter_gyro[cell_to], color='orange', s=50, marker='x', label='Cell TO')
        plt.scatter(x_axis[cell_hs], filter_gyro[cell_hs], color='lime', s=50, marker='x', label='Cell HS')

    plt.title("Filtered Gyro with Event Labels (TO=red, HS=blue, Cell=green/red x)")
    plt.xlabel("Time" if filter_time is not None else "Index")
    plt.ylabel("Filtered Gyro Z")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_mae(true_times, pred_times):
    n = min(len(true_times), len(pred_times))
    errors = np.abs(np.array(true_times[:n]) - np.array(pred_times[:n]))
    return np.mean(errors) if n > 0 else None

def main():

    cfg = get_cfg()
    test_folder = cfg.Users.test_folder 
    test_filenames = cfg.Users.test_filenames
    true_folder = cfg.Users.Dot_folder
    true_filenames = [f.replace(".csv", "_matched.csv") for f in cfg.Users.cell_filenames]
    
    mae_results = []

    model = load_model("/Users/chensiou/Desktop/code/gait_model_5.h5")

    for sub_testfile, true_file in zip(test_filenames, true_filenames):

        test_file_path = os.path.join(test_folder, sub_testfile)
        true_file_path = os.path.join(true_folder, true_file)
        df_test = pd.read_csv(test_file_path) 
        df_true = pd.read_csv(true_file_path)
        test_time = df_test["Time"].values 
        test_gyro = df_test["Gyroscope_Z"].values 
        
        #test_normale_gyro = normalize(test_gyro)    
        pred_events = predict(model, test_time, test_gyro)
        df_out = pd.read_csv(os.path.join(test_folder, sub_testfile), dtype={"Time": float})
        df_out["Time"] = df_out["Time"] 
        df_out["Pred_result"] = ""

        for t, event in pred_events:
            t = float(t)
            idx_list = np.where(np.isclose(df_out["Time"].values, t, atol=1e-3))[0]
            if len(idx_list) > 0:
                idx = idx_list[0]
                df_out.at[idx, "Pred_result"] = event


        print(f"[DEBUG] test檔名: {sub_testfile}, true檔名: {true_file}")
        #print(f"[DEBUG] test檔案時間: {df_test['Time'].min()} ~ {df_test['Time'].max()}")
        #print(f"[DEBUG] true檔案時間: {df_true['Cell_Time'].min()} ~ {df_true['Cell_Time'].max()}")
        
        true_hs_times = df_true[df_true["Label"] == "HS"]["Cell_Time"].values
        true_to_times = df_true[df_true["Label"] == "TO"]["Cell_Time"].values
        pred_hs_times = np.array([t for t, e in pred_events if e == "HS"])
        pred_to_times = np.array([t for t, e in pred_events if e == "TO"])

        mae_hs = calculate_mae(true_hs_times, pred_hs_times)
        mae_to = calculate_mae(true_to_times, pred_to_times)
        #print(f"{sub_testfile} - HS事件時間MAE: {mae_hs}")
        #print(f"{sub_testfile} - TO事件時間MAE: {mae_to}")
        mae_results.append({
            'Filename': sub_testfile,
            'MAE_HS': round(mae_hs, 4),
            'MAE_TO': round(mae_to, 4)
        })

        #plot_pred(df_out, test_gyro)
        mae_df = pd.DataFrame(mae_results)
        output_path = os.path.join(test_folder, 'mae_summary.csv')
        save_path = os.path.join(test_folder, sub_testfile.replace(".csv", "_with_pred.csv"))
        mae_df.to_csv(output_path, index=False)
        df_out.to_csv(save_path, index=False)
        print(f"已儲存預測結果: {save_path}")
        print(f"所有檔案MAE數據已儲存於: {output_path}")

    
if __name__ == "__main__":
    main()