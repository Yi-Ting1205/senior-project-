from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = load_model("gait_model_5.h5")

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

@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # 檢查欄位
        if "Gyroscope_Z" not in df.columns or "Time" not in df.columns:
            return JSONResponse(status_code=400, content={"error": "缺少 'Time' 或 'Gyroscope_Z' 欄位"})

        test_time = df["Time"].values
        test_gyro = df["Gyroscope_Z"].values

        # 預測
        results = predict(model, test_time, test_gyro)

        return {"status": "success", "predictions": [{"time": t, "event": e} for t, e in results]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
