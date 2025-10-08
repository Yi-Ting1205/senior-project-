from fastapi import FastAPI, UploadFile, File
from flatfoot_model import model_flat, predict_flatfoot
import pandas as pd
import io

app = FastAPI()

@app.post("/predict/flatfoot/")
async def flatfoot_predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    result = predict_flatfoot(model_flat, df)
    return result
