
from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from pathlib import Path

app = FastAPI(title="NeuroSense API")

# Load the already-trained model
MODEL_PATH = Path(__file__).parent / "models" / "neurosense_1d_best.keras"

model = None

if MODEL_PATH.exists():
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


@app.get("/")
def home():
    return {"message": "Welcome to NeuroSense API"}


@app.get("/health")
def health():
    return {"model_loaded": model is not None}


@app.post("/predict")
async def predict(eeg: UploadFile = File(...)):

    if model is None:
        raise HTTPException(503, "Model not loaded")

    # Read the uploaded EEG file
    content = await eeg.read()

    try:
        values = np.fromstring(
            content.decode("utf-8").replace(",", " "),
            sep=" ",
            dtype=np.float32
        )
    except (UnicodeDecodeError, ValueError):
        raise HTTPException(400, "Invalid EEG file")

    # Check the number of EEG values
    if len(values) != 178:
        raise HTTPException(
            400, "EEG file must contain 178 values"
        )

    if not np.isfinite(values).all():
        raise HTTPException(400, "EEG values must be valid numbers")

    # Prepare input for the model
    if len(model.input_shape) == 3:
        values = values.reshape(1, 178, 1)
    else:
        values = values.reshape(1, 178)

    # Generate prediction
    probability = float(
        model.predict(values, verbose=0).flatten()[0]
    )

    # Assume 1 = anomaly and 0 = normal
    prediction = "Anomaly" if probability >= 0.5 else "Normal"

    return {
        "prediction": prediction,
        "anomaly_probability": round(probability, 4)
    }


