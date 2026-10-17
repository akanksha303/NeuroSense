"""
NeuroSense FastAPI backend.
 
Run:  uvicorn main:app --reload --port 8000
 
Files expected in Backend/models/:
  neurosense_1d_best.*     EEG 1D-CNN (Keras), sigmoid output = P(anomaly)
  neurosense_mri.tflite    MRI classifier (TFLite)
  eeg_scaler.pkl           OPTIONAL: scaler fitted on the EEG training data
 
Settings (env vars, all optional):
  MRI_CLASSES      comma list in the SAME order as training, default: glioma,meningioma,notumor,pituitary
  MRI_PREPROCESS   mobilenet (x/127.5-1) | rescale (x/255) | none   default: mobilenet
  CORS_ORIGINS     comma list, default: http://localhost:8501,http://localhost:5173
"""
import glob, io, os
from contextlib import asynccontextmanager
from typing import Optional
 
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
 
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE, "models"))
EEG_LEN = 178
MAX_MRI_BYTES = 10 * 1024 * 1024
MRI_CLASSES = [c.strip() for c in os.getenv("MRI_CLASSES", "glioma,meningioma,notumor,pituitary").split(",")]
MRI_PREPROCESS = os.getenv("MRI_PREPROCESS", "mobilenet")
state = {}
 
 
def load_tflite(path):
    try:
        from tflite_runtime.interpreter import Interpreter   # lightweight, if installed
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp
 
 
@asynccontextmanager
async def lifespan(app: FastAPI):
    eeg_paths = sorted(glob.glob(os.path.join(MODEL_DIR, "neurosense_1d_best.*")))
    if eeg_paths:
        import tensorflow as tf
        state["eeg"] = tf.keras.models.load_model(eeg_paths[0])
    else:
        state["eeg"] = None
 
    scaler_path = os.path.join(MODEL_DIR, "eeg_scaler.pkl")
    if os.path.exists(scaler_path):
        import joblib
        state["scaler"] = joblib.load(scaler_path)
    else:
        state["scaler"] = None
 
    mri_path = os.path.join(MODEL_DIR, "neurosense_mri.tflite")
    state["mri"] = load_tflite(mri_path) if os.path.exists(mri_path) else None
    yield
    state.clear()
 
 
app = FastAPI(title="NeuroSense API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:5173").split(","),
    allow_methods=["*"], allow_headers=["*"],
)
 
 
# ---------- preprocessing ----------
def parse_eeg(raw: bytes) -> np.ndarray:
    try:
        text = raw.decode("utf-8", errors="ignore")
        vals = [float(v) for v in text.replace(",", " ").replace(";", " ").split()]
    except ValueError:
        raise HTTPException(400, "EEG file must contain only numbers (CSV or TXT).")
    if len(vals) != EEG_LEN:
        raise HTTPException(400, f"EEG file has {len(vals)} values; expected {EEG_LEN}.")
    x = np.array(vals, dtype="float32").reshape(1, EEG_LEN)
    if state.get("scaler") is not None:
        x = state["scaler"].transform(x).astype("float32")
    # match whatever input shape the saved model wants: (178,) or (178, 1)
    shape = state["eeg"].input_shape if state.get("eeg") is not None else (None, EEG_LEN, 1)
    return x.reshape(1, EEG_LEN, 1) if len(shape) == 3 else x
 
 
def parse_mri(raw: bytes) -> np.ndarray:
    if len(raw) > MAX_MRI_BYTES:
        raise HTTPException(400, "MRI image is larger than 10 MB.")
    if state.get("mri") is None:
        raise HTTPException(503, "MRI model is not loaded on the server.")
    inp = state["mri"].get_input_details()[0]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((w, h))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "MRI file is not a valid image (use JPG or PNG).")
    x = np.asarray(img, dtype="float32")
    if MRI_PREPROCESS == "mobilenet":
        x = x / 127.5 - 1.0
    elif MRI_PREPROCESS == "rescale":
        x = x / 255.0
    return np.expand_dims(x, 0)
 
 
# ---------- inference ----------
def run_eeg(x: np.ndarray) -> dict:
    if state.get("eeg") is None:
        raise HTTPException(503, "EEG model is not loaded on the server.")
    p = np.asarray(state["eeg"].predict(x, verbose=0)).ravel()
    return {"probability": float(p[0])}
 
 
def run_mri(x: np.ndarray) -> dict:
    interp = state["mri"]
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    if inp["dtype"] != np.float32:                       # quantised model: convert float -> int
        scale, zero = inp["quantization"]
        x = np.round(x / scale + zero).astype(inp["dtype"])
    interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
    interp.invoke()
    probs = interp.get_tensor(out["index"])[0].astype("float32")
    if out["dtype"] != np.float32:
        scale, zero = out["quantization"]
        probs = (probs - zero) * scale
    if abs(float(probs.sum()) - 1.0) > 0.05:             # logits -> softmax if needed
        e = np.exp(probs - probs.max())
        probs = e / e.sum()
    if len(probs) != len(MRI_CLASSES):
        raise HTTPException(500, f"Model has {len(probs)} outputs but MRI_CLASSES has {len(MRI_CLASSES)}.")
    by_class = {c: float(p) for c, p in zip(MRI_CLASSES, probs)}
    return {
        "label": MRI_CLASSES[int(np.argmax(probs))],
        "tumor_probability": 1.0 - by_class.get("notumor", 0.0),
        "probs": by_class,
    }
 
 
# ---------- routes ----------
@app.get("/")
def root():
    return {"service": "NeuroSense API", "docs": "/docs"}
 
 
@app.get("/health")
def health():
    return {"status": "ok", "eeg_loaded": state.get("eeg") is not None,
            "mri_loaded": state.get("mri") is not None, "eeg_scaler": state.get("scaler") is not None}
 
 
@app.post("/predict")
async def predict(
    eeg: Optional[UploadFile] = File(None),
    mri: Optional[UploadFile] = File(None),
    alpha: float = Form(0.5),
):
    if eeg is None and mri is None:
        raise HTTPException(400, "Upload an EEG file, an MRI image, or both.")
    if not 0.0 <= alpha <= 1.0:
        raise HTTPException(400, "alpha must be between 0 and 1.")
 
    out = {}
    if eeg is not None:
        out["eeg"] = run_eeg(parse_eeg(await eeg.read()))
    if mri is not None:
        out["mri"] = run_mri(parse_mri(await mri.read()))
 
    if "eeg" in out and "mri" in out:
        out["mode"], out["alpha"] = "fused", alpha
        out["final_risk"] = alpha * out["eeg"]["probability"] + (1 - alpha) * out["mri"]["tumor_probability"]
    elif "eeg" in out:
        out["mode"], out["final_risk"] = "eeg", out["eeg"]["probability"]
    else:
        out["mode"], out["final_risk"] = "mri", out["mri"]["tumor_probability"]
    return out