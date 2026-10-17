import os
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
io = __import__('io')

# Initialize FastAPI app
app = FastAPI(
    title="NeuroSense API",
    description="Multimodal Neurological Risk & Brain Tumor Classification Backend",
    version="1.0.0"
)

# Enable CORS for frontend integration (React / Streamlit / Vue)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EEG_MODEL_PATH = os.path.join(BASE_DIR, "models", "neurosense_1d_best.keras")
MRI_MODEL_PATH = os.path.join(BASE_DIR, "models", "neurosense_mri_best.keras")

eeg_model = None
mri_model = None

try:
    if os.path.exists(EEG_MODEL_PATH):
        eeg_model = tf.keras.models.load_model(EEG_MODEL_PATH)
        print("EEG 1D-CNN Model Loaded Successfully.")
    if os.path.exists(MRI_MODEL_PATH):
        mri_model = tf.keras.models.load_model(MRI_MODEL_PATH)
        print("MRI 2D-CNN Model Loaded Successfully.")
except Exception as e:
    print(f"Warning: Model loading failed: {e}")

# Class Labels for MRI
MRI_CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


@app.get("/")
def home():
    return {
        "status": "online",
        "system": "NeuroSense Clinical Intelligence Engine",
        "endpoints": ["/predict/eeg", "/predict/mri", "/predict/multimodal"]
    }


@app.post("/predict/eeg")
async def predict_eeg(file: UploadFile = File(...)):
    """
    Accepts a CSV file containing 178 EEG voltage time-step values 
    and returns functional anomaly risk.
    """
    if not eeg_model:
        raise HTTPException(status_code=500, detail="EEG model is not loaded on the server.")
    
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Take the first row of features (columns 1 to 178 or matching shape)
        # Assuming the CSV matches the 178 feature points
        features = df.iloc[0, 1:179].values.astype(float)
        
        # Reshape & scale
        data = features.reshape(1, -1)
        # Simple normalization standard check
        scaled_data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        final_input = np.expand_dims(scaled_data, axis=2)
        
        # Inference
        prob = float(eeg_model.predict(final_input)[0][0])
        risk = "High Risk / Tumor Region Anomaly Detected" if prob > 0.5 else "Normal / Healthy Baseline"
        confidence = round(float(prob if prob > 0.5 else 1 - prob) * 100, 2)
        
        return {
            "modality": "EEG",
            "prediction_score": prob,
            "classification": risk,
            "confidence_percentage": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing EEG file: {str(e)}")


@app.post("/predict/mri")
async def predict_mri(file: UploadFile = File(...)):
    """
    Accepts a brain scan image file (MRI) and returns structural tumor classification.
    """
    if not mri_model:
        raise HTTPException(status_code=500, detail="MRI model is not loaded on the server.")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Inference
        preds = mri_model.predict(img_array)[0]
        predicted_idx = int(np.argmax(preds))
        confidence = round(float(preds[predicted_idx]) * 100, 2)
        tumor_type = MRI_CLASSES[predicted_idx]
        
        return {
            "modality": "MRI",
            "predicted_class": tumor_type,
            "confidence_percentage": confidence,
            "all_probabilities": {MRI_CLASSES[i]: float(preds[i]) for i in range(len(MRI_CLASSES))}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing MRI image: {str(e)}")


@app.post("/predict/multimodal")
async def predict_multimodal(eeg_file: UploadFile = File(None), mri_file: UploadFile = File(None)):
    """
    Flexible router: Accepts EEG, MRI, or both. Executes late fusion if both are present.
    """
    if not eeg_file and not mri_file:
        raise HTTPException(status_code=400, detail="Please provide at least one file (EEG CSV or MRI Image).")
    
    response = {"modality": "Multimodal Fusion"}
    
    eeg_prob = None
    mri_result = None
    
    if eeg_file:
        eeg_res = await predict_eeg(eeg_file)
        response["eeg_analysis"] = eeg_res
        eeg_prob = eeg_res["prediction_score"]
        
    if mri_file:
        mri_res = await predict_mri(mri_file)
        response["mri_analysis"] = mri_res
        # Derive structural risk factor (e.g., 1 if tumor detected, 0 if No Tumor)
        mri_risk_score = 0.0 if mri_res["predicted_class"] == "No Tumor" else 1.0
        mri_result = mri_risk_score

    # Late Fusion Calculation if both are provided
    if eeg_prob is not None and mri_result is not None:
        alpha = 0.5  # Weight balance between functional (EEG) and structural (MRI)
        final_fusion_score = float((alpha * eeg_prob) + ((1 - alpha) * mri_result))
        response["unified_fusion_score"] = final_fusion_score
        response["unified_assessment"] = "High Clinical Risk / Anomaly Confirmed" if final_fusion_score > 0.5 else "Low Risk / Normal Findings"
    
    return response