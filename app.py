from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io   # ✅ THIS WAS MISSING
import os

app = FastAPI(title="Lung Disease Prediction API")

# Load model once at startup
MODEL_PATH = "unet_model.h5"

print("Checking model file...")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found")

print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

print("Model loaded successfully")
# Class labels
labels = ["normal", "benign", "malignant", "adenocarcio"]

# Suggestions
suggestions = {
    "normal": "No issues detected. Maintain regular checkups.",
    "benign": "Benign lesion found. Consult with doctor for monitoring.",
    "malignant": "Malignant tumor detected. Seek immediate medical attention.",
    "adenocarcio": "Adenocarcinoma detected. Contact oncologist urgently."
}


# Home route
@app.get("/")
def home():
    return {"status": "API running successfully"}

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("L")
        image = image.resize((256,256))

        image = np.array(image)/255.0
        image = image.reshape(1,256,256,1)

        pred = model.predict(image)

        index = int(np.argmax(pred))
        confidence = float(np.max(pred))

        disease = labels[index]

        return {
            "disease": disease,
            "confidence": confidence,
            "suggestion": suggestions[disease]
        }

    except Exception as e:
        return {
            "error": str(e)
        }