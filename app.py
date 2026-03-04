from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io   # ✅ THIS WAS MISSING

app = FastAPI(title="Lung Disease Prediction API")

# Load model once at startup
model = tf.keras.models.load_model("unet_model.h5")

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
    return {"message": "Lung Disease Prediction API is running"}


# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read image
        contents = await file.read()

        # Convert to grayscale
        image = Image.open(io.BytesIO(contents)).convert("L")

        # Resize
        image = image.resize((256,256))

        # Convert to numpy
        image = np.array(image)

        # Normalize
        image = image / 255.0

        # Reshape to model input
        image = image.reshape(1,256,256,1)

        # Predict
        pred = model.predict(image)

        index = int(np.argmax(pred))

        confidence = float(np.max(pred))

        disease = labels[index]

        suggestion = suggestions[disease]

        return {

            "disease": disease,

            "confidence": confidence,

            "suggestion": suggestion

        }

    except Exception as e:

        return {

            "error": str(e)

        }