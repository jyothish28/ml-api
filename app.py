from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Lung Disease Prediction API")

model = None

labels = ["normal", "benign", "malignant", "adenocarcio"]

suggestions = {
    "normal": "No issues detected. Maintain regular checkups.",
    "benign": "Benign lesion found. Consult doctor for monitoring.",
    "malignant": "Malignant tumor detected. Seek immediate medical attention.",
    "adenocarcio": "Adenocarcinoma detected. Contact oncologist urgently."
}


@app.on_event("startup")
def load_model():
    global model
    print("Loading model...")
    model = tf.keras.models.load_model("unet_model.h5", compile=False)
    print("Model loaded successfully")


@app.get("/")
def home():
    return {"message": "Lung Disease Prediction API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("L")
    image = image.resize((256,256))

    image = np.array(image) / 255.0
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