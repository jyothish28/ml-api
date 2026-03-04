from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi import HTTPException

app = FastAPI(title="Lung Disease Prediction API")

model = tf.keras.models.load_model(
    "unet_model.h5",
    compile=False
)

labels = ["normal", "benign", "malignant", "adenocarcio"]

suggestions = {
    "normal": "No issues detected. Maintain regular checkups.",
    "benign": "Benign lesion found. Consult with doctor for monitoring.",
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
    return {"message": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:

        contents = await file.read()

        print("Received image")
        print(image.shape)

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

    except Exception as e:
        print("Prediction error:", e)
        raise HTTPException(status_code=500, detail=str(e))