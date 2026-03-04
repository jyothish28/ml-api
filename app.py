from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import io
import keras
import h5py

app = FastAPI(title="Lung Disease Prediction API")

# Load model
model = keras.models.load_model("unet_model.h5", compile=False)

labels = ["normal", "benign", "malignant", "adenocarcio"]

suggestions = {
    "normal": "No issues detected. Maintain regular checkups.",
    "benign": "Benign lesion found. Consult doctor for monitoring.",
    "malignant": "Malignant tumor detected. Seek immediate medical attention.",
    "adenocarcio": "Adenocarcinoma detected. Contact oncologist urgently."
}

@app.get("/")
def home():
    return {"message": "Lung Disease Prediction API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        image = Image.open(io.BytesIO(contents)).convert("L")
        image = image.resize((256,256))

        image = np.array(image) / 255.0
        image = image.reshape(1,256,256,1)

        prediction = model.predict(image)

        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        disease = labels[index]

        return {
            "disease": disease,
            "confidence": confidence,
            "suggestion": suggestions[disease]
        }

    except Exception as e:
        return {"error": str(e)}