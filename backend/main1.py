#updated
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import io
from PIL import Image
import mlflow.keras
import mlflow
#from mlflow.keras import load_model
from tensorflow.keras.models import load_model
import os

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Connexion à DagsHub (MLflow Tracking)
#os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Sarah77340/MLOpsProject.mlflow"
#mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Charger le modèle depuis DagsHub avec son run ID
#run_id = "TON_RUN_ID_DAGSHUB"
#model_uri = f"runs:/{run_id}/emotion_model"
#model = load_model(model_uri)

app = FastAPI()

# Charger le modèle depuis MLflow
#run_id = "2019468f04bc4cb9ad71cb57f69970c9"
#model_uri = f"runs:/{run_id}/emotion_model"
#model = mlflow.keras.load_model(model_uri)

model = load_model("model/emotion_model.keras")

# Labels FER2013
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Classifieur Haar pour détection de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        frame = np.array(image)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        predictions = []

        if len(faces) == 0:
            return JSONResponse(content={"predictions": [], "message": "Aucun visage détecté."})

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_normalized = roi_resized.astype("float") / 255.0
            roi_normalized = np.reshape(roi_normalized, (48, 48, 1))  # Pour avoir la bonne forme
            roi_array = img_to_array(roi_normalized)
            roi_expanded = np.expand_dims(roi_array, axis=0)

            preds = model.predict(roi_expanded, verbose=0)
            emotion_index = np.argmax(preds)
            emotion = emotion_labels[emotion_index]
            confidence = float(np.max(preds))

            predictions.append({
                "emotion": emotion,
                "confidence": round(confidence, 3),
                "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })

        return JSONResponse(content={"predictions": predictions})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

    # Sert le frontend statique
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

    @app.get("/")
    def read_index():
        return FileResponse("frontend/index.html")
