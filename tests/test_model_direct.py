from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import mlflow.keras

# Étiquettes des émotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Charger le modèle via MLflow
model = mlflow.keras.load_model("runs:/2019468f04bc4cb9ad71cb57f69970c9/emotion_model")

# Charger l'image
image_path = "tests/test_image.jpg"
image = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
image = image.resize((48, 48))               # Redimensionner comme le modèle attend
image_array = np.array(image).astype("float") / 255.0
image_array = img_to_array(image_array)
image_array = np.expand_dims(image_array, axis=0)

# Prédiction
preds = model.predict(image_array)
print("Prédictions brutes :", preds)
print("Émotion prédite :", emotion_labels[np.argmax(preds)])
