
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.tensorflow
import mlflow.keras


mlflow.set_experiment("emotion_detection")

# Configuration
DATA_PATH = "data/raw/fer2013.csv"
MODEL_PATH = "model/emotion_model"
#MODEL_NAME = "EmotionClassifier" 
ARTIFACT_PATH = "emotion_model" 

#updated
# Charger les données
print("Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
X = np.array([np.fromstring(pix, sep=' ') for pix in df['pixels']])
X = X.reshape(-1, 48, 48, 1) / 255.0
y = to_categorical(df['emotion'], num_classes=7)

# Split train/test
print("Séparation train/test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Définir le modèle CNN
print("Création du modèle...")
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entraînement avec MLFlow
print("Démarrage de l'entraînement avec MLFlow...")
mlflow.tensorflow.autolog()
with mlflow.start_run() as run:
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_acc)

    # Sauvegarde du modèle
    #print(f"Enregistrement du modèle dans {MODEL_PATH}...")
    #model.save(f"{MODEL_PATH}.keras")

    # Sauvegarde avec Daghub
    mlflow.keras.log_model(
        model,
        artifact_path=ARTIFACT_PATH
    )

    #print(f"Modele enregistré dans le registry sous le nom “{MODEL_NAME}”")
    print(f"-> Tracking URI : {mlflow.get_tracking_uri()}")
    print(f"-> Run ID : {run.info.run_id}")


print("Entraînement terminé.")
