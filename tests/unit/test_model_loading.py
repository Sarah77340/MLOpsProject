import os
from tensorflow.keras.models import load_model

MODEL_PATH = "model/emotion_model.keras"

def test_model_file_exists():
    """Le fichier .keras doit exister dans le repo."""
    assert os.path.isfile(MODEL_PATH), f"{MODEL_PATH} introuvable"

def test_model_loads():
    """Le modèle doit se charger sans erreur et avoir des couches."""
    model = load_model(MODEL_PATH)
    assert len(model.layers) > 0