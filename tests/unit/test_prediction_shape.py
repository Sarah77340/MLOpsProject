import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "model/emotion_model.keras"

def test_prediction_shape():
    """Une prédiction sur un batch de 2 images renvoie (2, 7)."""
    model = load_model(MODEL_PATH)
    dummy_batch = np.random.rand(2, 48, 48, 1).astype("float32")
    preds = model.predict(dummy_batch, verbose=0)
    assert preds.shape == (2, 7)
