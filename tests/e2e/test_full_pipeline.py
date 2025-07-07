import io
import threading
import time

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

# Importation tardive pour éviter les effets de bord au démarrage
from backend.main import app

client = TestClient(app)

def generate_dummy_face():
    """Construit une pseudo-image 48x48 avec un 'visage' simple."""
    img = np.zeros((48, 48, 3), dtype=np.uint8)
    img[16:32, 12:36] = 255  # carré blanc = visage
    return Image.fromarray(img)

def test_full_predict_route():
    """Pipeline complet : upload → /predict → JSON OK."""
    img = generate_dummy_face()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("dummy.png", buf, "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    # On vérifie juste que la structure clé existe
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
