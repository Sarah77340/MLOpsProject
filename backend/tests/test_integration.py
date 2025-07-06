# backend/tests/test_integration.py

from fastapi.testclient import TestClient
from backend.main import app
import os

client = TestClient(app)

def test_predict_endpoint():
    test_image_path = "backend/tests/test_image.jpg"
    
    # Vérifie que le fichier existe
    assert os.path.exists(test_image_path), "Image de test manquante"

    with open(test_image_path, "rb") as image_file:
        response = client.post(
            "/predict",  # Change cette route si ton endpoint est différent
            files={"file": ("test_image.jpg", image_file, "image/jpeg")}
        )

    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
