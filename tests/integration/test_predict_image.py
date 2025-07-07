from fastapi.testclient import TestClient
from backend.main import app
from PIL import Image
import io

client = TestClient(app)

def test_predict_emotion_with_image():
    # Crée une fausse image 48x48
    image = Image.new("RGB", (48, 48), color="white")
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    files = {"file": ("assets/test_image.jpg", img_bytes, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
