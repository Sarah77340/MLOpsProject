from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_predict_emotion_with_invalid_url():
    response = client.get("/predict_url", params={"image_url": "http://invalid-url/image.jpg"})
    assert response.status_code == 500
    assert "error" in response.json()