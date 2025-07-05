import pytest
from fastapi.testclient import TestClient
from backend.main0 import app

client = TestClient(app)

def test_predict_valid_image():
    with open("tests/test_image.jpg", "rb") as f:
        response = client.post("/predict", files={"file": ("test_image.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    data = response.json()
    print("JSON response : ", data)
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
