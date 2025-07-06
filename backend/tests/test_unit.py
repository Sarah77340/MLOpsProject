import pytest
from fastapi.testclient import TestClient
from backend.main import app
import numpy as np
import cv2
from PIL import Image
import io

client = TestClient(app)

def create_dummy_face_image():
    """Creates a grayscale image with a simple face-like pattern"""
    image = np.zeros((48, 48), dtype=np.uint8)

    # Draw a basic face-like pattern: circle head, eyes, and mouth
    cv2.circle(image, (24, 24), 20, 255, -1)  # head
    cv2.circle(image, (17, 18), 2, 0, -1)     # left eye
    cv2.circle(image, (31, 18), 2, 0, -1)     # right eye
    cv2.ellipse(image, (24, 30), (8, 4), 0, 0, 180, 0, -1)  # mouth

    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG")
    buf.seek(0)
    return buf

def test_predict_returns_prediction():
    img_bytes = create_dummy_face_image()
    files = {"file": ("face.jpg", img_bytes, "image/jpeg")}
    
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
