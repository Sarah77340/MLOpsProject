import requests

def test_predict_invalid_image():
    url = "http://localhost:8000/predict"
    fake_file = {"file": ("fake.txt", b"This is not an image", "text/plain")}
    response = requests.post(url, files=fake_file)
    assert response.status_code == 500
    data = response.json()
    assert "error" in data