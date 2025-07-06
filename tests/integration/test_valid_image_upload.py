import requests

def test_predict_valid_image():
    url = "http://localhost:8000/predict"
    with open("tests/assets/test_image.jpg", "rb") as img:
        files = {"file": img}
        response = requests.post(url, files=files)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
