import requests

def test_e2e_predict():
    url = "http://localhost:8000/predict"
    with open("backend/tests/test_image.jpg", "rb") as f:
        files = {"file": ("test_image.jpg", f, "image/jpeg")}
        response = requests.post(url, files=files)

    print("Status Code:", response.status_code)
    data = response.json()
    print("Response JSON:", data)

    assert response.status_code == 200
    assert "predictions" in data
    assert len(data["predictions"]) > 0, "Aucun visage détecté dans l'image de test"

if __name__ == "__main__":
    test_e2e_predict()
    print("E2E test passed")
