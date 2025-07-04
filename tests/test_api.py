import requests

# API URL
url = "http://localhost:8000/predict"

# Fichier image de test
with open("tests/test_image.jpg", "rb") as f:
    files = {'file': f}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())
