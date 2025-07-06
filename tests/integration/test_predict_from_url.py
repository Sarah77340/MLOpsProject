import requests

def test_predict_url_valid():
    image_url = "https://previews.123rf.com/images/peopleimages12/peopleimages122301/peopleimages12230126139/197301608-sad-face-white-background-and-portrait-of-woman-isolated-in-studio-for-upset-depression-and.jpg"
    url = f"http://localhost:8000/predict_url?image_url={image_url}"
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data