from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_root_route_returns_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
