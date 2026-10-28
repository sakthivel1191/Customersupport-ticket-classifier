from fastapi.testclient import TestClient
from src.main import app
def test_health():
    with TestClient(app) as client:
        response=client.get("/health")
    assert response.status_code==200
    assert response.json()["status"]=="ok"
    assert response.json()["models_loaded"] is True

def test_predictions():
    with TestClient(app) as client:
        response=client.post(
            "/predict",
            json={"text":"i eas charged twice"}
        )
    assert response.status_code==200
    result=response.json()
    assert result["category"] in [
        "Billing","Technicl","Account","Delivery"
    ]
    assert result["priority"] in ["Low","Medium","High"]
    assert 0<=result["confidence"]<=1
def test_empty_text():
    with TestClient(app) as client:
        response=client.post(
            "/predict",
            json={"text":" "}
        )
    assert response.status_code==422
def test_whitespace_text():
    with TestClient(app) as client:
        response=client.post(
            "/predict",
            json={"text":"   "}
        )
    assert response.status_code==422