import pytest
from fastapi.testclient import TestClient



@pytest.fixture(scope="module")
def client(artifact_dir, monkeypatch_module):
    monkeypatch_module.setenv("ARTIFACT_DIR", str(artifact_dir))
    from starcast.api.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body["validation_rmse"]) >= {"hybrid", "xgboost"}
    assert "X-Process-Time-Ms" in r.headers


def test_predict_known_pair(client):
    r = client.get("/predict", params={"user_id": "u000", "business_id": "b000"})
    assert r.status_code == 200
    body = r.json()
    assert 1.0 <= body["predicted_stars"] <= 5.0
    assert body["known_user"] and body["known_business"]
    assert set(body["components"]) == {"xgboost", "matrix_factorization", "bayesian_baseline"}


def test_predict_cold_start_user(client):
    r = client.get("/predict", params={"user_id": "brand-new-user", "business_id": "b000"})
    assert r.status_code == 200
    assert r.json()["known_user"] is False


def test_predict_unknown_both_is_404(client):
    r = client.get("/predict", params={"user_id": "nobody", "business_id": "nowhere"})
    assert r.status_code == 404


def test_predict_requires_params(client):
    assert client.get("/predict", params={"user_id": "u000"}).status_code == 422


def test_recommend_excludes_rated_and_closed(client, artifact_dir):
    from starcast.models import HybridRecommender
    rated = HybridRecommender.load(str(artifact_dir)).user_rated["u000"]
    r = client.get("/recommend/u000", params={"k": 5})
    assert r.status_code == 200
    recs = r.json()["recommendations"]
    assert 0 < len(recs) <= 5
    ids = [x["business_id"] for x in recs]
    assert not set(ids) & rated
    assert "b001" not in ids  # closed business
    scores = [x["predicted_stars"] for x in recs]
    assert scores == sorted(scores, reverse=True)
    assert [x["rank"] for x in recs] == list(range(1, len(recs) + 1))


def test_recommend_filters(client):
    recs = client.get("/recommend/u001", params={"k": 50, "city": "las vegas", "category": "nightlife"}).json()["recommendations"]
    assert recs
    assert all(x["city"] == "Las Vegas" and "Nightlife" in x["categories"] for x in recs)


def test_recommend_cold_start_user(client):
    body = client.get("/recommend/someone-new", params={"k": 3}).json()
    assert body["known_user"] is False
    assert len(body["recommendations"]) == 3


def test_recommend_no_matches_returns_empty(client):
    body = client.get("/recommend/u000", params={"city": "Atlantis"}).json()
    assert body["recommendations"] == []


def test_recommend_validates_k(client):
    assert client.get("/recommend/u000", params={"k": 0}).status_code == 422


def test_recommend_defaults_to_home_state(client, artifact_dir):
    from starcast.models import HybridRecommender
    home = HybridRecommender.load(str(artifact_dir)).home_state("u000")
    body = client.get("/recommend/u000", params={"k": 50}).json()
    assert body["filters"]["state"] == home
    assert all(x["state"] == home for x in body["recommendations"])
    anywhere = client.get("/recommend/u000", params={"k": 50, "anywhere": True}).json()
    assert anywhere["filters"]["state"] is None
    assert {x["state"] for x in anywhere["recommendations"]} == {"AZ", "NV"}
