from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_uplift_stub_or_real():
    r = client.post(
        "/uplift",
        json={
            "employee_features": {
                "pre_perf": 2.5,
                "tenure_yrs": 5.0,
                "role_level": 3,
                "dept": "R&D",
            },
            "training_id": "T-SKILL-001",
            "n_boot": 10,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert "cate" in body
    assert body["ci_lower"] <= body["cate"] <= body["ci_upper"] + 1e-9 or body["ci_upper"] >= body["ci_lower"]
    assert body["ci_level"] == 0.90


def test_panel_shape_deterministic():
    from training_roi.data import make_panel

    a = make_panel(n=200, seed=42)
    b = make_panel(n=200, seed=42)
    assert (a == b).all().all()
    assert {
        "employee_id", "training_id", "treatment", "pre_perf", "post_perf",
        "perf_uplift", "tenure_yrs", "role_level", "dept", "true_cate",
    } <= set(a.columns)
    assert a["treatment"].isin([0, 1]).all()
