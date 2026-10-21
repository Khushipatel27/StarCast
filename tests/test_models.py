import json
import random

import numpy as np

from starcast.models import BiasMF, HybridRecommender, blend, fit_ridge
from starcast.training.metrics import error_distribution, rmse


def test_fit_ridge_recovers_linear_weights():
    rng = np.random.RandomState(0)
    a, b, c = rng.rand(3, 500) * 4 + 1
    y = 0.5 + 0.6 * a + 0.3 * b + 0.1 * c
    w = fit_ridge([a, b, c], y, l2=1e-8)
    assert np.allclose(w, [0.5, 0.6, 0.3, 0.1], atol=1e-4)


def test_blend_clips_to_star_range():
    out = blend([0.0, 1.0, 1.0, 1.0], np.array([3.0, 0.1]), np.array([3.0, 0.1]), np.array([3.0, 0.1]))
    assert out.tolist() == [5.0, 1.0]


def test_mf_learns_user_and_item_biases():
    rng = random.Random(0)
    rows = [(f"u{u}", f"b{b}", 5.0 if b < 5 else 1.0) for u in range(30) for b in range(10) if rng.random() < 0.7]
    mf = BiasMF(n_epochs=30, lr=0.05).fit(rows, 3.0)
    assert mf.predict("u0", "b0") > mf.predict("u0", "b9")
    assert mf.score_all_items("u0").shape == (len(mf.b_map),)
    assert 1.0 <= mf.predict("unknown", "unknown") <= 5.0


def test_mf_state_round_trip():
    rows = [("u1", "b1", 4.0), ("u2", "b1", 2.0), ("u1", "b2", 5.0)]
    mf = BiasMF(n_epochs=2).fit(rows, 3.5)
    clone = BiasMF.from_state(mf.state_dict())
    assert clone.predict("u1", "b2") == mf.predict("u1", "b2")


def test_metrics():
    assert rmse([1, 2, 3], [1, 2, 5]) == np.sqrt(4 / 3)
    dist = error_distribution([1.0, 2.5, 5.0], [1.2, 1.0, 1.0])
    assert dist == {">=0 and <1": 1, ">=1 and <2": 1, ">=2 and <3": 0, ">=3 and <4": 0, ">=4": 1}


def test_training_writes_metrics(artifact_dir):
    metrics = json.loads((artifact_dir / "metrics.json").read_text())
    rmses = metrics["validation"]["rmse"]
    assert set(rmses) >= {"hybrid", "xgboost", "matrix_factorization", "bayesian_baseline"}
    assert rmses["hybrid"] < rmses["global_mean"]


def test_recommender_loads_artifacts(artifact_dir):
    m = HybridRecommender.load(str(artifact_dir))
    assert m.is_known_user("u000")
    assert m.home_state("u000") in {"AZ", "NV"}
    assert 1.0 <= m.predict("u000", "b000")["predicted_stars"] <= 5.0
