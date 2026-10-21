import pytest

from starcast.features import (BAYES_K, FEATURE_NAMES, N_FEATURES, FeatureBuilder, bayes_avg,
                               rating_stats)

EMPTY_META = {k: {} for k in ("users", "biz", "rv_u", "rv_b", "chk", "tip_u", "tip_b", "pht")}
ROWS = [("u1", "b1", 5.0), ("u1", "b2", 3.0), ("u1", "b3", 4.0), ("u2", "b1", 1.0)]
G = 3.25
U_ENC = FEATURE_NAMES.index("u_enc")
B_ENC = FEATURE_NAMES.index("b_enc")


@pytest.fixture
def fb():
    u_stats, b_stats = rating_stats(ROWS)
    return FeatureBuilder(EMPTY_META, u_stats, b_stats, G)


def test_feature_vector_matches_names(fb):
    feats, _ = fb.predict_row("u1", "b1")
    assert len(feats) == N_FEATURES == len(FEATURE_NAMES) == len(fb.train_row("u1", "b1", 5.0))


def test_rating_stats():
    u_stats, b_stats = rating_stats(ROWS)
    assert u_stats["u1"] == (12.0, 3, 50.0)
    assert b_stats["b1"] == (6.0, 2, 26.0)


def test_train_row_leaves_own_label_out(fb):
    # u1 rated b1=5, b2=3, b3=4; without the b1 row the user's mean is (3+4)/2.
    row = fb.train_row("u1", "b1", 5.0)
    assert row[U_ENC] == pytest.approx(3.5)
    # b1 was rated 5 (u1) and 1 (u2); without u1's row the business mean is 1.
    assert row[B_ENC] == pytest.approx(1.0)


def test_train_row_single_rating_falls_back_to_global(fb):
    row = fb.train_row("u2", "b1", 1.0)  # u2 has only this rating
    assert row[U_ENC] == pytest.approx(G)


def test_predict_row_uses_bayesian_smoothing(fb):
    feats, baseline = fb.predict_row("u1", "b1")
    assert feats[U_ENC] == pytest.approx((12.0 + BAYES_K * G) / (3 + BAYES_K))
    assert feats[B_ENC] == pytest.approx((6.0 + BAYES_K * G) / (2 + BAYES_K))
    assert baseline == pytest.approx(0.45 * feats[U_ENC] + 0.55 * feats[B_ENC])


def test_predict_row_cold_start_is_global_mean(fb):
    feats, baseline = fb.predict_row("new-user", "new-biz")
    assert feats[U_ENC] == feats[B_ENC] == pytest.approx(G)
    assert baseline == pytest.approx(G)
    assert feats[FEATURE_NAMES.index("u_cold")] == 1.0
    assert feats[FEATURE_NAMES.index("b_cold")] == 1.0


def test_bayes_avg_shrinks_toward_prior():
    assert bayes_avg(0.0, 0, 3.0) == 3.0
    assert 3.0 < bayes_avg(50.0, 10, 3.0) < 5.0
