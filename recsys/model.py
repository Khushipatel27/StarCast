"""Serving-side model: loads trained artifacts and answers predict/recommend."""
import json
import os
import pickle

import numpy as np
import xgboost as xgb

from .features import FeatureBuilder, MAX_STARS, MIN_STARS

MODEL_FILE = "model.pkl"
XGB_FILE = "xgb.json"
METRICS_FILE = "metrics.json"


def blend(weights, xgb_pred, mf_pred, bayes_pred):
    w = weights
    out = w[0] + w[1] * xgb_pred + w[2] * mf_pred + w[3] * bayes_pred
    return np.clip(out, MIN_STARS, MAX_STARS)


class HybridRecommender:
    """XGBoost + biased MF + Bayesian baseline, combined by a ridge blend.

    /recommend is two-stage: MF dot-products retrieve the top candidates out of
    every known business, then the full hybrid model re-ranks them.
    """

    def __init__(self, bundle, booster, metrics=None):
        self.meta = bundle["meta"]
        self.info = bundle["biz_info"]
        self.u_stats = bundle["u_stats"]
        self.b_stats = bundle["b_stats"]
        self.global_avg = bundle["global_avg"]
        self.mf = bundle["mf"]
        self.weights = np.asarray(bundle["blend_weights"])
        self.user_rated = bundle["user_rated"]
        self.booster = booster
        self.metrics = metrics or {}
        self.fb = FeatureBuilder(self.meta, self.u_stats, self.b_stats, self.global_avg)
        self.item_ids = np.array(sorted(self.mf.b_map, key=self.mf.b_map.get))
        empty = {"is_open": False, "city": "", "categories": ""}
        item_info = [self.info.get(b, empty) for b in self.item_ids]
        self.item_has_info = np.array([b in self.info for b in self.item_ids])
        self.item_open = np.array([i["is_open"] for i in item_info])
        self.item_city = np.array([i["city"].lower() for i in item_info])
        self.item_cats = [i["categories"].lower() for i in item_info]
        self.item_state = np.array([self.info.get(b, {}).get("state", "") for b in self.item_ids])

    @classmethod
    def load(cls, artifact_dir):
        with open(os.path.join(artifact_dir, MODEL_FILE), "rb") as fh:
            bundle = pickle.load(fh)
        booster = xgb.Booster()
        booster.load_model(os.path.join(artifact_dir, XGB_FILE))
        metrics = None
        mpath = os.path.join(artifact_dir, METRICS_FILE)
        if os.path.exists(mpath):
            with open(mpath) as fh:
                metrics = json.load(fh)
        return cls(bundle, booster, metrics)

    def is_known_user(self, uid):
        return uid in self.u_stats

    def is_known_business(self, bid):
        return bid in self.b_stats or bid in self.info

    def _score(self, pairs):
        rows, bayes = zip(*(self.fb.predict_row(u, b) for u, b in pairs))
        X = np.asarray(rows, dtype=np.float32)
        xgb_pred = self.booster.inplace_predict(X).astype(np.float64)
        mf_pred = self.mf.predict_many(pairs)
        bayes = np.asarray(bayes)
        return blend(self.weights, xgb_pred, mf_pred, bayes), xgb_pred, mf_pred, bayes

    def predict(self, uid, bid):
        final, xp, mp, bp = self._score([(uid, bid)])
        return {
            "predicted_stars": round(float(final[0]), 4),
            "components": {
                "xgboost": round(float(xp[0]), 4),
                "matrix_factorization": round(float(mp[0]), 4),
                "bayesian_baseline": round(float(bp[0]), 4),
            },
        }

    def home_state(self, uid):
        """State where the user has rated the most businesses (None if unknown)."""
        counts = {}
        for bid in self.user_rated.get(uid, ()):
            st = self.info.get(bid, {}).get("state")
            if st:
                counts[st] = counts.get(st, 0) + 1
        return max(counts, key=counts.get) if counts else None

    def recommend(self, uid, k=10, city=None, state=None, category=None, include_closed=False,
                  exclude_rated=True, n_candidates=300):
        seen = self.user_rated.get(uid, set()) if exclude_rated else set()
        city_l = city.lower() if city else None
        cat_l = category.lower() if category else None

        mask = self.item_has_info.copy()
        if not include_closed:
            mask &= self.item_open
        if city_l:
            mask &= self.item_city == city_l
        if state:
            mask &= self.item_state == state.upper()
        if cat_l:
            mask &= np.fromiter((cat_l in c for c in self.item_cats), bool, len(self.item_cats))
        for bid in seen:
            i = self.mf.b_map.get(bid)
            if i is not None:
                mask[i] = False
        if not mask.any():
            return []

        # Stage 1: MF retrieval over every eligible business.
        scores = self.mf.score_all_items(uid)
        scores = np.where(mask, scores, -np.inf)
        n = int(min(max(n_candidates, k), mask.sum()))
        cand_idx = np.argpartition(-scores, n - 1)[:n]
        candidates = self.item_ids[cand_idx].tolist()

        # Stage 2: re-rank with the full hybrid model.
        final, _, _, _ = self._score([(uid, b) for b in candidates])
        order = np.argsort(-final)[:k]
        out = []
        for rank, j in enumerate(order, 1):
            bid = candidates[j]
            inf = self.info[bid]
            out.append({
                "rank": rank,
                "business_id": bid,
                "name": inf["name"],
                "city": inf["city"],
                "state": inf["state"],
                "categories": inf["categories"],
                "yelp_stars": inf["stars"],
                "predicted_stars": round(float(final[j]), 4),
            })
        return out
