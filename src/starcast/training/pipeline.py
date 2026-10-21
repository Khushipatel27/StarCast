"""Offline training: builds features, fits XGBoost + MF, learns the blend, saves artifacts.

    starcast-train --data-dir data --out artifacts
"""
import argparse
import json
import os
import pickle
import time
import zlib

import numpy as np
import xgboost as xgb

from .. import config
from .. import data as D
from ..features import FEATURE_NAMES, N_FEATURES, FeatureBuilder, rating_stats
from ..models import BiasMF, blend, fit_ridge
from .metrics import error_distribution, rmse

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def load_metadata(data_dir, keep_users, keep_biz):
    """Loads every metadata source, keeping only users/businesses seen in the ratings."""
    path = lambda name: os.path.join(data_dir, name)
    optional = lambda name, loader, empty: loader(path(name)) if os.path.exists(path(name)) else empty

    users = D.load_users(path("user.json"), keep_users)
    log(f"users loaded: {len(users)}")
    biz, info = D.load_businesses(path("business.json"))
    biz = {k: v for k, v in biz.items() if k in keep_biz}
    info = {k: v for k, v in info.items() if k in keep_biz}
    log(f"businesses loaded: {len(biz)}")
    rv_u, rv_b = optional("review_train.json", D.load_review_aggs, ({}, {}))
    chk = optional("checkin.json", D.load_checkins, {})
    tip_u, tip_b = optional("tip.json", D.load_tips, ({}, {}))
    pht = optional("photo.json", D.load_photos, {})
    log("review / checkin / tip / photo aggregates loaded")

    only = lambda d, keep: {k: v for k, v in d.items() if k in keep}
    meta = {
        "users": users, "biz": biz,
        "rv_u": only(rv_u, keep_users), "rv_b": only(rv_b, keep_biz),
        "chk": only(chk, keep_biz),
        "tip_u": only(tip_u, keep_users), "tip_b": only(tip_b, keep_biz),
        "pht": only(pht, keep_biz),
    }
    return meta, info


def fit_models(rows, meta, n_trees, mf_params, n_jobs):
    g = float(np.mean([r for _, _, r in rows]))
    u_stats, b_stats = rating_stats(rows)
    fb = FeatureBuilder(meta, u_stats, b_stats, g)
    X = np.asarray([fb.train_row(u, b, r) for u, b, r in rows], dtype=np.float32)
    y = np.asarray([r for _, _, r in rows], dtype=np.float32)
    log(f"feature matrix {X.shape}")
    dtrain = xgb.DMatrix(X, label=y, feature_names=list(FEATURE_NAMES))
    booster = xgb.train({**config.XGB_PARAMS, "nthread": n_jobs}, dtrain, num_boost_round=n_trees)
    log(f"xgboost trained ({n_trees} trees)")
    mf = BiasMF(**mf_params).fit(rows, g)
    log(f"matrix factorization trained ({mf_params['n_epochs']} epochs)")
    return g, u_stats, b_stats, fb, booster, mf


def component_preds(pairs, fb, booster, mf):
    rows, bayes = zip(*(fb.predict_row(u, b) for u, b in pairs))
    X = np.asarray(rows, dtype=np.float32)
    return booster.inplace_predict(X).astype(np.float64), mf.predict_many(pairs), np.asarray(bayes)


def train(data_dir, out_dir, n_trees=config.N_TREES, mf_epochs=config.MF_PARAMS["n_epochs"],
          n_jobs=-1, holdout_mod=config.HOLDOUT_MOD):
    mf_params = {**config.MF_PARAMS, "n_epochs": mf_epochs}
    train_rows = D.read_ratings(os.path.join(data_dir, "yelp_train.csv"))
    val_path = os.path.join(data_dir, "yelp_val.csv")
    val_rows = D.read_ratings(val_path) if os.path.exists(val_path) else []
    log(f"ratings: train={len(train_rows)} val={len(val_rows)}")

    keep_users = {u for u, _, _ in train_rows} | {u for u, _, _ in val_rows}
    keep_biz = {b for _, b, _ in train_rows} | {b for _, b, _ in val_rows}
    meta, info = load_metadata(data_dir, keep_users, keep_biz)

    # 1) Learn blend weights on an internal holdout carved from train only.
    is_hold = [zlib.crc32(f"{u}||{b}".encode()) % holdout_mod == 0 for u, b, _ in train_rows]
    fit_rows = [r for r, h in zip(train_rows, is_hold) if not h]
    hold_rows = [r for r, h in zip(train_rows, is_hold) if h]
    log(f"blend calibration: fit={len(fit_rows)} holdout={len(hold_rows)}")
    _, _, _, fb, booster, mf = fit_models(fit_rows, meta, n_trees, mf_params, n_jobs)
    xp, mp, bp = component_preds([(u, b) for u, b, _ in hold_rows], fb, booster, mf)
    y_hold = np.array([r for _, _, r in hold_rows])
    weights = fit_ridge([xp, mp, bp], y_hold, config.RIDGE_L2)
    hold_rmse = rmse(blend(weights, xp, mp, bp), y_hold)
    log(f"blend weights [bias, xgb, mf, bayes] = {np.round(weights, 4).tolist()}  holdout RMSE={hold_rmse:.4f}")

    # 2) Refit everything on the full training set.
    g, u_stats, b_stats, fb, booster, mf = fit_models(train_rows, meta, n_trees, mf_params, n_jobs)

    metrics = {
        "n_train": len(train_rows), "n_features": N_FEATURES, "n_trees": n_trees,
        "blend_weights": dict(zip(("bias", "xgboost", "mf", "bayes"), weights.tolist())),
        "internal_holdout_rmse": hold_rmse,
    }
    if val_rows:
        xp, mp, bp = component_preds([(u, b) for u, b, _ in val_rows], fb, booster, mf)
        y_val = np.array([r for _, _, r in val_rows])
        final = blend(weights, xp, mp, bp)
        metrics["validation"] = {
            "n": len(val_rows),
            "rmse": {
                "hybrid": rmse(final, y_val),
                "xgboost": rmse(np.clip(xp, 1, 5), y_val),
                "matrix_factorization": rmse(mp, y_val),
                "bayesian_baseline": rmse(bp, y_val),
                "global_mean": rmse(np.full(len(y_val), g), y_val),
            },
            "error_distribution": error_distribution(final, y_val),
        }
        log(f"validation RMSE: {json.dumps(metrics['validation']['rmse'])}")

    user_rated = {}
    for u, b, _ in train_rows:
        user_rated.setdefault(u, set()).add(b)

    os.makedirs(out_dir, exist_ok=True)
    bundle = {"meta": meta, "biz_info": info, "u_stats": u_stats, "b_stats": b_stats,
              "global_avg": g, "mf": mf.state_dict(), "blend_weights": weights.tolist(),
              "user_rated": user_rated}
    with open(os.path.join(out_dir, config.MODEL_FILE), "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    booster.save_model(os.path.join(out_dir, config.XGB_FILE))
    with open(os.path.join(out_dir, config.METRICS_FILE), "w") as fh:
        json.dump(metrics, fh, indent=2)
    log(f"artifacts written to {out_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data",
                    help="folder with yelp_train.csv, yelp_val.csv and the Yelp JSON files")
    ap.add_argument("--out", default=config.DEFAULT_ARTIFACT_DIR)
    ap.add_argument("--n-trees", type=int, default=config.N_TREES)
    ap.add_argument("--mf-epochs", type=int, default=config.MF_PARAMS["n_epochs"])
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    train(a.data_dir, a.out, a.n_trees, a.mf_epochs, a.n_jobs)


if __name__ == "__main__":
    main()
