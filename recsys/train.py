"""Offline training: builds features, fits XGBoost + MF, learns the blend, saves artifacts.

    python -m recsys.train --data-dir path/to/yelp/data --out artifacts
"""
import argparse
import json
import os
import pickle
import time
import zlib

import numpy as np
import xgboost as xgb

from . import features as F
from .mf import BiasMF
from .model import METRICS_FILE, MODEL_FILE, XGB_FILE, blend

XGB_PARAMS = dict(
    objective="reg:squarederror", max_depth=6, learning_rate=0.03, subsample=0.82,
    colsample_bytree=0.82, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", seed=42, verbosity=0,
)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def error_distribution(p, y):
    d = np.abs(np.asarray(p) - np.asarray(y))
    edges = [0, 1, 2, 3, 4, np.inf]
    labels = [">=0 and <1", ">=1 and <2", ">=2 and <3", ">=3 and <4", ">=4"]
    return {lab: int(((d >= lo) & (d < hi)).sum()) for lab, lo, hi in zip(labels, edges[:-1], edges[1:])}


def load_metadata(data_dir, keep_users, keep_biz):
    j = lambda name: os.path.join(data_dir, name)
    users = F.load_users(j("user.json"), keep_users)
    log(f"users loaded: {len(users)}")
    biz, info = F.load_businesses(j("business.json"))
    biz = {k: v for k, v in biz.items() if k in keep_biz}
    info = {k: v for k, v in info.items() if k in keep_biz}
    log(f"businesses loaded: {len(biz)}")
    rv_u, rv_b = F.load_review_aggs(j("review_train.json")) if os.path.exists(j("review_train.json")) else ({}, {})
    chk = F.load_checkins(j("checkin.json")) if os.path.exists(j("checkin.json")) else {}
    tip_u, tip_b = F.load_tips(j("tip.json")) if os.path.exists(j("tip.json")) else ({}, {})
    pht = F.load_photos(j("photo.json")) if os.path.exists(j("photo.json")) else {}
    log("review / checkin / tip / photo aggregates loaded")
    meta = {
        "users": users, "biz": biz,
        "rv_u": {k: v for k, v in rv_u.items() if k in keep_users},
        "rv_b": {k: v for k, v in rv_b.items() if k in keep_biz},
        "chk": {k: v for k, v in chk.items() if k in keep_biz},
        "tip_u": {k: v for k, v in tip_u.items() if k in keep_users},
        "tip_b": {k: v for k, v in tip_b.items() if k in keep_biz},
        "pht": {k: v for k, v in pht.items() if k in keep_biz},
    }
    return meta, info


def fit_models(rows, meta, n_trees, mf_epochs, n_jobs):
    g = float(np.mean([r for _, _, r in rows]))
    u_stats, b_stats = F.rating_stats(rows)
    fb = F.FeatureBuilder(meta, u_stats, b_stats, g)
    X = np.asarray([fb.train_row(u, b, r) for u, b, r in rows], dtype=np.float32)
    y = np.asarray([r for _, _, r in rows], dtype=np.float32)
    log(f"feature matrix {X.shape}")
    booster = xgb.train({**XGB_PARAMS, "nthread": n_jobs}, xgb.DMatrix(X, label=y,
                        feature_names=list(F.FEATURE_NAMES)), num_boost_round=n_trees)
    log(f"xgboost trained ({n_trees} trees)")
    mf = BiasMF(n_epochs=mf_epochs).fit(rows, g)
    log(f"matrix factorization trained ({mf_epochs} epochs)")
    return g, u_stats, b_stats, fb, booster, mf


def component_preds(pairs, fb, booster, mf):
    rows, bayes = zip(*(fb.predict_row(u, b) for u, b in pairs))
    X = np.asarray(rows, dtype=np.float32)
    return (booster.inplace_predict(X).astype(np.float64), mf.predict_many(pairs), np.asarray(bayes))


def fit_ridge(cols, y, l2=1.0):
    A = np.column_stack([np.ones(len(y))] + list(cols))
    reg = l2 * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def train(data_dir, out_dir, n_trees=520, mf_epochs=10, n_jobs=-1, holdout_mod=10):
    train_rows = F.read_ratings(os.path.join(data_dir, "yelp_train.csv"))
    val_path = os.path.join(data_dir, "yelp_val.csv")
    val_rows = F.read_ratings(val_path) if os.path.exists(val_path) else []
    log(f"ratings: train={len(train_rows)} val={len(val_rows)}")

    keep_users = {u for u, _, _ in train_rows} | {u for u, _, _ in val_rows}
    keep_biz = {b for _, b, _ in train_rows} | {b for _, b, _ in val_rows}
    meta, info = load_metadata(data_dir, keep_users, keep_biz)

    # 1) Learn blend weights on an internal holdout carved from train only.
    is_hold = [zlib.crc32(f"{u}||{b}".encode()) % holdout_mod == 0 for u, b, _ in train_rows]
    fit_rows = [r for r, h in zip(train_rows, is_hold) if not h]
    hold_rows = [r for r, h in zip(train_rows, is_hold) if h]
    log(f"blend calibration: fit={len(fit_rows)} holdout={len(hold_rows)}")
    _, _, _, fb, booster, mf = fit_models(fit_rows, meta, n_trees, mf_epochs, n_jobs)
    xp, mp, bp = component_preds([(u, b) for u, b, _ in hold_rows], fb, booster, mf)
    y_hold = np.array([r for _, _, r in hold_rows])
    weights = fit_ridge([xp, mp, bp], y_hold)
    hold_rmse = rmse(blend(weights, xp, mp, bp), y_hold)
    log(f"blend weights [bias, xgb, mf, bayes] = {np.round(weights, 4).tolist()}  holdout RMSE={hold_rmse:.4f}")

    # 2) Refit everything on the full training set.
    g, u_stats, b_stats, fb, booster, mf = fit_models(train_rows, meta, n_trees, mf_epochs, n_jobs)

    metrics = {"n_train": len(train_rows), "n_features": F.N_FEATURES, "n_trees": n_trees,
               "blend_weights": {"bias": weights[0], "xgboost": weights[1], "mf": weights[2], "bayes": weights[3]},
               "internal_holdout_rmse": hold_rmse}
    if val_rows:
        xp, mp, bp = component_preds([(u, b) for u, b, _ in val_rows], fb, booster, mf)
        y_val = np.array([r for _, _, r in val_rows])
        final = blend(weights, xp, mp, bp)
        metrics["validation"] = {
            "n": len(val_rows),
            "rmse": {"hybrid": rmse(final, y_val), "xgboost": rmse(np.clip(xp, 1, 5), y_val),
                     "matrix_factorization": rmse(mp, y_val), "bayesian_baseline": rmse(bp, y_val),
                     "global_mean": rmse(np.full(len(y_val), g), y_val)},
            "error_distribution": error_distribution(final, y_val),
        }
        log(f"validation RMSE: {json.dumps(metrics['validation']['rmse'], indent=None)}")

    user_rated = {}
    for u, b, _ in train_rows:
        user_rated.setdefault(u, set()).add(b)

    os.makedirs(out_dir, exist_ok=True)
    bundle = {"meta": meta, "biz_info": info, "u_stats": u_stats, "b_stats": b_stats,
              "global_avg": g, "mf": mf, "blend_weights": weights.tolist(), "user_rated": user_rated}
    with open(os.path.join(out_dir, MODEL_FILE), "wb") as fh:
        pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
    booster.save_model(os.path.join(out_dir, XGB_FILE))
    with open(os.path.join(out_dir, METRICS_FILE), "w") as fh:
        json.dump(metrics, fh, indent=2, default=float)
    log(f"artifacts written to {out_dir}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder with yelp_train.csv, yelp_val.csv and the Yelp JSON files")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--n-trees", type=int, default=520)
    ap.add_argument("--mf-epochs", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=-1)
    a = ap.parse_args()
    train(a.data_dir, a.out, a.n_trees, a.mf_epochs, a.n_jobs)


if __name__ == "__main__":
    main()
