# Yelp Hybrid Recommender API

A rating-prediction and recommendation service for Yelp businesses. A hybrid model
(XGBoost on 73 engineered features + biased matrix factorization + Bayesian baseline,
combined with a learned ridge blend) is served through FastAPI and packaged with Docker.

Extends my DSCI 553 (USC) data-mining coursework from a batch Spark job into a deployable service.

## Results

Trained on 455,854 ratings (11.3K users, 24.7K businesses); evaluated on 142,044 held-out ratings.

| Model                          | Validation RMSE |
|--------------------------------|-----------------|
| Global mean                    | 1.1222          |
| Bayesian user/business average | 1.0374          |
| Biased matrix factorization    | 1.0072          |
| XGBoost (73 features)          | 0.9805          |
| **Hybrid blend**               | **0.9791**      |

72% of predictions land within 1 star of the true rating.

Serving latency (Docker, local, 50 requests): `/predict` p50 10 ms, `/recommend` p50 16 ms / p95 32 ms.
Model load time at startup: ~1.5 s. Image size: 615 MB.

## How it works

```
                 ┌──────────── offline (recsys/train.py) ────────────┐
 Yelp ratings ──►│ features: user/business metadata, review, tip,    │
 + 6 JSON files  │ check-in, photo aggregates; leave-one-out rating  │──► artifacts/
                 │ encodings for train rows, Bayesian-smoothed at    │    model.pkl
                 │ inference  ──►  XGBoost  +  biased MF  +  ridge   │    xgb.json
                 │ blend fit on a 10% train-only holdout             │    metrics.json
                 └───────────────────────────────────────────────────┘
                                         │
                 ┌──────────── online (app/main.py) ─────────────────┐
 GET /predict ──►│ build features ─► XGBoost, MF, baseline ─► blend  │
 GET /recommend ►│ stage 1: MF dot-product retrieval over all        │
                 │          eligible businesses ─► top 300           │
                 │ stage 2: re-rank candidates with the full hybrid  │
                 └───────────────────────────────────────────────────┘
```

- **Leakage control.** Training rows use leave-one-out user/business averages, so a row's own label is never in its features. At inference, averages are Bayesian-smoothed toward the global mean (k=10), so cold-start users and businesses get sensible scores.
- **Train-only blending.** Blend weights are learned on a hash-based 10% holdout of the training set, never on validation labels. All models are then refit on the full training set.
- **Two-stage recommendations.** Cheap MF scoring narrows ~25K businesses to 300 candidates. The expensive feature + XGBoost path runs only on those candidates.

## API

Interactive docs at `http://localhost:8000/docs` once running.

### `GET /predict?user_id=&business_id=`

```bash
curl "localhost:8000/predict?user_id=wf1GqnKQuvH-V3QN80UOOQ&business_id=fThrN4tfupIGetkrz18JOg"
```
```json
{
  "user_id": "wf1GqnKQuvH-V3QN80UOOQ",
  "business_id": "fThrN4tfupIGetkrz18JOg",
  "predicted_stars": 3.7577,
  "components": {"xgboost": 3.6892, "matrix_factorization": 4.1202, "bayesian_baseline": 3.9294},
  "known_user": true,
  "known_business": true
}
```
Unknown users or businesses get a cold-start prediction. If both are unknown, the endpoint returns 404.

### `GET /recommend/{user_id}`

| Param            | Default | Meaning                                                  |
|------------------|---------|----------------------------------------------------------|
| `k`              | 10      | number of results (1–100)                                |
| `city`           | –       | exact city, case-insensitive                             |
| `state`          | user's home state | two-letter code; home state = where the user rated most |
| `category`       | –       | substring of categories, e.g. `pizza`                    |
| `anywhere`       | false   | disable the home-state default                           |
| `include_closed` | false   | include closed businesses                                |
| `exclude_rated`  | true    | skip businesses the user already rated                   |

```bash
curl "localhost:8000/recommend/wf1GqnKQuvH-V3QN80UOOQ?k=3&city=Las%20Vegas&category=pizza"
```
```json
{
  "user_id": "wf1GqnKQuvH-V3QN80UOOQ",
  "known_user": true,
  "k": 3,
  "filters": {"city": "Las Vegas", "state": null, "category": "pizza", "include_closed": false, "exclude_rated": true},
  "recommendations": [
    {"rank": 1, "business_id": "e6m1p7oPbq7Gq37eMhSsWA", "name": "Joey's NY Pizza", "city": "Las Vegas",
     "state": "NV", "categories": "Pasta Shops, Sandwiches, Restaurants, Food, Pizza, ...",
     "yelp_stars": 4.5, "predicted_stars": 4.5628}
  ]
}
```

### `GET /health`
Returns status, model load time, catalog sizes, and validation metrics. Docker uses it as the healthcheck.

## Running it

### 1. Get the data
Download the [Yelp Open Dataset](https://www.yelp.com/dataset). Place `yelp_train.csv`, `yelp_val.csv` (format `user_id,business_id,stars`), `user.json`, `business.json`, `review_train.json`, `checkin.json`, `tip.json` and `photo.json` in one folder. The data is not redistributed in this repo.

### 2. Train (about 5 minutes on a laptop)
```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements-dev.txt
python -m recsys.train --data-dir path/to/data --out artifacts
```

### 3a. Serve with Docker
```bash
docker compose up --build -d
curl localhost:8000/health
```

### 3b. Or serve locally
```bash
uvicorn app.main:app --port 8000
```

### Tests
```bash
pytest
```
The suite generates a small synthetic Yelp-format dataset and trains a small model on it, then tests both endpoints end to end. It covers cold-start handling, filters, ranking order, exclusion of already-rated businesses, and input validation.

## Layout

```
recsys/features.py   data loaders + FeatureBuilder (LOO / Bayesian encodings)
recsys/mf.py         biased matrix factorization (SGD)
recsys/train.py      offline training, blend calibration, metrics, artifact export
recsys/model.py      HybridRecommender: predict + two-stage recommend
app/main.py          FastAPI app
tests/               synthetic-data end-to-end tests
```
