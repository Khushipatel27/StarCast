# Yelp Hybrid Recommender API

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688)
![XGBoost](https://img.shields.io/badge/XGBoost-hybrid%20model-orange)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![Tests](https://img.shields.io/badge/tests-13%20passing-brightgreen)

A rating-prediction and recommendation service for Yelp businesses. It predicts how many stars a user would give a business and returns a personalized top-K list of businesses for any user.

Under the hood, a **hybrid model** combines three predictors with a learned ridge blend:

- **XGBoost** on 73 engineered features
- **Biased matrix factorization**
- **Bayesian-smoothed baseline**

It is served through **FastAPI** and packaged with **Docker**.

> This project extends my DSCI 553 (Foundations and Applications of Data Mining, USC) coursework. I turned a batch Spark job into a deployable service with a two-stage recommendation endpoint, tests, and containerization.

---

## Highlights

- **0.9791 RMSE** on 142,044 held-out ratings: 13% better than a global-mean baseline and 5.6% better than Bayesian user/business averages.
- **Two-stage recommendation.** Matrix factorization retrieves candidates from about 25K businesses, then the full hybrid model re-ranks them.
- **Fast serving.** `/predict` has a 10 ms median latency, and `/recommend` has a 16 ms median and a 32 ms 95th percentile, both measured in Docker.
- **Leakage-safe training.** Training rows use leave-one-out target encoding, and blend weights are learned only on a holdout taken from the training set.
- **Cold-start handling.** Unknown users and businesses still get sensible scores through Bayesian smoothing toward the global mean.
- **Production basics.** The service has typed request/response schemas, input validation, a health check, a latency header, a non-root container, and a 615 MB CPU-only image.

---

## Quick start

```bash
# 1. install
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt

# 2. train (about 5 minutes on a laptop; see "Data" below)
python -m recsys.train --data-dir path/to/data --out artifacts

# 3. serve with Docker
docker compose up --build -d
curl localhost:8000/health
```

Then open **http://localhost:8000/docs** for interactive Swagger docs.

---

## Results

Trained on 455,854 ratings covering 11.3K users and 24.7K businesses, then evaluated on 142,044 held-out ratings.

| Model                          | Validation RMSE |
|--------------------------------|-----------------|
| Global mean                    | 1.1222          |
| Bayesian user/business average | 1.0374          |
| Biased matrix factorization    | 1.0072          |
| XGBoost (73 features)          | 0.9805          |
| **Hybrid blend**               | **0.9791**      |

**Error distribution** (absolute error of the hybrid model):

| Error       | Count   | Share |
|-------------|---------|-------|
| < 1 star    | 102,823 | 72.4% |
| 1–2 stars   | 32,049  | 22.6% |
| 2–3 stars   | 6,287   | 4.4%  |
| 3–4 stars   | 881     | 0.6%  |
| ≥ 4 stars   | 4       | 0.0%  |

**Serving performance** (Docker on a local laptop, 50 requests per endpoint):

| Endpoint      | p50   | p95   |
|---------------|-------|-------|
| `/predict`    | 10 ms | 22 ms |
| `/recommend`  | 16 ms | 32 ms |

The model loads in about 1.5 s at startup. Artifacts total about 37 MB.

---

## Architecture

```
                 ┌──────────── offline: recsys/train.py ─────────────┐
 Yelp ratings ──►│ features: user/business metadata, review, tip,    │
 + 6 JSON files  │ check-in, photo aggregates; leave-one-out rating  │──► artifacts/
                 │ encodings for train rows, Bayesian-smoothed at    │    ├─ model.pkl
                 │ inference  ──►  XGBoost  +  biased MF  +  ridge   │    ├─ xgb.json
                 │ blend fit on a 10% train-only holdout             │    └─ metrics.json
                 └───────────────────────────────────────────────────┘
                                         │
                 ┌──────────── online: app/main.py ──────────────────┐
 GET /predict ──►│ build features ─► XGBoost, MF, baseline ─► blend  │
 GET /recommend ►│ stage 1: MF dot-product retrieval over all        │
                 │          eligible businesses ─► top 300           │
                 │ stage 2: re-rank candidates with the full hybrid  │
                 └───────────────────────────────────────────────────┘
```

### Features (73 total)

| Group              | Examples                                                                 |
|--------------------|--------------------------------------------------------------------------|
| User profile       | average stars, review count, fans, elite years, account age, compliments |
| Business profile   | stars, review count, price range, open status, location, 10 attribute flags, 10 category flags |
| Review aggregates  | mean useful/funny/cool votes and review length, per user and per business |
| Engagement         | check-in volume, tip counts and likes, photo counts and caption rate     |
| Rating encodings   | user/business average rating (leave-one-out or Bayesian), rating std, interaction terms, cold-start flags |

### Design decisions

- **Leave-one-out encoding.** A naive user-average feature contains the row's own label, so the model learns to copy it and overfits. Training rows use averages computed *without* the current rating. At inference, averages are smoothed toward the global mean with `(sum + 10·μ) / (count + 10)`, so a user with one rating isn't treated as fully known.
- **Train-only blend calibration.** A stable hash of `(user, business)` holds out 10% of the training data. Base models are fit on the other 90%, and ridge weights are learned on the holdout. Everything is then refit on 100%. Validation labels are never used for tuning, so the reported RMSE is honest.
- **Two-stage retrieval.** Running the 73-feature XGBoost path over all 25K businesses for every request would be slow. Matrix factorization scores are a single matrix-vector product, so they shortlist 300 candidates first, the same retrieve-then-rank pattern production recommenders use.
- **Home-state default.** With no location filter, recommendations come from the state where the user has rated the most businesses. Without this, one user's top picks could mix Toronto and Cleveland. `anywhere=true` disables it.
- **CPU-only XGBoost in Docker.** The standard `xgboost` wheel bundles CUDA/NCCL libraries. Switching to `xgboost-cpu` cut the image from 1.81 GB to 615 MB.

---

## API reference

### `GET /predict`

Predicts the star rating a user would give a business.

| Param         | Required | Description     |
|---------------|----------|-----------------|
| `user_id`     | yes      | Yelp user ID     |
| `business_id` | yes      | Yelp business ID |

```bash
curl "localhost:8000/predict?user_id=wf1GqnKQuvH-V3QN80UOOQ&business_id=fThrN4tfupIGetkrz18JOg"
```
```json
{
  "user_id": "wf1GqnKQuvH-V3QN80UOOQ",
  "business_id": "fThrN4tfupIGetkrz18JOg",
  "predicted_stars": 3.7577,
  "components": {
    "xgboost": 3.6892,
    "matrix_factorization": 4.1202,
    "bayesian_baseline": 3.9294
  },
  "known_user": true,
  "known_business": true
}
```

| Status | When                                              |
|--------|---------------------------------------------------|
| 200    | at least one of the user or business is known     |
| 404    | neither the user nor the business is known        |
| 422    | a required parameter is missing                   |

### `GET /recommend/{user_id}`

Returns the top-K businesses for a user, ranked by predicted rating.

| Param            | Default           | Description                                         |
|------------------|-------------------|-----------------------------------------------------|
| `k`              | `10`              | number of results (1–100)                           |
| `city`           | none              | exact city match, case-insensitive                  |
| `state`          | user's home state | two-letter state code, e.g. `NV`                    |
| `category`       | none              | substring of the business categories, e.g. `pizza`  |
| `anywhere`       | `false`           | turn off the home-state default                     |
| `include_closed` | `false`           | include closed businesses                           |
| `exclude_rated`  | `true`            | skip businesses the user has already rated          |

```bash
curl "localhost:8000/recommend/wf1GqnKQuvH-V3QN80UOOQ?k=3&city=Las%20Vegas&category=pizza"
```
```json
{
  "user_id": "wf1GqnKQuvH-V3QN80UOOQ",
  "known_user": true,
  "k": 3,
  "filters": {"city": "Las Vegas", "state": null, "category": "pizza",
              "include_closed": false, "exclude_rated": true},
  "recommendations": [
    {"rank": 1, "business_id": "e6m1p7oPbq7Gq37eMhSsWA", "name": "Joey's NY Pizza",
     "city": "Las Vegas", "state": "NV",
     "categories": "Pasta Shops, Sandwiches, Restaurants, Food, Pizza, ...",
     "yelp_stars": 4.5, "predicted_stars": 4.5628},
    {"rank": 2, "...": "..."},
    {"rank": 3, "...": "..."}
  ]
}
```

New users without rating history still get recommendations, ranked by business-side signals. If no business matches the filters, the endpoint returns an empty list.

### `GET /health`

Returns service status, model load time, catalog sizes, and validation RMSE. The Docker health check uses it.

Every response includes an `X-Process-Time-Ms` header.

---

## Data

The project uses the [Yelp Open Dataset](https://www.yelp.com/dataset). The data is **not** included in this repo because of Yelp's license terms and its size (~3 GB). Put these files in one folder:

| File                 | Contents                                    |
|----------------------|---------------------------------------------|
| `yelp_train.csv`     | `user_id,business_id,stars` training ratings |
| `yelp_val.csv`       | same format, used only for evaluation       |
| `user.json`          | user profiles                               |
| `business.json`      | business metadata                           |
| `review_train.json`  | review text and votes                       |
| `checkin.json`       | check-in counts                             |
| `tip.json`           | tips                                        |
| `photo.json`         | photo metadata                              |

`yelp_val.csv` is optional. Without it, training still runs but skips the validation metrics. `user.json` and `business.json` are required. The review, check-in, tip, and photo files are optional; if one is missing, those features fall back to default values.

---

## Training

```bash
python -m recsys.train --data-dir path/to/data --out artifacts
```

| Flag          | Default     | Description                    |
|---------------|-------------|--------------------------------|
| `--data-dir`  | required    | folder containing the data     |
| `--out`       | `artifacts` | where artifacts are written    |
| `--n-trees`   | `520`       | XGBoost boosting rounds        |
| `--mf-epochs` | `10`        | matrix factorization epochs    |
| `--n-jobs`    | `-1`        | XGBoost threads (-1 = all)     |

Output:

```
artifacts/
├── model.pkl      # feature tables, rating stats, MF factors, blend weights
├── xgb.json       # XGBoost booster
└── metrics.json   # RMSE per model, error distribution, blend weights
```

---

## Running the service

**Docker (recommended)**

```bash
docker compose up --build -d     # build and start on port 8000
docker compose logs -f api       # follow logs
docker compose down              # stop
```

The image copies `artifacts/` in at build time, so train before building. To swap in retrained artifacts without rebuilding, uncomment the `volumes` block in `docker-compose.yml`.

**Locally**

```bash
uvicorn app.main:app --port 8000
```

Set `ARTIFACT_DIR` to load artifacts from somewhere other than `./artifacts`.

---

## Testing

```bash
pytest
```

The 13 tests don't need the real dataset. They generate a small synthetic Yelp-format dataset, train a reduced model on it, and test the API end to end:

- feature vector shape matches the feature names
- metrics are written, and the hybrid beats the global mean
- `/predict` handles known pairs, cold-start users, unknown pairs (404), and missing parameters (422)
- `/recommend` ranks results in order, excludes already-rated and closed businesses, applies city/state/category filters, uses the home-state default, handles cold-start users, returns an empty list when nothing matches, and validates `k`

---

## Project structure

```
yelp-hybrid-recsys/
├── app/
│   └── main.py            # FastAPI app: routes, schemas, middleware
├── recsys/
│   ├── features.py        # data loaders + FeatureBuilder (LOO / Bayesian encodings)
│   ├── mf.py              # biased matrix factorization (SGD)
│   ├── model.py           # HybridRecommender: predict + two-stage recommend
│   └── train.py           # training, blend calibration, metrics, artifact export
├── tests/
│   ├── conftest.py        # synthetic dataset + small trained model fixture
│   └── test_api.py        # end-to-end API tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt       # runtime dependencies
├── requirements-dev.txt   # + pytest, httpx
└── pyproject.toml
```

---

## Tech stack

**Modeling:** Python, NumPy, XGBoost, matrix factorization (SGD), ridge regression
**Serving:** FastAPI, Pydantic, Uvicorn
**Infrastructure:** Docker, Docker Compose
**Testing:** pytest, FastAPI TestClient

---

## Limitations and future work

- **No text understanding.** Review text only contributes its length. Sentence embeddings of reviews and business descriptions could add a content-based signal.
- **Static model.** New ratings require a full retrain. Incremental MF updates or a scheduled retraining job would fix this.
- **Matrix factorization training is a pure-Python SGD loop** (35–90 s per fit). It could be vectorized or replaced with ALS.
- **Explanations.** SHAP values per recommendation ("recommended because you rate Italian places highly") would make results easier to interpret.
- **Experiment tracking.** Logging runs to MLflow would make hyperparameter comparisons reproducible.
- **Scale.** Candidate retrieval is brute force over 25K items. A larger catalog would need an approximate nearest-neighbor index (FAISS).

---

## Acknowledgments

- Data: [Yelp Open Dataset](https://www.yelp.com/dataset)
- Built on coursework from DSCI 553 at the University of Southern California
