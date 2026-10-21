"""Central settings: artifact layout, model hyperparameters, serving defaults."""
import os

# Artifacts
DEFAULT_ARTIFACT_DIR = "artifacts"
MODEL_FILE = "model.pkl"
XGB_FILE = "xgb.json"
METRICS_FILE = "metrics.json"


def artifact_dir():
    return os.environ.get("ARTIFACT_DIR", DEFAULT_ARTIFACT_DIR)


# Training
XGB_PARAMS = dict(
    objective="reg:squarederror", max_depth=6, learning_rate=0.03, subsample=0.82,
    colsample_bytree=0.82, min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", seed=42, verbosity=0,
)
N_TREES = 520
MF_PARAMS = dict(n_factors=12, n_epochs=10, lr=0.0075, reg=0.055, reg_bias=0.025, seed=553)
HOLDOUT_MOD = 10  # 1 in HOLDOUT_MOD training rows calibrates the blend
RIDGE_L2 = 1.0

# Serving
N_CANDIDATES = 300  # stage-1 retrieval size for /recommend
MAX_K = 100
