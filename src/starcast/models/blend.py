"""Ridge blend of the component models' predictions."""
import numpy as np

from ..features import MAX_STARS, MIN_STARS


def fit_ridge(cols, y, l2=1.0):
    """Weights [bias, w1, w2, ...] minimising ||y - A w||^2 + l2 ||w[1:]||^2."""
    A = np.column_stack([np.ones(len(y))] + list(cols))
    reg = l2 * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def blend(weights, xgb_pred, mf_pred, bayes_pred):
    w = weights
    out = w[0] + w[1] * xgb_pred + w[2] * mf_pred + w[3] * bayes_pred
    return np.clip(out, MIN_STARS, MAX_STARS)
