"""Evaluation metrics."""
import numpy as np

ERROR_BINS = (
    (">=0 and <1", 0, 1),
    (">=1 and <2", 1, 2),
    (">=2 and <3", 2, 3),
    (">=3 and <4", 3, 4),
    (">=4", 4, np.inf),
)


def rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def error_distribution(pred, y):
    d = np.abs(np.asarray(pred) - np.asarray(y))
    return {label: int(((d >= lo) & (d < hi)).sum()) for label, lo, hi in ERROR_BINS}
