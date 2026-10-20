"""Biased matrix factorization trained with SGD on explicit ratings."""
import numpy as np

from .features import MAX_STARS, MIN_STARS


class BiasMF:
    def __init__(self, n_factors=12, n_epochs=10, lr=0.0075, reg=0.055, reg_bias=0.025, seed=553):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.reg_bias = reg_bias
        self.seed = seed

    def fit(self, rows, global_avg):
        self.g = float(global_avg)
        self.u_map, self.b_map = {}, {}
        us = np.fromiter((self.u_map.setdefault(u, len(self.u_map)) for u, _, _ in rows), np.int32, len(rows))
        bs = np.fromiter((self.b_map.setdefault(b, len(self.b_map)) for _, b, _ in rows), np.int32, len(rows))
        ys = np.fromiter((r for _, _, r in rows), np.float64, len(rows))

        rng = np.random.RandomState(self.seed)
        self.bu = np.zeros(len(self.u_map))
        self.bi = np.zeros(len(self.b_map))
        self.P = rng.normal(0.0, 0.025, (len(self.u_map), self.n_factors))
        self.Q = rng.normal(0.0, 0.025, (len(self.b_map), self.n_factors))

        bu, bi, P, Q, g = self.bu, self.bi, self.P, self.Q, self.g
        reg, reg_b = self.reg, self.reg_bias
        idx = np.arange(len(ys))
        for ep in range(self.n_epochs):
            rng.shuffle(idx)
            lr = self.lr * (0.92 ** ep)
            for u, b, r in zip(us[idx].tolist(), bs[idx].tolist(), ys[idx].tolist()):
                pu, qi = P[u], Q[b]
                err = r - (g + bu[u] + bi[b] + pu @ qi)
                bu[u] += lr * (err - reg_b * bu[u])
                bi[b] += lr * (err - reg_b * bi[b])
                pu_old = pu.copy()
                pu += lr * (err * qi - reg * pu)
                qi += lr * (err * pu_old - reg * qi)
        return self

    def predict(self, u, b):
        ui, bi_ = self.u_map.get(u), self.b_map.get(b)
        p = self.g
        if ui is not None:
            p += self.bu[ui]
        if bi_ is not None:
            p += self.bi[bi_]
        if ui is not None and bi_ is not None:
            p += float(self.P[ui] @ self.Q[bi_])
        return max(MIN_STARS, min(MAX_STARS, p))

    def predict_many(self, pairs):
        return np.array([self.predict(u, b) for u, b in pairs])

    def score_all_items(self, u):
        """Raw MF score for every known business; used for candidate retrieval."""
        ui = self.u_map.get(u)
        scores = self.g + self.bi
        if ui is not None:
            scores = scores + self.bu[ui] + self.Q @ self.P[ui]
        return scores
