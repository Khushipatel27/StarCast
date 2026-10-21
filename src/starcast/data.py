"""Readers for the Yelp ratings CSVs and the six metadata JSON files."""
import json
import math
from collections import defaultdict

from .features import ATTR_KEYS, CAT_TERMS, COMPLIMENT_KEYS, GLOBAL_FALLBACK


def _f(v, d):
    try:
        return float(v) if v is not None else float(d)
    except (ValueError, TypeError):
        return float(d)


def _bool_attr(attrs, key):
    v = attrs.get(key)
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if v is None:
        return 0.0
    return 1.0 if str(v).strip().lower().strip("\"'") in ("true", "yes", "1", "free") else 0.0


def iter_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def read_ratings(path):
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            p = line.strip().split(",")
            if len(p) >= 3:
                rows.append((p[0], p[1], float(p[2])))
    return rows


def load_users(path, keep=None):
    out = {}
    for o in iter_json(path):
        uid = o.get("user_id")
        if not uid or (keep is not None and uid not in keep):
            continue
        since = o.get("yelping_since") or "2010-01-01"
        try:
            age = 2022 - int(since[:4])
        except ValueError:
            age = 5
        elite = o.get("elite") or ""
        elite_years = 0 if (not elite or elite == "None") else len(elite.split(","))
        comp = sum(_f(o.get(k), 0) for k in COMPLIMENT_KEYS)
        out[uid] = (
            _f(o.get("average_stars"), GLOBAL_FALLBACK), _f(o.get("review_count"), 0),
            _f(o.get("fans"), 0), _f(o.get("useful"), 0), _f(o.get("funny"), 0),
            _f(o.get("cool"), 0), float(elite_years), float(age), comp,
        )
    return out


def load_businesses(path):
    """Returns (feature tuples, display info) keyed by business_id."""
    feats, info = {}, {}
    for o in iter_json(path):
        bid = o.get("business_id")
        if not bid:
            continue
        attrs = o.get("attributes") or {}
        try:
            price = int(attrs.get("RestaurantsPriceRange2", 0) or 0)
        except (ValueError, TypeError):
            price = 0
        cats = o.get("categories") or ""
        cats_lower = cats.lower()
        feats[bid] = tuple(
            [_f(o.get("stars"), GLOBAL_FALLBACK), _f(o.get("review_count"), 0),
             _f(o.get("is_open"), 1), float(price), float(len(cats.split(",")) if cats else 0),
             _f(o.get("latitude"), 0), _f(o.get("longitude"), 0)]
            + [_bool_attr(attrs, k) for k in ATTR_KEYS]
            + [1.0 if t in cats_lower else 0.0 for t in CAT_TERMS]
        )
        info[bid] = {
            "name": o.get("name") or "",
            "city": o.get("city") or "",
            "state": o.get("state") or "",
            "categories": cats,
            "stars": _f(o.get("stars"), 0),
            "is_open": bool(_f(o.get("is_open"), 1)),
        }
    return feats, info


def load_review_aggs(path):
    u_acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    b_acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0])
    for o in iter_json(path):
        uid, bid = o.get("user_id"), o.get("business_id")
        if not uid or not bid:
            continue
        vals = (_f(o.get("useful"), 0), _f(o.get("funny"), 0), _f(o.get("cool"), 0),
                float(len(o.get("text") or "")))
        for acc in (u_acc[uid], b_acc[bid]):
            for i, v in enumerate(vals):
                acc[i] += v
            acc[4] += 1

    def fin(a):
        n = a[4]
        return (a[0] / n, a[1] / n, a[2] / n, math.log1p(a[3] / n))

    return {k: fin(v) for k, v in u_acc.items()}, {k: fin(v) for k, v in b_acc.items()}


def load_checkins(path):
    out = defaultdict(lambda: [0.0, 0.0])
    for o in iter_json(path):
        bid = o.get("business_id")
        if not bid:
            continue
        t = o.get("time") or {}
        total = 0
        for v in t.values():
            try:
                total += int(v)
            except (ValueError, TypeError):
                pass
        out[bid][0] += total
        out[bid][1] += len(t)
    return {k: tuple(v) for k, v in out.items()}


def load_tips(path):
    u_acc = defaultdict(lambda: [0, 0.0])
    b_acc = defaultdict(lambda: [0, 0.0])
    for o in iter_json(path):
        uid, bid = o.get("user_id"), o.get("business_id")
        if not uid or not bid:
            continue
        likes = _f(o.get("likes"), 0)
        for acc in (u_acc[uid], b_acc[bid]):
            acc[0] += 1
            acc[1] += likes

    def fin(a):
        return (float(a[0]), a[1] / a[0])

    return {k: fin(v) for k, v in u_acc.items()}, {k: fin(v) for k, v in b_acc.items()}


def load_photos(path):
    acc = defaultdict(lambda: [0.0, 0.0])
    for o in iter_json(path):
        bid = o.get("business_id")
        if not bid:
            continue
        acc[bid][0] += 1.0
        acc[bid][1] += 1.0 if o.get("caption") else 0.0
    return {k: (v[0], v[1] / v[0]) for k, v in acc.items()}
