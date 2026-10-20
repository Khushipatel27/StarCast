"""Parsers and feature construction for the Yelp rating model.

Ported from the Spark RDD competition pipeline to plain Python so the same
feature code runs at training time and inside the API process.
"""
import json
import math
from collections import defaultdict

GLOBAL_FALLBACK = 3.75
MIN_STARS = 1.0
MAX_STARS = 5.0
BAYES_K = 10.0

ATTR_KEYS = (
    "RestaurantsReservations", "RestaurantsTakeOut", "RestaurantsDelivery",
    "BusinessAcceptsCreditCards", "BikeParking", "GoodForKids", "HasTV",
    "OutdoorSeating", "RestaurantsGoodForGroups", "RestaurantsTableService",
)
CAT_TERMS = (
    "restaurants", "food", "nightlife", "bars", "shopping", "beauty & spas",
    "home services", "health & medical", "automotive", "local services",
)
COMPLIMENT_KEYS = (
    "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute",
    "compliment_list", "compliment_note", "compliment_plain", "compliment_cool",
    "compliment_funny", "compliment_writer", "compliment_photos",
)

DEF_USER = (GLOBAL_FALLBACK, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0)
DEF_BIZ = (GLOBAL_FALLBACK, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0) + (0.0,) * 20
DEF_RV = (0.0, 0.0, 0.0, 0.0)
DEF_PAIR = (0.0, 0.0)

FEATURE_NAMES = (
    ["u_avg_stars", "u_log_review_count", "u_log_fans", "u_log_useful", "u_log_funny",
     "u_log_cool", "u_elite_years", "u_account_age", "u_log_compliments"]
    + ["b_stars", "b_log_review_count", "b_is_open", "b_price", "b_n_categories", "b_lat", "b_lon"]
    + [f"b_attr_{k}" for k in ATTR_KEYS]
    + [f"b_cat_{t.replace(' & ', '_').replace(' ', '_')}" for t in CAT_TERMS]
    + ["u_rv_useful", "u_rv_funny", "u_rv_cool", "u_rv_log_len"]
    + ["b_rv_useful", "b_rv_funny", "b_rv_cool", "b_rv_log_len"]
    + ["b_log_checkins", "b_log_checkin_slots"]
    + ["u_log_tips", "u_tip_likes", "b_log_tips", "b_tip_likes"]
    + ["b_log_photos", "b_photo_caption_frac"]
    + ["u_enc", "b_enc", "u_minus_b_enc"]
    + ["u_log_train_cnt", "b_log_train_cnt", "u_train_std", "b_train_std", "abs_u_b_enc",
       "bayes_mix", "u_meta_avg", "b_meta_avg", "u_meta_minus_enc", "b_meta_minus_enc",
       "u_meta_minus_b_meta", "u_cold", "b_cold", "log_cnt_product", "u_enc_centered",
       "b_enc_centered", "bayes_mix_centered", "abs_u_meta_b_meta"]
)
N_FEATURES = len(FEATURE_NAMES)


def clip(x):
    return max(MIN_STARS, min(MAX_STARS, x))


def bayes_avg(total, count, g):
    return (total + BAYES_K * g) / (count + BAYES_K)


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


def _std(total, count, sumsq):
    if count <= 1:
        return 0.0
    mean = total / count
    var = sumsq / count - mean * mean
    return math.sqrt(var) if var > 0.0 else 0.0


def rating_stats(rows):
    """Per-user / per-business sum, count, sum-of-squares from (u, b, r) rows."""
    u, b = defaultdict(lambda: [0.0, 0, 0.0]), defaultdict(lambda: [0.0, 0, 0.0])
    for uid, bid, r in rows:
        for acc in (u[uid], b[bid]):
            acc[0] += r
            acc[1] += 1
            acc[2] += r * r
    return {k: tuple(v) for k, v in u.items()}, {k: tuple(v) for k, v in b.items()}


class FeatureBuilder:
    """Builds the model feature vector for a (user, business) pair.

    Training rows use leave-one-out rating encodings (the row's own rating is
    removed from the user/business average) so the model never sees its own
    label. Prediction rows use Bayesian-smoothed averages toward the global
    mean, which keeps cold-start users and businesses well-behaved.
    """

    def __init__(self, meta, u_stats, b_stats, global_avg):
        self.users = meta["users"]
        self.biz = meta["biz"]
        self.rv_u, self.rv_b = meta["rv_u"], meta["rv_b"]
        self.chk = meta["chk"]
        self.tip_u, self.tip_b = meta["tip_u"], meta["tip_b"]
        self.pht = meta["pht"]
        self.u_stats, self.b_stats = u_stats, b_stats
        self.g = global_avg

    def _base(self, uid, bid, u_enc, b_enc):
        uf = list(self.users.get(uid, DEF_USER))
        bf = list(self.biz.get(bid, DEF_BIZ))
        for i in (1, 2, 3, 4, 5, 8):
            uf[i] = math.log1p(uf[i])
        bf[1] = math.log1p(bf[1])
        ck = self.chk.get(bid, DEF_PAIR)
        tu = self.tip_u.get(uid, DEF_PAIR)
        tb = self.tip_b.get(bid, DEF_PAIR)
        pt = self.pht.get(bid, DEF_PAIR)
        return (uf + bf + list(self.rv_u.get(uid, DEF_RV)) + list(self.rv_b.get(bid, DEF_RV))
                + [math.log1p(ck[0]), math.log1p(ck[1])]
                + [math.log1p(tu[0]), tu[1], math.log1p(tb[0]), tb[1]]
                + [math.log1p(pt[0]), pt[1]]
                + [u_enc, b_enc, u_enc - b_enc])

    def _extra(self, uid, bid, u_enc, b_enc, uc, bc, u_std, b_std):
        g = self.g
        um = float(self.users.get(uid, DEF_USER)[0])
        bm = float(self.biz.get(bid, DEF_BIZ)[0])
        lu, lb = math.log1p(max(0.0, uc)), math.log1p(max(0.0, bc))
        mix = 0.45 * u_enc + 0.55 * b_enc
        return [lu, lb, u_std, b_std, abs(u_enc - b_enc), mix, um, bm, um - u_enc,
                bm - b_enc, um - bm, 1.0 if uc <= 0 else 0.0, 1.0 if bc <= 0 else 0.0,
                lu * lb, u_enc - g, b_enc - g, mix - g, abs(um - bm)]

    def train_row(self, uid, bid, r):
        g = self.g
        us, uc, uss = self.u_stats[uid]
        bs, bc, bss = self.b_stats[bid]
        u_enc = (us - r) / (uc - 1) if uc > 1 else g
        b_enc = (bs - r) / (bc - 1) if bc > 1 else g
        u_std = _std(us - r, uc - 1, uss - r * r)
        b_std = _std(bs - r, bc - 1, bss - r * r)
        return (self._base(uid, bid, u_enc, b_enc)
                + self._extra(uid, bid, u_enc, b_enc, uc - 1, bc - 1, u_std, b_std))

    def predict_row(self, uid, bid):
        """Returns (features, bayes_baseline)."""
        g = self.g
        us, uc, uss = self.u_stats.get(uid, (0.0, 0, 0.0))
        bs, bc, bss = self.b_stats.get(bid, (0.0, 0, 0.0))
        u_enc = bayes_avg(us, uc, g)
        b_enc = bayes_avg(bs, bc, g)
        feats = (self._base(uid, bid, u_enc, b_enc)
                 + self._extra(uid, bid, u_enc, b_enc, uc, bc,
                               _std(us, uc, uss), _std(bs, bc, bss)))
        return feats, 0.45 * u_enc + 0.55 * b_enc
