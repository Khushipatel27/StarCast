"""Synthetic Yelp-format data for tests and CI.

    python -m tests.sample_data artifacts   # train a small model on it
"""
import json
import pathlib
import random
import sys
import tempfile

from starcast.training import train

N_USERS, N_BIZ = 60, 40
CITIES = ["Phoenix", "Las Vegas"]
CATS = ["Restaurants, Pizza", "Bars, Nightlife", "Shopping", "Beauty & Spas"]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def write_sample_dataset(data):
    """Writes a small Yelp-format dataset (ratings CSVs + 6 JSON files) into `data`."""
    data = pathlib.Path(data)
    data.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)
    users = [f"u{i:03d}" for i in range(N_USERS)]
    bizs = [f"b{i:03d}" for i in range(N_BIZ)]
    u_bias = {u: rng.uniform(-1, 1) for u in users}
    b_bias = {b: rng.uniform(-1, 1) for b in bizs}

    pairs = {(rng.choice(users), rng.choice(bizs)) for _ in range(1500)}
    rated = [(u, b, min(5, max(1, round(3.7 + u_bias[u] + b_bias[b] + rng.gauss(0, 0.6)))))
             for u, b in sorted(pairs)]
    split = int(len(rated) * 0.8)
    for name, rows in (("yelp_train.csv", rated[:split]), ("yelp_val.csv", rated[split:])):
        with open(data / name, "w") as fh:
            fh.write("user_id,business_id,stars\n")
            fh.writelines(f"{u},{b},{r}\n" for u, b, r in rows)

    _write_jsonl(data / "user.json", [
        {"user_id": u, "average_stars": round(3.7 + u_bias[u], 2), "review_count": rng.randint(1, 300),
         "fans": rng.randint(0, 20), "useful": 3, "funny": 1, "cool": 2, "elite": "2015,2016",
         "yelping_since": "2012-05-01"} for u in users])
    _write_jsonl(data / "business.json", [
        {"business_id": b, "name": f"Biz {i}", "city": CITIES[i % 2], "state": "AZ" if i % 2 == 0 else "NV",
         "stars": round(3.7 + b_bias[b], 1), "review_count": rng.randint(5, 500), "is_open": 0 if i == 1 else 1,
         "categories": CATS[i % 4], "latitude": 33.4, "longitude": -112.0,
         "attributes": {"RestaurantsPriceRange2": "2", "GoodForKids": "True"}}
        for i, b in enumerate(bizs)])
    _write_jsonl(data / "review_train.json", [
        {"user_id": u, "business_id": b, "useful": 1, "funny": 0, "cool": 0, "text": "good food " * r}
        for u, b, r in rated[:split]])
    _write_jsonl(data / "checkin.json", [{"business_id": b, "time": {"Mon-12": 3, "Fri-19": 7}} for b in bizs])
    _write_jsonl(data / "tip.json", [{"user_id": users[i], "business_id": bizs[i % N_BIZ], "likes": 1} for i in range(30)])
    _write_jsonl(data / "photo.json", [{"business_id": b, "caption": "menu"} for b in bizs[:10]])
    return data


def train_sample_model(out_dir):
    with tempfile.TemporaryDirectory() as tmp:
        write_sample_dataset(tmp)
        return train(tmp, str(out_dir), n_trees=30, mf_epochs=3, n_jobs=1)


if __name__ == "__main__":
    train_sample_model(sys.argv[1] if len(sys.argv) > 1 else "artifacts")
