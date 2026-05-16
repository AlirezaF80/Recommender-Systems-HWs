"""HW1 preprocessing: temporal split, cold-start filter, rating matrices."""

import csv
from pathlib import Path

import numpy as np

TRAIN_RATIO = 0.8


def load_ratings(path):
    """Read ratings CSV. Each row: (userId, movieId, rating, timestamp)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    int(row["userId"]),
                    int(row["movieId"]),
                    float(row["rating"]),
                    int(row["timestamp"]),
                )
            )
    return rows


def split_by_time(rows, train_ratio=TRAIN_RATIO):
    """Sort by timestamp, then split 80/20 by row count."""
    sorted_rows = sorted(rows, key=lambda r: (r[3], r[0], r[1]))
    split_idx = int(train_ratio * len(sorted_rows))
    train = sorted_rows[:split_idx]
    test = sorted_rows[split_idx:]
    return train, test


def filter_test(train, test):
    """Keep test rows only for users who have at least one train rating."""
    train_users = {r[0] for r in train}
    filtered = [r for r in test if r[0] in train_users]
    return filtered


def count_stats(rows):
    """Return counts: ratings, users, items."""
    users = {r[0] for r in rows}
    items = {r[1] for r in rows}
    return {
        "ratings": len(rows),
        "users": len(users),
        "items": len(items),
    }


def all_movie_ids(rows):
    """Sorted list of every movieId in the dataset."""
    return sorted({r[1] for r in rows})


def build_matrices(train, test, movie_ids):
    """
    Build train and test rating matrices with shared movie columns.
    Missing ratings are np.nan.
    """
    item_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    n_items = len(movie_ids)

    train_users = sorted({r[0] for r in train})
    test_users = sorted({r[0] for r in test})
    train_user_to_idx = {uid: i for i, uid in enumerate(train_users)}
    test_user_to_idx = {uid: i for i, uid in enumerate(test_users)}

    train_matrix = np.full((len(train_users), n_items), np.nan)
    test_matrix = np.full((len(test_users), n_items), np.nan)

    for user_id, movie_id, rating, _ in train:
        train_matrix[train_user_to_idx[user_id], item_id_to_idx[movie_id]] = rating

    for user_id, movie_id, rating, _ in test:
        test_matrix[test_user_to_idx[user_id], item_id_to_idx[movie_id]] = rating

    return {
        "train_matrix": train_matrix,
        "test_matrix": test_matrix,
        "item_id_to_idx": item_id_to_idx,
        "train_user_id_to_idx": train_user_to_idx,
        "test_user_id_to_idx": test_user_to_idx,
    }


def preprocess(path):
    """Run full preprocessing pipeline. Returns a dict with all results."""
    path = Path(path)
    rows = load_ratings(path)
    train, test_raw = split_by_time(rows)
    test = filter_test(train, test_raw)
    movie_ids = all_movie_ids(rows)
    matrices = build_matrices(train, test, movie_ids)

    train_stats = count_stats(train)
    test_stats = count_stats(test)

    return {
        "train": train,
        "test": test,
        "train_matrix": matrices["train_matrix"],
        "test_matrix": matrices["test_matrix"],
        "item_id_to_idx": matrices["item_id_to_idx"],
        "train_user_id_to_idx": matrices["train_user_id_to_idx"],
        "test_user_id_to_idx": matrices["test_user_id_to_idx"],
        "stats": {
            "train_ratings": train_stats["ratings"],
            "train_users": train_stats["users"],
            "train_items": train_stats["items"],
            "test_ratings": test_stats["ratings"],
            "test_users": test_stats["users"],
            "test_items": test_stats["items"],
            "total_movies_in_matrices": len(movie_ids),
        },
    }


def print_stats(result):
    """Print preprocessing summary for the report."""
    s = result["stats"]
    train_m = result["train_matrix"]
    test_m = result["test_matrix"]

    print("=== Preprocessing results ===\n")
    print("Train set:")
    print(f"  ratings: {s['train_ratings']}")
    print(f"  users:   {s['train_users']}")
    print(f"  items:   {s['train_items']}")
    print()
    print("Test set (after cold-start filter):")
    print(f"  ratings: {s['test_ratings']}")
    print(f"  users:   {s['test_users']}")
    print(f"  items:   {s['test_items']}")
    print()
    print("Rating matrices (shared movie columns):")
    print(f"  train: {train_m.shape[0]} users x {train_m.shape[1]} movies")
    print(f"  test:  {test_m.shape[0]} users x {test_m.shape[1]} movies")
