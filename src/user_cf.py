"""System 2: user-based collaborative filtering (Pearson)."""

import math

from metrics import N_VALUES, average_metrics_per_user
from preprocess import MOVIEID_IDX
from ranking import group_by_user, top_n_from_scores

K_VALUES = [2, 5, 10]


def build_profiles(train):
    """Build per-user train ratings and mean rating."""
    ratings = {}
    for user_id, movie_id, rating, _ in train:
        if user_id not in ratings:
            ratings[user_id] = {}
        ratings[user_id][movie_id] = rating

    user_means = {}
    for user_id, movies in ratings.items():
        user_means[user_id] = sum(movies.values()) / len(movies)

    return ratings, user_means


def pearson_sim(ratings_u, ratings_v):
    """Pearson correlation on co-rated items; 0 if fewer than 2 co-ratings."""
    common = set(ratings_u) & set(ratings_v)
    if len(common) < 2:
        return 0.0

    mean_u = sum(ratings_u[i] for i in common) / len(common)
    mean_v = sum(ratings_v[i] for i in common) / len(common)

    num = 0.0
    den_u = 0.0
    den_v = 0.0
    for movie_id in common:
        diff_u = ratings_u[movie_id] - mean_u
        diff_v = ratings_v[movie_id] - mean_v
        num += diff_u * diff_v
        den_u += diff_u * diff_u
        den_v += diff_v * diff_v

    if den_u == 0.0 or den_v == 0.0:
        return 0.0
    return num / (math.sqrt(den_u) * math.sqrt(den_v))


def top_k_neighbors(user_id, all_ratings, k):
    """Top-k most similar users by Pearson (exclude self)."""
    sims = []
    ratings_u = all_ratings[user_id]
    for other_id, ratings_v in all_ratings.items():
        if other_id == user_id:
            continue
        sim = pearson_sim(ratings_u, ratings_v)
        sims.append((other_id, sim))
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:k]


def predict_rating(user_id, movie_id, neighbors, all_ratings, user_means):
    """Mean-centered weighted prediction; fallback to user mean."""
    mean_u = user_means[user_id]
    num = 0.0
    den = 0.0
    for neighbor_id, sim in neighbors:
        neighbor_ratings = all_ratings.get(neighbor_id, {})
        if movie_id not in neighbor_ratings:
            continue
        r_vi = neighbor_ratings[movie_id]
        mean_v = user_means[neighbor_id]
        num += sim * (r_vi - mean_v)
        den += abs(sim)
    if den == 0.0:
        return mean_u
    return mean_u + num / den


def predict_test_user(user_id, test_movies, all_ratings, user_means, neighbors):
    """Predict scores for movies the user rated in test."""
    scores = {}
    for movie_id in test_movies:
        scores[movie_id] = predict_rating(
            user_id, movie_id, neighbors, all_ratings, user_means
        )
    return scores


def evaluate_user_cf(result, k_values=None, n_values=None):
    """Return {k: {n: {hit_rate, precision}}} over test users."""
    if k_values is None:
        k_values = K_VALUES
    if n_values is None:
        n_values = N_VALUES

    all_ratings, user_means = build_profiles(result["train"])
    test_by_user = group_by_user(result["test"])

    metrics = {}
    for k in k_values:
        metrics[k] = {}
        neighbors_by_user = {
            user_id: top_k_neighbors(user_id, all_ratings, k)
            for user_id in test_by_user
        }
        scores_by_user = {}
        for user_id, user_rows in test_by_user.items():
            test_movies = list({r[MOVIEID_IDX] for r in user_rows})
            scores_by_user[user_id] = predict_test_user(
                user_id,
                test_movies,
                all_ratings,
                user_means,
                neighbors_by_user[user_id],
            )

        for n in n_values:
            recommended_by_user = {
                user_id: top_n_from_scores(scores_by_user[user_id], n)
                for user_id in test_by_user
            }
            metrics[k][n] = average_metrics_per_user(
                recommended_by_user, test_by_user, n
            )

    return metrics


def print_user_cf_results(metrics, k_values=None, n_values=None):
    """Print metric tables for each k."""
    if k_values is None:
        k_values = sorted(metrics.keys())
    if n_values is None:
        n_values = N_VALUES

    print("=== System 2: User-based CF (Pearson) ===\n")
    print("Mean-centered prediction; candidates = test movies per user\n")

    for k in k_values:
        print(f"k={k}")
        print(f"{'n':<6}{'HitRate@n':<14}{'Precision@n'}")
        for n in n_values:
            m = metrics[k][n]
            print(f"{n:<6}{m['hit_rate']:<14.4f}{m['precision']:.4f}")
        print()
