"""System 2: user-based collaborative filtering (Pearson)."""

import math

from tqdm import tqdm

from metrics import N_VALUES, average_metrics_per_user
from preprocess import MOVIEID_IDX
from ranking import group_by_user

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


def build_movie_index(all_ratings):
    """Invert profiles: movie_id -> {user_id: rating}."""
    movie_users = {}
    for user_id, movies in all_ratings.items():
        for movie_id, rating in movies.items():
            if movie_id not in movie_users:
                movie_users[movie_id] = {}
            movie_users[movie_id][user_id] = rating
    return movie_users


def co_rating_counts(user_id, ratings_u, movie_users):
    """How many train movies other users share with user_id."""
    overlap = {}
    for movie_id in ratings_u:
        for other_id in movie_users.get(movie_id, {}):
            if other_id == user_id:
                continue
            overlap[other_id] = overlap.get(other_id, 0) + 1
            if overlap[other_id] >= 2:
                break
    return overlap


def pearson_sim(ratings_u, ratings_v):
    """Pearson correlation on co-rated items; 0 if fewer than 2 co-ratings."""
    common_pairs = []
    if len(ratings_u) <= len(ratings_v):
        for movie_id, rating_u in ratings_u.items():
            rating_v = ratings_v.get(movie_id)
            if rating_v is not None:
                common_pairs.append((rating_u, rating_v))
    else:
        for movie_id, rating_v in ratings_v.items():
            rating_u = ratings_u.get(movie_id)
            if rating_u is not None:
                common_pairs.append((rating_u, rating_v))

    n = len(common_pairs)
    if n < 2:
        return 0.0

    mean_u = sum(ru for ru, _ in common_pairs) / n
    mean_v = sum(rv for _, rv in common_pairs) / n

    num = 0.0
    den_u = 0.0
    den_v = 0.0
    for rating_u, rating_v in common_pairs:
        diff_u = rating_u - mean_u
        diff_v = rating_v - mean_v
        num += diff_u * diff_v
        den_u += diff_u * diff_u
        den_v += diff_v * diff_v

    if den_u == 0.0 or den_v == 0.0:
        return 0.0
    return num / (math.sqrt(den_u) * math.sqrt(den_v))


def ranked_neighbor_sims(user_id, all_ratings, movie_users):
    """All other users sorted by Pearson (excluding self)."""
    ratings_u = all_ratings[user_id]
    overlap = co_rating_counts(user_id, ratings_u, movie_users)
    sims = []
    for other_id in all_ratings:
        if other_id == user_id:
            continue
        if overlap.get(other_id, 0) < 2:
            sims.append((other_id, 0.0))
        else:
            sims.append((other_id, pearson_sim(ratings_u, all_ratings[other_id])))
    sims.sort(key=lambda x: (x[1], x[0]), reverse=True)
    return sims


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


def ranked_movie_ids_from_scores(movie_scores):
    """All movieIds by predicted score (highest first), for slicing top-n."""
    ranked = sorted(movie_scores.items(), key=lambda x: (x[1], -x[0]), reverse=True)
    return [movie_id for movie_id, _ in ranked]


def evaluate_user_cf(result, k_values=None, n_values=None):
    """Return {k: {n: {hit_rate, precision}}} over test users."""
    if k_values is None:
        k_values = K_VALUES
    if n_values is None:
        n_values = N_VALUES

    all_ratings, user_means = build_profiles(result["train"])
    movie_users = build_movie_index(all_ratings)
    test_by_user = group_by_user(result["test"])

    ranked_neighbors_by_user = {}
    for user_id in tqdm(test_by_user, desc="User CF neighbors", unit="user"):
        ranked_neighbors_by_user[user_id] = ranked_neighbor_sims(
            user_id, all_ratings, movie_users
        )

    metrics = {k: {} for k in k_values}
    for k in tqdm(k_values, desc="User CF k", unit="k"):
        neighbors_by_user = {
            user_id: ranked_neighbors_by_user[user_id][:k]
            for user_id in test_by_user
        }
        scores_by_user = {}
        for user_id, user_rows in tqdm(
            test_by_user.items(),
            desc=f"User CF predict (k={k})",
            unit="user",
            leave=False,
        ):
            test_movies = list({r[MOVIEID_IDX] for r in user_rows})
            scores_by_user[user_id] = predict_test_user(
                user_id,
                test_movies,
                all_ratings,
                user_means,
                neighbors_by_user[user_id],
            )

        ranked_movies_by_user = {
            user_id: ranked_movie_ids_from_scores(scores)
            for user_id, scores in scores_by_user.items()
        }
        for n in n_values:
            recommended_by_user = {
                user_id: ranked_movies_by_user[user_id][:n]
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
