"""System 3: item-based collaborative filtering (Adjusted Cosine)."""

import math

from tqdm import tqdm

from metrics import N_VALUES, average_metrics_per_user
from preprocess import MOVIEID_IDX
from ranking import group_by_user, top_n_from_scores
from user_cf import K_VALUES, build_profiles


def build_item_ratings(user_ratings):
    """Invert user profiles to item_ratings[movieId][userId] = rating."""
    item_ratings = {}
    for user_id, movies in user_ratings.items():
        for movie_id, rating in movies.items():
            if movie_id not in item_ratings:
                item_ratings[movie_id] = {}
            item_ratings[movie_id][user_id] = rating
    return item_ratings


def adjusted_cosine_sim(ratings_i, ratings_j, user_means):
    """Adjusted cosine on co-rating users; 0 if fewer than 2 co-raters."""
    common = set(ratings_i) & set(ratings_j)
    if len(common) < 2:
        return 0.0

    num = 0.0
    den_i = 0.0
    den_j = 0.0
    for user_id in common:
        mean_u = user_means[user_id]
        diff_i = ratings_i[user_id] - mean_u
        diff_j = ratings_j[user_id] - mean_u
        num += diff_i * diff_j
        den_i += diff_i * diff_i
        den_j += diff_j * diff_j

    if den_i == 0.0 or den_j == 0.0:
        return 0.0
    return num / (math.sqrt(den_i) * math.sqrt(den_j))


def top_k_similar_items(movie_id, item_ratings, user_means, k):
    """Top-k most similar items by adjusted cosine (exclude self)."""
    ratings_i = item_ratings[movie_id]
    sims = []
    for other_id, ratings_j in item_ratings.items():
        if other_id == movie_id:
            continue
        sim = adjusted_cosine_sim(ratings_i, ratings_j, user_means)
        sims.append((other_id, sim))
    sims.sort(key=lambda x: (-x[1], x[0]))
    return sims[:k]


def predict_rating(user_id, movie_id, neighbors, user_ratings, user_means):
    """Mean-centered weighted prediction from similar items user rated in train."""
    mean_u = user_means[user_id]
    user_movies = user_ratings.get(user_id, {})
    num = 0.0
    den = 0.0
    for neighbor_id, sim in neighbors:
        if neighbor_id not in user_movies:
            continue
        r_uj = user_movies[neighbor_id]
        num += sim * (r_uj - mean_u)
        den += abs(sim)
    if den == 0.0:
        return mean_u
    return mean_u + num / den


def predict_test_user(
    user_id, test_movies, user_ratings, user_means, neighbors_cache
):
    """Predict scores for movies the user rated in test."""
    scores = {}
    for movie_id in test_movies:
        neighbors = neighbors_cache[movie_id]
        scores[movie_id] = predict_rating(
            user_id, movie_id, neighbors, user_ratings, user_means
        )
    return scores


def evaluate_item_cf(result, k_values=None, n_values=None):
    """Return {k: {n: {hit_rate, precision}}} over test users."""
    if k_values is None:
        k_values = K_VALUES
    if n_values is None:
        n_values = N_VALUES

    user_ratings, user_means = build_profiles(result["train"])
    item_ratings = build_item_ratings(user_ratings)
    test_by_user = group_by_user(result["test"])
    test_movie_ids = {r[MOVIEID_IDX] for r in result["test"]}

    metrics = {}
    for k in tqdm(k_values, desc="Item CF k", unit="k"):
        metrics[k] = {}
        neighbors_cache = {}
        for movie_id in tqdm(
            test_movie_ids,
            desc=f"Item CF neighbors (k={k})",
            unit="item",
            leave=False,
        ):
            if movie_id in item_ratings:
                neighbors_cache[movie_id] = top_k_similar_items(
                    movie_id, item_ratings, user_means, k
                )
            else:
                neighbors_cache[movie_id] = []
        scores_by_user = {}
        for user_id, user_rows in tqdm(
            test_by_user.items(),
            desc=f"Item CF predict (k={k})",
            unit="user",
            leave=False,
        ):
            test_movies = list({r[MOVIEID_IDX] for r in user_rows})
            scores_by_user[user_id] = predict_test_user(
                user_id,
                test_movies,
                user_ratings,
                user_means,
                neighbors_cache,
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


def print_item_cf_results(metrics, k_values=None, n_values=None):
    """Print metric tables for each k."""
    if k_values is None:
        k_values = sorted(metrics.keys())
    if n_values is None:
        n_values = N_VALUES

    print("=== System 3: Item-based CF (Adjusted Cosine) ===\n")
    print("Mean-centered prediction; candidates = test movies per user\n")

    for k in k_values:
        print(f"k={k}")
        print(f"{'n':<6}{'HitRate@n':<14}{'Precision@n'}")
        for n in n_values:
            m = metrics[k][n]
            print(f"{n:<6}{m['hit_rate']:<14.4f}{m['precision']:.4f}")
        print()
