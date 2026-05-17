"""System 1: popularity-based recommender."""

from metrics import N_VALUES, average_metrics
from preprocess import MOVIEID_IDX, RATING_IDX
from ranking import group_by_user


def count_popularity(train):
    """Count how many ratings each movie has in train."""
    popularity = {}
    for _, movie_id, _, _ in train:
        popularity[movie_id] = popularity.get(movie_id, 0) + 1
    return popularity


def global_top_n(popularity, n):
    """Top-n movies by rating count; ties broken by smaller movieId."""
    ranked = sorted(popularity.items(), key=lambda x: (x[1], x[0]), reverse=True)
    return [movie_id for movie_id, _ in ranked[:n]]


def evaluate_popularity(result, n_values=None):
    """Mean HitRate@n and Precision@n over all test users."""
    if n_values is None:
        n_values = N_VALUES

    popularity = count_popularity(result["train"])
    test_by_user = group_by_user(result["test"])

    metrics = {}
    for n in n_values:
        recommended = global_top_n(popularity, n)
        metrics[n] = average_metrics(recommended, test_by_user, n)
    return metrics


def print_popularity_results(metrics, n_values=None):
    """Print results table for console / report."""
    if n_values is None:
        n_values = sorted(metrics.keys())

    print("=== System 1: Popularity ===\n")
    print("Popularity = rating count on train\n")
    print(f"{'n':<6}{'HitRate@n':<14}{'Precision@n'}")
    for n in n_values:
        m = metrics[n]
        print(f"{n:<6}{m['hit_rate']:<14.4f}{m['precision']:.4f}")
