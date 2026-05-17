"""Run preprocessing and print stats."""

from pathlib import Path

from popularity import evaluate_popularity, print_popularity_results
from preprocess import preprocess, print_stats
from item_cf import evaluate_item_cf, print_item_cf_results
from slim import evaluate_slim, print_slim_results
from user_cf import evaluate_user_cf, print_user_cf_results

RATINGS_PATH = Path(__file__).resolve().parent / "MovieLens" / "ratings.csv"


def main():
    result = preprocess(RATINGS_PATH)
    print_stats(result)
    metrics = evaluate_popularity(result)
    print_popularity_results(metrics)
    user_cf_metrics = evaluate_user_cf(result)
    print_user_cf_results(user_cf_metrics)
    item_cf_metrics = evaluate_item_cf(result)
    print_item_cf_results(item_cf_metrics)
    slim_metrics = evaluate_slim(result)
    print_slim_results(slim_metrics)
    return result, metrics, user_cf_metrics, item_cf_metrics, slim_metrics


if __name__ == "__main__":
    main()
