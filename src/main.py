"""Run preprocessing and print stats."""

from pathlib import Path

from popularity import evaluate_popularity, print_popularity_results
from preprocess import preprocess, print_stats

RATINGS_PATH = Path(__file__).resolve().parent / "MovieLens" / "ratings.csv"


def main():
    result = preprocess(RATINGS_PATH)
    print_stats(result)
    metrics = evaluate_popularity(result)
    print_popularity_results(metrics)
    return result, metrics


if __name__ == "__main__":
    main()
