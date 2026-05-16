"""Run preprocessing and print stats."""

from pathlib import Path

from preprocess import preprocess, print_stats

RATINGS_PATH = Path(__file__).resolve().parent / "MovieLens" / "ratings.csv"

def main():
    result = preprocess(RATINGS_PATH)
    print_stats(result)
    return result


if __name__ == "__main__":
    main()
