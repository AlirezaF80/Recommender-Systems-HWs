"""Top-n ranking helpers shared across recommender systems."""

from preprocess import MOVIEID_IDX, RATING_IDX, USERID_IDX


def group_by_user(rows):
    """Group rating rows by userId."""
    by_user = {}
    for row in rows:
        user_id = row[USERID_IDX]
        if user_id not in by_user:
            by_user[user_id] = []
        by_user[user_id].append(row)
    return by_user


def user_top_n(user_rows, n):
    """Top-n movieIds for one user by rating (highest first)."""
    sorted_rows = sorted(
        user_rows, key=lambda r: (r[RATING_IDX], r[MOVIEID_IDX]), reverse=True
    )
    seen = set()
    top = []
    for row in sorted_rows:
        movie_id = row[MOVIEID_IDX]
        if movie_id in seen:
            continue
        seen.add(movie_id)
        top.append(movie_id)
        if len(top) == n:
            break
    return top
