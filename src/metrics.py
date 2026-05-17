"""HitRate@n and Precision@n — shared by Systems 1–4."""

from ranking import user_top_n

N_VALUES = [1, 5, 10, 15]


def hit_rate(recommended, relevant):
    """1 if recommended and relevant overlap, else 0."""
    return 1 if set(recommended) & set(relevant) else 0


def precision_at_n(recommended, relevant, n):
    """|intersection| / n."""
    return len(set(recommended) & set(relevant)) / n


def average_metrics(recommended, test_by_user, n):
    """
    Mean HitRate@n and Precision@n over all users in test_by_user.
    recommended: same top-n list for every user (e.g. global popular).
    """
    num_users = len(test_by_user)
    hit_sum = 0
    precision_sum = 0.0
    for user_rows in test_by_user.values():
        relevant = user_top_n(user_rows, n)
        hit_sum += hit_rate(recommended, relevant)
        precision_sum += precision_at_n(recommended, relevant, n)
    return {
        "hit_rate": hit_sum / num_users,
        "precision": precision_sum / num_users,
    }


def average_metrics_per_user(recommended_by_user, test_by_user, n):
    """
    Mean HitRate@n and Precision@n when each user has their own recommended list.
    For Systems 2–4 (per-user predictions).
    """
    num_users = len(test_by_user)
    hit_sum = 0
    precision_sum = 0.0
    for user_id, user_rows in test_by_user.items():
        recommended = recommended_by_user[user_id]
        relevant = user_top_n(user_rows, n)
        hit_sum += hit_rate(recommended, relevant)
        precision_sum += precision_at_n(recommended, relevant, n)
    return {
        "hit_rate": hit_sum / num_users,
        "precision": precision_sum / num_users,
    }
