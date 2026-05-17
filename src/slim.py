"""System 4: SLIM with pure-NumPy Elastic Net (coordinate descent)."""

import numpy as np
from tqdm import tqdm

from metrics import N_VALUES, average_metrics_per_user
from preprocess import MOVIEID_IDX
from ranking import group_by_user, top_n_from_scores

L1 = 0.01
L2 = 0.01
MAX_ITER = 500
TOL = 1e-4
def soft_threshold(value, threshold):
    """Soft-thresholding operator for L1."""
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


def elastic_net_fit(X, y, l1, l2, max_iter=MAX_ITER, tol=TOL):
    """Coordinate descent for Elastic Net; returns weight vector w."""
    n_samples, n_features = X.shape
    if n_features == 0:
        return np.array([])

    w = np.zeros(n_features)
    col_norm_sq = (X * X).sum(axis=0) / n_samples + l2

    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(n_features):
            if col_norm_sq[j] == 0.0:
                w[j] = 0.0
                continue
            residual = y - X @ w + X[:, j] * w[j]
            rho = X[:, j] @ residual / n_samples
            w[j] = soft_threshold(rho, l1) / col_norm_sq[j]

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


def center_train_matrix(train_matrix):
    """User mean-center; NaN -> 0 for dot products."""
    user_means = np.nanmean(train_matrix, axis=1)
    r_centered = train_matrix - user_means[:, np.newaxis]
    r_centered = np.where(np.isnan(r_centered), 0.0, r_centered)
    return r_centered, user_means


def train_slim(train_matrix, l1=L1, l2=L2):
    """Train item-item weight matrix W (column-wise Elastic Net)."""
    r_centered, _ = center_train_matrix(train_matrix)
    n_users, n_items = train_matrix.shape
    W = np.zeros((n_items, n_items))

    for j in tqdm(range(n_items), desc="SLIM columns", unit="item"):
        rated = ~np.isnan(train_matrix[:, j])
        if rated.sum() < 2:
            continue

        y = r_centered[rated, j]
        other_cols = [c for c in range(n_items) if c != j]
        X_all = r_centered[rated][:, other_cols]
        active_mask = (X_all != 0).any(axis=0)
        if not active_mask.any():
            continue
        X = X_all[:, active_mask]
        w = elastic_net_fit(X, y, l1, l2)
        active_cols = [other_cols[i] for i, m in enumerate(active_mask) if m]
        for idx, col_i in enumerate(active_cols):
            W[col_i, j] = w[idx]

    return W


def predict_score(user_row_centered, W, user_mean, item_idx):
    """Predict rating for one item."""
    return user_mean + user_row_centered @ W[:, item_idx]


def predict_scores(user_row_centered, W, user_mean, item_indices):
    """Predict scores for candidate item indices."""
    return {
        item_idx: predict_score(user_row_centered, W, user_mean, item_idx)
        for item_idx in item_indices
    }


def evaluate_slim(result, n_values=None, l1=L1, l2=L2):
    """Return {n: {hit_rate, precision}} over test users."""
    if n_values is None:
        n_values = N_VALUES

    train_matrix = result["train_matrix"]
    item_id_to_idx = result["item_id_to_idx"]
    train_user_id_to_idx = result["train_user_id_to_idx"]

    print("Training SLIM (~9.7k item columns; can take a long time)...")
    W = train_slim(train_matrix, l1=l1, l2=l2)

    r_centered, user_means = center_train_matrix(train_matrix)
    idx_to_movie_id = {idx: mid for mid, idx in item_id_to_idx.items()}
    test_by_user = group_by_user(result["test"])

    scores_by_user = {}
    for user_id, user_rows in tqdm(
        test_by_user.items(), desc="SLIM predict", unit="user"
    ):
        u_idx = train_user_id_to_idx[user_id]
        user_row = r_centered[u_idx]
        mean_u = user_means[u_idx]
        item_indices = [item_id_to_idx[r[MOVIEID_IDX]] for r in user_rows]
        scores_by_user[user_id] = {
            idx_to_movie_id[idx]: predict_score(user_row, W, mean_u, idx)
            for idx in item_indices
        }

    metrics = {}
    for n in n_values:
        recommended_by_user = {
            user_id: top_n_from_scores(scores_by_user[user_id], n)
            for user_id in test_by_user
        }
        metrics[n] = average_metrics_per_user(
            recommended_by_user, test_by_user, n
        )

    return metrics


def print_slim_results(metrics, n_values=None):
    """Print metric table for SLIM."""
    if n_values is None:
        n_values = sorted(metrics.keys())

    print("=== System 4: SLIM ===\n")
    print("User mean-centered; Elastic Net (numpy coordinate descent)\n")
    print(f"{'n':<6}{'HitRate@n':<14}{'Precision@n'}")
    for n in n_values:
        m = metrics[n]
        print(f"{n:<6}{m['hit_rate']:<14.4f}{m['precision']:.4f}")
