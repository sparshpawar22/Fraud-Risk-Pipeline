"""
stage2_unsupervised.py
------------------------
Stage 2 of the pipeline: fraud discovery via anomaly detection on the
high-risk subset produced by Stage 1.

FIX vs. the original project: `Unsupervised/code.ipynb` in the original
repo was an empty notebook - the Isolation Forest + K-Means dual anomaly
score described in the write-up was never actually implemented. This
module implements it for real:

- IsolationForest gives a GLOBAL anomaly score (how far a point is from
  the bulk of the high-risk population overall).
- KMeans gives a LOCAL anomaly score (distance from a point to its own
  cluster's centroid, i.e. how unusual it is relative to its peer group).
- Both scores are min-max normalized to [0, 1] and blended into a single
  fraud_score = alpha * global + (1 - alpha) * local.

This is unsupervised by construction - it does not use default_ind at all,
only the feature values of the high-risk subset. That keeps it honest: it
is a ranking/prioritization tool for fraud analysts, not a validated
fraud classifier (there are no ground-truth fraud labels in this dataset
to validate against).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def compute_dual_anomaly_scores(
    X_high_risk: pd.DataFrame,
    n_clusters: int = 8,
    contamination: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute global (IsolationForest) + local (KMeans distance-to-centroid)
    anomaly scores for a high-risk subset. Returns a DataFrame indexed like
    X_high_risk with columns: global_score, local_score (both 0-1, higher =
    more anomalous).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_high_risk)

    # --- Global anomalies: Isolation Forest ---
    iso = IsolationForest(
        n_estimators=300, contamination=contamination, random_state=random_state, n_jobs=-1
    )
    iso.fit(X_scaled)
    # score_samples: higher = more normal. Flip sign so higher = more anomalous.
    global_raw = -iso.score_samples(X_scaled)
    global_score = _minmax(global_raw)

    # --- Local anomalies: KMeans distance-to-own-centroid ---
    k = min(n_clusters, max(2, len(X_high_risk) // 10))
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_labels = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_
    dists = np.linalg.norm(X_scaled - centroids[cluster_labels], axis=1)
    local_score = _minmax(dists)

    return pd.DataFrame(
        {
            "global_score": global_score,
            "local_score": local_score,
            "cluster": cluster_labels,
        },
        index=X_high_risk.index,
    )


def blend_fraud_score(scores_df: pd.DataFrame, alpha: float = 0.5) -> pd.Series:
    """fraud_score = alpha * global_score + (1 - alpha) * local_score."""
    return alpha * scores_df["global_score"] + (1 - alpha) * scores_df["local_score"]
