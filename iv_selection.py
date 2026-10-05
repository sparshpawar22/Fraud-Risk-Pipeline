"""
iv_selection.py
----------------
Information Value (IV) based feature selection.

FIX vs. the original project: IV must be computed on the TRAINING split
only. The original notebook computed IV on the full dataset (all 62,484
rows) before ever splitting into train/test, which leaks test-set label
information into the feature-selection step and inflates downstream
metrics. Every function here takes X_train/y_train explicitly - there is
no code path that lets you compute IV against data that includes the
test set.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _iv_for_series(x: pd.Series, y: pd.Series, bins: int = 10, eps: float = 1e-6) -> float:
    """IV for a single numeric feature vs. a binary target."""
    ser = x.copy()
    if ser.nunique(dropna=True) > bins:
        try:
            binned = pd.qcut(ser, q=bins, duplicates="drop")
        except Exception:
            binned = pd.cut(ser, bins=bins)
    else:
        binned = ser.astype(object)

    binned = binned.astype(str).fillna("Missing")
    grp = pd.concat([binned.rename("bin"), y.rename("target")], axis=1).groupby("bin")["target"].agg(
        ["count", "sum"]
    )
    grp = grp.rename(columns={"sum": "events", "count": "total"})
    grp["non_events"] = grp["total"] - grp["events"]

    total_events = grp["events"].sum()
    total_non_events = grp["non_events"].sum()
    if total_events == 0 or total_non_events == 0:
        return np.nan

    event_rate = (grp["events"] / total_events).replace(0, eps)
    non_event_rate = (grp["non_events"] / total_non_events).replace(0, eps)
    woe = np.log(event_rate / non_event_rate)
    iv_bin = (event_rate - non_event_rate) * woe
    return float(iv_bin.sum())


def compute_iv_table(X_train: pd.DataFrame, y_train: pd.Series, bins: int = 10) -> pd.DataFrame:
    """IV for every numeric column in X_train, computed against y_train only."""
    rows = []
    for col in X_train.columns:
        if X_train[col].nunique(dropna=True) <= 1:
            continue
        iv = _iv_for_series(X_train[col], y_train, bins=bins)
        rows.append({"feature": col, "iv": iv})
    return pd.DataFrame(rows).dropna(subset=["iv"]).sort_values("iv", ascending=False).reset_index(drop=True)


def select_features_by_iv(
    iv_table: pd.DataFrame, weak_thr: float = 0.10, strong_thr: float = 0.30
) -> dict[str, list[str]]:
    """Bucket features into strong / medium / weak by IV threshold."""
    strong = iv_table[iv_table["iv"] >= strong_thr]["feature"].tolist()
    medium = iv_table[(iv_table["iv"] >= weak_thr) & (iv_table["iv"] < strong_thr)]["feature"].tolist()
    weak = iv_table[iv_table["iv"] < weak_thr]["feature"].tolist()
    return {"strong": strong, "medium": medium, "weak": weak}
