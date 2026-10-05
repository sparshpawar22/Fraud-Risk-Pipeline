"""
data_prep.py
------------
Loads the raw applicant dataset and performs light, well-documented cleaning.

Design notes / fixes vs. the original project:
- Sentinel values: several identity/address-matching columns use -1 to mean
  "not applicable" (e.g. no bureau hit). We leave these as -1 for tree models
  (XGBoost splits on them natively and they carry real signal - "no bureau
  hit" is itself predictive) but flag them so downstream code is explicit
  about it rather than accidentally treating -1 as a real numeric value in
  distance-based models (KMeans, IsolationForest use scaled data instead).
- unique_identifier is kept as an ID column, never used as a model feature.
- appl_month is categorical (Jan-Jun); we leave it out of the numeric-only
  Stage 1 model for simplicity, matching the original project's scope, but
  it is retained in the returned frame so it's available for reporting.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

TARGET_COL = "default_ind"
ID_COL = "unique_identifier"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw CSV."""
    df = pd.read_csv(path)
    return df


def get_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns, excluding the ID and target."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in (TARGET_COL, ID_COL)]


def basic_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Small data-quality summary: missing %, sentinel (-1) counts, dtype."""
    numeric_cols = get_numeric_feature_cols(df)
    rows = []
    for col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        sentinel_count = int((df[col] == -1).sum()) if col in numeric_cols else 0
        rows.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_pct": round(missing_pct, 2),
                "sentinel_minus1_count": sentinel_count,
                "n_unique": df[col].nunique(dropna=True),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False).reset_index(drop=True)
