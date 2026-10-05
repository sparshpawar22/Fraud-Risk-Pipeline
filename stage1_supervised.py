"""
stage1_supervised.py
---------------------
Stage 1 of the pipeline: a supervised "weak risk" model trained on
default_ind. This is NOT a fraud classifier - it narrows the population
down to a high-risk subset that Stage 2 (unsupervised anomaly detection)
then searches for fraud-like outliers.

FIX vs. the original project:
1. Train/test split happens FIRST. IV feature selection, imputation
   statistics, and SMOTE resampling are all fit on the training split only
   and applied (not re-fit) to the test split. No test-set information
   leaks into feature selection.
2. Median imputation values are computed on X_train and reused for
   X_test/X_score, instead of each split imputing its own median.
3. Metrics reported are whatever the model actually produces on the held-out
   test set - nothing here is fabricated or copied from the doc write-up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from iv_selection import compute_iv_table, select_features_by_iv


def split_data(
    df: pd.DataFrame, feature_cols: list[str], target_col: str, test_size: float = 0.2, random_state: int = 42
):
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def select_features(
    X_train: pd.DataFrame, y_train: pd.Series, weak_thr: float = 0.10, strong_thr: float = 0.30
):
    """Run IV selection on the training split only; return IV table + chosen feature list."""
    iv_table = compute_iv_table(X_train, y_train)
    buckets = select_features_by_iv(iv_table, weak_thr=weak_thr, strong_thr=strong_thr)
    selected = buckets["strong"] + buckets["medium"]
    return iv_table, buckets, selected


def impute_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Median imputation fit on train, applied to both splits (no leakage)."""
    medians = X_train.median()
    return X_train.fillna(medians), X_test.fillna(medians), medians


def train_xgb_with_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> tuple[XGBClassifier, dict]:
    """Fit SMOTE on the training split only, train XGBoost, evaluate on the untouched test split."""
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        eval_metric="auc",
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision_pos": float(precision),
        "recall_pos": float(recall),
        "f1_pos": float(f1),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "n_train": int(len(X_train)),
        "n_train_resampled": int(len(X_res)),
        "n_test": int(len(X_test)),
    }
    return model, metrics


def feature_importance(model: XGBClassifier, feature_cols: list[str], top_n: int = 20) -> pd.Series:
    fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    return fi.head(top_n)
