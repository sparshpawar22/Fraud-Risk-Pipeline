"""
run_pipeline.py
----------------
End-to-end run of the corrected two-stage fraud detection pipeline.
Run from the project root with:

    python src/run_pipeline.py

Produces (all under outputs/):
- reports/data_quality.csv
- reports/iv_table.csv
- reports/stage1_metrics.json
- reports/stage1_feature_importance.csv
- reports/fraud_candidates.csv         (Stage 2 ranked output)
- shap_plots/shap_summary_group_*.png
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(__file__))

from data_prep import load_data, get_numeric_feature_cols, basic_data_quality_report, TARGET_COL
from stage1_supervised import split_data, select_features, impute_train_test, train_xgb_with_smote, feature_importance
from stage2_unsupervised import compute_dual_anomaly_scores, blend_fraud_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "66acd42635424_r3_data.csv")
OUT_REPORTS = os.path.join(ROOT, "outputs", "reports")
OUT_SHAP = os.path.join(ROOT, "outputs", "shap_plots")


def main():
    os.makedirs(OUT_REPORTS, exist_ok=True)
    os.makedirs(OUT_SHAP, exist_ok=True)

    print("[1/6] Loading data...")
    df = load_data(DATA_PATH)
    print(f"    shape={df.shape}")

    dq = basic_data_quality_report(df)
    dq.to_csv(os.path.join(OUT_REPORTS, "data_quality.csv"), index=False)

    numeric_cols = get_numeric_feature_cols(df)

    print("[2/6] Train/test split (BEFORE any feature selection - fixes leakage)...")
    X_train, X_test, y_train, y_test = split_data(df, numeric_cols, TARGET_COL)
    print(f"    train={X_train.shape}, test={X_test.shape}")

    print("[3/6] IV feature selection (train split only)...")
    iv_table, buckets, selected_features = select_features(X_train, y_train)
    iv_table.to_csv(os.path.join(OUT_REPORTS, "iv_table.csv"), index=False)
    print(f"    strong={len(buckets['strong'])} medium={len(buckets['medium'])} weak={len(buckets['weak'])}")
    print(f"    using {len(selected_features)} strong+medium features for Stage 1 model")

    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]
    X_train_imp, X_test_imp, medians = impute_train_test(X_train_sel, X_test_sel)

    print("[4/6] Training Stage 1 XGBoost (SMOTE fit on train only)...")
    model, metrics = train_xgb_with_smote(X_train_imp, y_train, X_test_imp, y_test)
    with open(os.path.join(OUT_REPORTS, "stage1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"    accuracy={metrics['accuracy']:.4f} roc_auc={metrics['roc_auc']:.4f} "
          f"precision(1)={metrics['precision_pos']:.3f} recall(1)={metrics['recall_pos']:.3f} "
          f"f1(1)={metrics['f1_pos']:.3f}")

    fi = feature_importance(model, selected_features)
    fi.to_csv(os.path.join(OUT_REPORTS, "stage1_feature_importance.csv"), header=["importance"])

    print("[5/6] SHAP explainability on test set...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_imp)
    features = list(X_test_imp.columns)
    group_size = 10
    groups = [features[i : i + group_size] for i in range(0, len(features), group_size)]
    for i, group in enumerate(groups, start=1):
        idx = [X_test_imp.columns.get_loc(c) for c in group]
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[:, idx], X_test_imp[group], show=False)
        plt.title(f"SHAP Summary - Stage 1 Model (group {i})")
        fname = os.path.join(OUT_SHAP, f"shap_summary_group_{i}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"    saved {len(groups)} SHAP summary plots to outputs/shap_plots/")

    print("[6/6] Stage 2: dual anomaly scoring on the high-risk subset...")
    # Score the FULL dataset with the trained Stage 1 model to define "high risk",
    # rather than only ever looking at labeled defaulters - this lets Stage 2 also
    # surface fraud-like behavior among applicants who haven't defaulted yet.
    X_all_sel = df[selected_features].fillna(medians)
    risk_proba = model.predict_proba(X_all_sel)[:, 1]
    df_scored = df.copy()
    df_scored["stage1_risk_score"] = risk_proba

    top_pct = 0.10  # top 10% by predicted risk defines the Stage-2 search region
    threshold = df_scored["stage1_risk_score"].quantile(1 - top_pct)
    high_risk = df_scored[df_scored["stage1_risk_score"] >= threshold].copy()
    print(f"    high-risk subset: {len(high_risk)} rows (top {top_pct*100:.0f}% by Stage 1 risk score)")

    X_high_risk = high_risk[selected_features].fillna(medians)
    anomaly_scores = compute_dual_anomaly_scores(X_high_risk)
    high_risk = high_risk.join(anomaly_scores)
    high_risk["fraud_score"] = blend_fraud_score(high_risk, alpha=0.5)
    high_risk = high_risk.sort_values("fraud_score", ascending=False)

    out_cols = [
        "unique_identifier", "appl_month", "prod_name", "state_code", "default_ind",
        "stage1_risk_score", "global_score", "local_score", "cluster", "fraud_score",
    ]
    high_risk[out_cols].to_csv(os.path.join(OUT_REPORTS, "fraud_candidates.csv"), index=False)
    print(f"    wrote outputs/reports/fraud_candidates.csv ({len(high_risk)} ranked candidates)")

    print("[sanity check] Stage 2 fraud_score vs. default_ind (weak proxy - NOT ground-truth fraud)...")
    base_rate = float(df[TARGET_COL].mean())
    hr_rate = float(high_risk[TARGET_COL].mean())
    top100_rate = float(high_risk.sort_values("fraud_score", ascending=False).head(100)[TARGET_COL].mean())
    top500_rate = float(high_risk.sort_values("fraud_score", ascending=False).head(500)[TARGET_COL].mean())
    lift = {
        "population_default_rate": base_rate,
        "high_risk_subset_default_rate": hr_rate,
        "high_risk_subset_lift_x": hr_rate / base_rate,
        "top100_fraud_score_default_rate": top100_rate,
        "top100_fraud_score_lift_x": top100_rate / base_rate,
        "top500_fraud_score_default_rate": top500_rate,
        "top500_fraud_score_lift_x": top500_rate / base_rate,
        "note": (
            "default_ind is a proxy label, not verified fraud. This checks whether the "
            "unsupervised fraud_score concentrates the (weak) signal we do have; it is not "
            "a substitute for validating against real fraud outcomes."
        ),
    }
    with open(os.path.join(OUT_REPORTS, "stage2_sanity_check.json"), "w") as f:
        json.dump(lift, f, indent=2)
    print(f"    population default rate: {base_rate:.4f}")
    print(f"    high-risk subset default rate: {hr_rate:.4f} ({hr_rate/base_rate:.1f}x lift)")
    print(f"    top-100 fraud_score default rate: {top100_rate:.4f} ({top100_rate/base_rate:.1f}x lift)")
    print(f"    top-500 fraud_score default rate: {top500_rate:.4f} ({top500_rate/base_rate:.1f}x lift)")

    print("\nDone. See outputs/reports/ and outputs/shap_plots/ for all results.")


if __name__ == "__main__":
    main()
