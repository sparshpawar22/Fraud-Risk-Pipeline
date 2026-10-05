# Methodology

## 1. Why this is hard

The dataset gives labels for **defaulters** (`default_ind = 1`), not fraudsters. In reality, only some
defaulters are fraudsters — many are just over-leveraged, hit by financial hardship, or made poor but
non-fraudulent decisions. Treating every defaulter as fraud would mix two different behavioral
populations together and mislabel a lot of "honest but unlucky" customers as fraud.

So the real target — fraud — has no direct supervised signal. Fraud is a minority *within* a minority:

```
Overall population  →  many good customers
  └── Defaulters      →  a small subset (0.58% of applicants here)
        └── Fraudsters → an even smaller, unlabeled subset
```

This is a weakly-supervised + anomaly-detection problem: default status is a *weak proxy* for
risk, and fraud has to be discovered as behavioral anomalies within the high-risk region that proxy
points to.

## 2. Two-stage design

**Stage 1 (supervised):** use default labels as a weak signal to identify high-risk / suspicious
behavior — not to directly predict fraud. Output: a filtered high-risk subset where fraud is more
likely to be concentrated.

**Stage 2 (unsupervised):** within that subset, run anomaly detection to isolate accounts that look
different even from other high-risk accounts.

```
Full population → Stage 1 (XGBoost + IV + SHAP) → high-risk subset → Stage 2 (Isolation Forest + K-Means) → ranked fraud candidates
```

This avoids the two failure modes of using either approach alone:
- **Supervised alone** learns "who doesn't pay back," which conflates fraud with ordinary financial
  hardship.
- **Unsupervised alone** on the full population is noisy — most behavioral diversity in a general
  population isn't fraud, so a full-population anomaly search returns a high false-positive rate with
  little business focus.

## 3. Stage 1 in detail

**Feature families used:** transactional (spend, payments, attempted/declined transactions,
merchant category), behavioral (trade-line counts, delinquency history), and identity/bureau
(address changes, ID-to-address matches, bureau score, bureau mismatches).

**Information Value (IV) feature selection.** IV is a standard credit-risk technique for ranking how
predictive a variable is of a binary outcome. It's computed **on the training split only** — this is a
deliberate fix from the original version of this project, which computed IV on the full dataset before
splitting, leaking test-set information into feature selection.

Features are bucketed as strong (IV ≥ 0.30), medium (0.10 ≤ IV < 0.30), or weak (IV < 0.10); Stage 1
trains on strong + medium features only.

**XGBoost**, trained with SMOTE-resampled training data (SMOTE fit on the training split only, never
touching the test split), produces a `stage1_risk_score` per applicant — higher means more
default-like/risky. This score, not `default_ind` directly, is what defines Stage 2's search region
(the top 10% by score).

**SHAP** is used to check the model aligns with business intuition (e.g. address instability and
identity mismatches should raise risk) and to give per-applicant explanations for why an account
landed in the high-risk region.

## 4. Stage 2 in detail

Stage 2 operates only on the Stage-1 high-risk subset and does not use `default_ind` at all.

**Isolation Forest (global anomalies).** Randomly partitions the feature space; points that isolate
with fewer splits are more anomalous. Captures accounts whose overall pattern (spend, merchant
mix, geography) is far from the bulk of the high-risk group.

**K-Means (local/density anomalies).** Clusters the high-risk subset into K groups; an account's
distance to its own cluster centroid measures how unusual it is *relative to its peers*, catching
subtler anomalies that aren't extreme globally but are odd within their local behavioral neighborhood.

**Blended score:**

```
fraud_score = alpha * global_score + (1 - alpha) * local_score
```

with both scores min-max normalized to [0, 1] before blending (`alpha = 0.5` by default in
`src/stage2_unsupervised.py`).

## 5. Honest evaluation

There is no ground-truth fraud label in this dataset, so Stage 2's output cannot be validated in the
usual sense. What this repo does instead: check whether `fraud_score` concentrates the one weak
signal available (`default_ind`), as a sanity check, not a validation.

Reproducible results from `python src/run_pipeline.py` (see `outputs/reports/` for the raw
numbers):

- Stage 1 test-set metrics: 99.47% accuracy, 0.954 ROC-AUC, 32.9% recall / 58.5% precision on the
  positive (default) class.
- Stage 1 high-risk subset (top 10% by score): default rate 5.6%, a 9.7x lift over the 0.58%
  population base rate.
- Stage 2 top-100 by `fraud_score`: default rate 13.0% (22.4x lift).
- Stage 2 top-500 by `fraud_score`: default rate 18.2% (31.3x lift).

These numbers are real and reproducible, and noticeably more modest than an earlier draft of this
project's write-up claimed (which described ~75% recall and a ~30% reduction in false positives —
figures that didn't correspond to anything the code actually produced). Given the severity of the
class imbalance (0.58% base rate) and the fact that `default_ind` is a noisy proxy for fraud, treat
`fraud_score` as a way to prioritize a manageable number of accounts for human review, not as a
final fraud/not-fraud decision.

## 6. Suggested next steps

- Encode categorical features (`prod_name`, `acq_channel`, `state_code`, `bureau_src`,
  `appl_month`) instead of dropping them from the numeric-only Stage 1/2 models.
- If any confirmed fraud cases become available (even a small hand-labeled sample), use them to
  actually validate — rather than sanity-check — Stage 2's ranking.
- Tune the Stage 2 hyperparameters (`top_pct` high-risk cutoff, `n_clusters`, blend `alpha`) against a
  labeled validation set once one exists, rather than using the current reasonable defaults.
- Consider cost-sensitive learning or threshold tuning on Stage 1's `predict_proba` output, since the
  business cost of a missed fraud case and a false alarm are unlikely to be symmetric.
