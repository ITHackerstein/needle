# Project Plan — Credit Card Fraud Detection

## Dataset facts

[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), at `dataset/creditcard.csv` (151 MB).

- 284,807 rows, 31 columns: `Time`, `V1`–`V28`, `Amount`, `Class`.
- 492 frauds = **0.173%** of rows (1:578 imbalance).
- `Time` spans exactly 2 days (0–172,792 seconds).
- 773 fully-duplicated rows.
- `V1`–`V28` are PCA components of undisclosed original features; `Time` and `Amount` are raw.

## 0. Framing: get the metric right first

This is the decision that makes or breaks the project. Accuracy is meaningless (99.83% by predicting "never fraud"), and **ROC-AUC is misleading** here — with 578 negatives per positive, a large FPR change barely moves the ROC curve.

- **Primary metric: PR-AUC / average precision.** Threshold-free, sensitive to the minority class.
- **Reporting metric: recall @ fixed precision** (e.g. recall at precision ≥ 0.90), or precision/recall at a fixed alert budget (e.g. "we can manually review 100 transactions/day"). This is what a fraud team actually buys.
- **Optional cost metric:** `cost = C_fn · FN + C_fp · FP`. A missed fraud costs roughly the transaction amount; a false positive costs a fixed review. `Amount` is in the data, so this is computable and far more persuasive than F1.
- Report ROC-AUC only as a secondary number, for comparability with published results.

### Why ROC-AUC is misleading here — worked example

The root cause is the denominators. ROC's x-axis is `FPR = FP / all_negatives`, and there are 284,315 negatives. Precision's denominator is `FP + TP`, where `TP` caps at 492. The same false positives get divided by a huge number in one metric and a tiny one in the other.

**Scaled-down intuition.** Take 10 frauds and 5,780 legit transactions (same 1:578 ratio). Two models, both catching 8 of the 10 frauds:

- Model A fires 8 false alarms → FPR = 8/5780 = **0.14%**, precision = 8/16 = **50%**
- Model B fires 240 false alarms → FPR = 240/5780 = **4.2%**, precision = 8/248 = **3.2%**

On a ROC plot both curves hug the top-left corner; the gap between them is 4 percentage points of x-axis. But A gives an analyst one real fraud per two alerts, while B gives 30 false alarms per catch and is unusable. ROC-AUC ranks them as near-equals.

**Full-dataset version.** Fix recall at 0.80 (394 of 492 frauds caught) and vary FPR:

| FPR | False positives | Precision |
|---|---|---|
| 0.0005 | 142 | 73.5% |
| 0.001 | 284 | 58.1% |
| 0.005 | 1,422 | 21.7% |
| 0.01 | 2,843 | 12.2% |
| 0.05 | 14,216 | 2.7% |

The entire range from "deployable" to "worthless" fits inside FPR ∈ [0.0005, 0.05] — the leftmost 5% of the ROC curve. ROC-AUC integrates over the full [0, 1] range, so ~95% of the reported number comes from a region we would never operate in. Two models whose ROC-AUCs differ by 0.002 can differ by 6× in false-alarm volume at the recall we actually ship at.

Stated concretely: an FPR of 5% sounds harmless, and in a balanced problem it is. Here it means 14,216 false alarms to catch 394 frauds — roughly 7,100 wasted investigations per day against a team that can review maybe a hundred.

## 1. Data foundation

1. Load with explicit dtypes (`Class` is quoted in the CSV, so it parses as string by default — cast to int8).
2. **Drop the 773 exact duplicates before splitting.** Otherwise identical rows land in both train and test and inflate scores.
3. EDA, focused rather than exhaustive:
   - Class counts; `Amount` distribution by class (fraud amounts are systematically different); fraud rate vs. hour-of-day derived from `Time`.
   - Per-feature separation between classes via a KS or per-feature-AUC ranking — better than 28 histograms.
   - No missing values expected; assert it rather than assume.
4. Features:
   - `V1`–`V28` are already PCA outputs → **no further scaling, no further PCA**.
   - `Amount` is heavy-tailed → `log1p` or a robust scaler.
   - Do not use `Time` raw (an absolute offset that will not generalize). Derive `hour_of_day = (Time / 3600) % 24` and drop the raw column.

## 2. Validation strategy

Two splits, serving different purposes:

- **Model selection: repeated stratified 5-fold CV.** With 492 positives each fold holds ~98 frauds, so variance is large. Use 3–5 repeats and report mean ± std. Single-split numbers on this dataset are not trustworthy.
- **Final estimate: temporal holdout.** Sort by `Time`, train on day 1, test on day 2. Fraud patterns drift, and this is the only split that speaks to production behaviour. Expect it to score *worse* than CV — that gap is a finding, not a bug.

**The critical rule:** any resampling (SMOTE, undersampling) and any scaler fitting happens **inside the CV fold, on the training part only**. Use an `imblearn.Pipeline` so this is structural rather than something to remember. Oversampling before splitting is the most common error in published notebooks on this dataset and produces near-perfect fake scores.

Fix a seed everywhere and record it with results.

## 3. Modeling ladder

Climb in order; stop when gains flatten.

| Step | Model | Purpose |
|---|---|---|
| 0 | Stratified dummy | Sanity floor for PR-AUC (≈0.0017) |
| 1 | Logistic regression, `class_weight='balanced'` | Interpretable baseline; often surprisingly strong here |
| 2 | Random forest / ExtraTrees, `class_weight='balanced_subsample'` | Non-linear baseline |
| 3 | **LightGBM or XGBoost with `scale_pos_weight`** | Expected winner; tune with Optuna against PR-AUC |
| 4 | Unsupervised comparison: Isolation Forest, or an autoencoder trained on legitimate transactions only | Answers "do we even need labels?" — and it is the setup that generalizes to novel fraud |

Imbalance handling — test these as pipeline variants rather than assuming a winner:

- class weights / `scale_pos_weight` (usually the best cost/benefit),
- random undersampling of the majority,
- SMOTE / ADASYN.

SMOTE tends to underperform plain class weighting on this dataset, because it interpolates in a PCA space where the minority class is not locally dense. Worth demonstrating rather than asserting.

## 4. Threshold selection

Train probability-ranked models, then pick the operating point as an explicit, separate step on validation data — never leave it at 0.5.

Deliverables: a precision-recall curve with the chosen threshold marked, plus a table of (threshold → alerts/day, precision, recall, cost).

Also check calibration (`CalibratedClassifierCV`, reliability curve). If a score gates a manual review queue, it should mean something.

## 5. Interpretation & deliverables

- SHAP values on the winning model. Even with anonymized `V*` features, the ranking and interaction structure is informative, and it is the standard expectation for a credit-risk model.
- Confusion matrix at the chosen threshold, plus a look at the actual missed frauds (if the FNs are low-`Amount`, the cost impact is small).
- Written summary: metric choice and why; leakage precautions; the CV vs. temporal-holdout gap; chosen threshold and its business reading; honest limitations (492 positives, 2 days of data, anonymized features, no way to validate against real drift).

## 6. Code layout

The repo already uses a `src/` layout with `uv`. Dependencies to add:

```
pandas numpy scikit-learn imbalanced-learn lightgbm optuna shap matplotlib
pytest jupyter   # dev
```

```
src/machine_learning_project/
  data.py          # load, dedupe, feature engineering
  splits.py        # stratified CV + temporal holdout
  models.py        # pipeline factories per candidate
  evaluate.py      # PR-AUC, recall@precision, cost, plots
  tune.py          # Optuna search
  cli.py           # entry point wired to [project.scripts]
notebooks/         # exploration only — logic lives in src/
reports/           # figures + written summary
dataset/           # gitignored, 151 MB
```

Notebooks for exploration; anything reused moves into `src/` so results stay reproducible.

## Open decisions

1. **How much does the unsupervised branch (3.4) matter?** For a portfolio/learning project it is the most distinctive part; for the best classifier alone it is optional.
2. `dataset/` is untracked and 151 MB — add it to `.gitignore` with a download note in the README rather than committing it.
