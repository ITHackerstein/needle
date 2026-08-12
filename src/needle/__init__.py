import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from needle import plots
from needle.common import cross_validation_split, probability, ranking
from needle.config import MIN_PRECISION, OBJECTIVES
from needle.data import load_data, temporal_split, extract_features
from needle.models import CANDIDATE_PIPELINES
from needle.evaluate import cross_validate_candidates, recall_at_precision
from needle.threshold import (
    apply_threshold,
    review_cost_sensitivity,
    select_threshold,
    selection_scores,
    threshold_from_rate,
    threshold_table
)
from needle.calibrate import calibrated_candidate, calibration_report
from needle.tune import (
    DEFAULT_TUNED,
    load_tuning,
    result_path,
    save_tuning,
    tune_model,
    tuned_candidate
)

OBJECTIVE = "cost"
POINT_COLUMNS = ["threshold", "alert_rate", "alerts_per_day", "precision", "recall", "cost"]

def _holdout(candidate, X_first, y_first, X_second, y_second) -> tuple:
    print(f"\n=== temporal holdout: {candidate.label()} ===")
    model = candidate.build().fit(X_first, y_first)
    scores, probabilities = ranking(model, X_second), probability(model, X_second)

    for name, value in (
        ("pr_auc", average_precision_score(y_second, scores)),
        ("roc_auc", roc_auc_score(y_second, scores)),  # secondary, for comparability only
        (f"recall_at_p{int(MIN_PRECISION * 100)}", recall_at_precision(y_second, scores))
    ):
        print(f"  {name:22} {value:.4f}")
    return scores, probabilities

def _tuned_candidates(X, y, model_names=DEFAULT_TUNED, retune: bool = False) -> list:
    candidates = []
    for model_name in model_names:
        path = result_path(model_name)
        if not retune and path.exists():
            result = load_tuning(model_name)
            print(f"  {model_name:20} reusing {path} ({result.n_trials} trials, "
                  f"search pr_auc={result.search_pr_auc:.4f})")
        else:
            print(f"\n  ### searching {model_name} - this is the slow part ###")
            result = tune_model(model_name, X, y, revalidate=False)
            print(f"  wrote {save_tuning(result)}")

        candidates.append(tuned_candidate(result))
    return candidates

def _operating_point(candidate, X, y, amounts) -> tuple[dict, "pd.Series"]:
    print("\n=== threshold selection (day 1, out-of-fold) ===")
    oof_ranking, oof_probabilities = selection_scores(candidate, X, y)
    validation_scores = oof_ranking if oof_probabilities is None else oof_probabilities

    choices = pd.DataFrame(
        select_threshold(y, validation_scores, amounts, objective=objective)
        for objective in OBJECTIVES
    )
    print(choices.to_string(index=False))

    print("\n  where the cost optimum moves with the price of one review:")
    print(review_cost_sensitivity(y, validation_scores, amounts).to_string(index=False))

    chosen = choices[choices["objective"] == OBJECTIVE].iloc[0].to_dict()
    print(f"\n  shipping '{OBJECTIVE}': threshold {chosen['threshold']:.6f}, "
          f"the top {chosen['alert_rate'] * 100:.3f}% of transactions")
    return chosen, validation_scores

def _transfer(chosen: dict, y_true, y_score, amounts) -> dict:
    print("\n=== the chosen point, carried to day 2 ===")
    by_rate = apply_threshold(
        y_true, y_score, threshold_from_rate(y_score, chosen["alert_rate"]), amounts
    )
    rows = {
        "expected (day 1 out-of-fold)": chosen,
        "achieved (same threshold)": apply_threshold(y_true, y_score, chosen["threshold"], amounts),
        "achieved (same alert rate)": by_rate
    }
    print(pd.DataFrame(rows).T[POINT_COLUMNS].astype(float).to_string())
    return by_rate

def _calibration(candidate, X_first, y_first, X_second, y_second, raw_probabilities):
    if raw_probabilities is None:
        print("\n=== calibration: skipped, the winning model emits no probabilities ===")
        return None

    print("\n=== calibration (fit on day 1, checked on day 2) ===")
    model = calibrated_candidate(candidate).fit(X_first, y_first)
    calibrated_probabilities = model.predict_proba(X_second)[:, 1]

    for label, scores in (("raw", raw_probabilities), ("sigmoid", calibrated_probabilities)):
        table, brier = calibration_report(y_second, scores)
        print(f"\n  {label}: brier={brier:.6f} "
              f"pr_auc={average_precision_score(y_second, scores):.4f} "
              f"mean_score={scores.mean():.6f} vs base rate {y_second.mean():.6f}")
        print(table.to_string(index=False))
    return calibrated_probabilities

def main(retune: bool = False) -> None:
    pd.set_option("display.width", 200)

    df = load_data("dataset/creditcard.csv")
    split = temporal_split(df["Time"])
    X, y = extract_features(df)

    X_first, y_first = X.iloc[split.train], y.iloc[split.train]
    X_second, y_second = X.iloc[split.test], y.iloc[split.test]
    amounts_first, amounts_second = df["Amount"].iloc[split.train], df["Amount"].iloc[split.test]
    print(f"day 1: {len(X_first):,} rows, {int(y_first.sum())} frauds")
    print(f"day 2: {len(X_second):,} rows, {int(y_second.sum())} frauds")

    cv = cross_validation_split()

    print("\n=== hyperparameter search (day 1 only) ===")
    tuned = _tuned_candidates(X_first, y_first, retune=retune)
    candidates = list(CANDIDATE_PIPELINES) + tuned
    by_label = {candidate.label(): candidate for candidate in candidates}

    print("\n=== model selection (day 1 only) ===")
    rows = []
    for candidate in candidates:
        row = cross_validate_candidates(X_first, y_first, (candidate,), cv=cv)
        print(f"  {row.at[0, 'label']:36} "
              f"pr_auc={row.at[0, 'pr_auc_mean']:.4f} ± {row.at[0, 'pr_auc_std']:.4f} "
              f"({row.at[0, 'fit_seconds']:.1f}s/fit)")
        rows.append(row)

    leaderboard = pd.concat(rows, ignore_index=True).sort_values(
        "pr_auc_mean", ascending=False, ignore_index=True
    )
    print()
    print(leaderboard.to_string())

    best = leaderboard.iloc[0]
    winner = by_label[best["label"]]
    chosen, validation_scores = _operating_point(winner, X_first, y_first, amounts_first)

    scores, probabilities = _holdout(winner, X_first, y_first, X_second, y_second)
    print(f"\n  CV pr_auc {best['pr_auc_mean']:.4f} -> holdout, gap is the finding")

    units = "probability" if probabilities is not None else "decision function"
    table_scores = scores if probabilities is None else probabilities
    achieved = _transfer(chosen, y_second, table_scores, amounts_second)

    print(f"\n=== operating points (day 2, thresholds in {units} units) ===")
    print(threshold_table(y_second, table_scores, amounts_second, n_rows=12).to_string())

    calibrated_probabilities = _calibration(
        winner, X_first, y_first, X_second, y_second, probabilities
    )

    print("\n=== figures ===")
    figures = [
        plots.precision_recall(
            {
                "day 1 (out-of-fold)": (y_first, validation_scores),
                "day 2 (holdout)": (y_second, table_scores)
            },
            chosen=achieved
        ),
        plots.cost_vs_alerts(y_second, table_scores, amounts_second, chosen=achieved)
    ]
    if calibrated_probabilities is not None:
        figures.append(plots.reliability({
            "raw": (y_second, probabilities),
            "sigmoid": (y_second, calibrated_probabilities)
        }))
    for path in figures:
        print(f"  wrote {path}")
