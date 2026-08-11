import numpy as np
import pandas as pd
from needle.data import load_data, temporal_split, extract_features, cross_validation_split
from needle.models import CANDIDATES, make_model
from needle.evaluate import (
    candidate_label,
    cross_validate_candidates,
    probability,
    ranking,
    summarize,
    threshold_table,
    unpack_candidate
)
# Aliased on purpose: an unaliased `tune` would shadow the needle.tune submodule on
# the package itself, and `load` sitting next to load_data reads as the wrong thing.
from needle.tune import (
    DEFAULT_TUNED,
    result_path,
    load as load_tuning,
    save as save_tuning,
    tune as tune_model
)

def _holdout(candidate, X_first, y_first, X_second, y_second, amounts) -> tuple:
    """Returns (ranking scores, probabilities or None) - the first for ranking and
    fraud-overlap work, the second for anything that reports a threshold.
    """
    model_name, imbalance_method, params = unpack_candidate(candidate)
    print(f"\n=== temporal holdout: {candidate_label(candidate)} ===")
    model = make_model(model_name, imbalance_method, **params).fit(X_first, y_first)
    scores, probabilities = ranking(model, X_second), probability(model, X_second)

    summary = summarize(y_second, scores, amounts=amounts, threshold_scores=probabilities)
    for key, value in summary.items():
        print(f"  {key:22} {value:.4f}")
    return scores, probabilities

def _tuned_candidates(X, y, model_names=DEFAULT_TUNED, retune: bool = False) -> list[tuple]:
    """Optuna searches on day 1 only, cached under reports/ so reruns stay cheap.

    `revalidate=False` because the leaderboard below re-scores every candidate on the
    full repeated CV anyway; re-scoring inside the search too would pay for 15 extra
    fits per model and report the same number twice.
    """
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

        candidates.append((result.model, result.imbalance, result.params))
    return candidates

def main(retune: bool = False) -> None:
    pd.set_option("display.width", 200)

    df = load_data("dataset/creditcard.csv")
    split = temporal_split(df["Time"])
    X, y = extract_features(df)

    X_first, y_first = X.iloc[split.train], y.iloc[split.train]
    X_second, y_second = X.iloc[split.test], y.iloc[split.test]
    print(f"day 1: {len(X_first):,} rows, {int(y_first.sum())} frauds")
    print(f"day 2: {len(X_second):,} rows, {int(y_second.sum())} frauds")

    cv = cross_validation_split()

    print("\n=== hyperparameter search (day 1 only) ===")
    tuned = _tuned_candidates(X_first, y_first, retune=retune)
    candidates = list(CANDIDATES) + tuned
    by_label = {candidate_label(candidate): candidate for candidate in candidates}

    print("\n=== model selection (day 1 only) ===")
    rows = []
    for candidate in candidates:
        row = cross_validate_candidates(X_first, y_first, [candidate], cv=cv)
        print(f"  {row.at[0, 'label']:36} "
              f"pr_auc={row.at[0, 'pr_auc_mean']:.4f} ± {row.at[0, 'pr_auc_std']:.4f} "
              f"({row.at[0, 'fit_seconds']:.1f}s/fit)")
        rows.append(row)

    leaderboard = pd.concat(rows, ignore_index=True).sort_values(
        "pr_auc_mean", ascending=False, ignore_index=True
    )
    print()
    print(leaderboard.to_string())

    amounts = df["Amount"].iloc[split.test]
    best = leaderboard.iloc[0]
    scores, probabilities = _holdout(
        by_label[best["label"]], X_first, y_first, X_second, y_second, amounts
    )

    print(f"\n  CV pr_auc {best['pr_auc_mean']:.4f} -> holdout, gap is the finding")
    units = "probability" if probabilities is not None else "decision function"
    print(f"\n=== operating points (day 2, thresholds in {units} units) ===")
    table_scores = scores if probabilities is None else probabilities
    print(threshold_table(y_second, table_scores, amounts, n_rows=12).to_string())
