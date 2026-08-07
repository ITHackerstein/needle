import numpy as np
import pandas as pd
from machine_learning_project.data import load_data, temporal_split, extract_features, cross_validation_split
from machine_learning_project.models import CANDIDATES, make_model
from machine_learning_project.evaluate import cross_validate_candidates, response, summarize, threshold_table

def _holdout(model_name, imbalance_method, X_first, y_first, X_second, y_second, amounts) -> np.ndarray:
    print(f"\n=== temporal holdout: {model_name} / {imbalance_method} ===")
    model = make_model(model_name, imbalance_method).fit(X_first, y_first)
    scores = response(model, X_second)

    for key, value in summarize(y_second, scores, amounts=amounts).items():
        print(f"  {key:22} {value:.4f}")
    return scores

def main() -> None:
    pd.set_option("display.width", 200)

    df = load_data("dataset/creditcard.csv")
    split = temporal_split(df["Time"])
    X, y = extract_features(df)

    X_first, y_first = X.iloc[split.train], y.iloc[split.train]
    X_second, y_second = X.iloc[split.test], y.iloc[split.test]
    print(f"day 1: {len(X_first):,} rows, {int(y_first.sum())} frauds")
    print(f"day 2: {len(X_second):,} rows, {int(y_second.sum())} frauds")

    cv = cross_validation_split()

    print("\n=== model selection (day 1 only) ===")
    rows = []
    for candidate in CANDIDATES:
        row = cross_validate_candidates(X_first, y_first, [candidate], cv=cv)
        print(f"  {row.at[0, 'model']:20} {row.at[0, 'imbalance']:8} "
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
    scores = _holdout(best["model"], best["imbalance"], X_first, y_first, X_second, y_second, amounts)

    print(f"\n  CV pr_auc {best['pr_auc_mean']:.4f} -> holdout, gap is the finding")
    print("\n=== operating points (day 2) ===")
    print(threshold_table(y_second, scores, amounts, n_rows=12).to_string())

    # The autoencoder is unlikely to top a leaderboard it shares with supervised
    # models, but it is the only candidate fit without labels, so what it catches
    # on day 2 is worth seeing even when it loses.
    if best["model"] != "autoencoder":
        autoencoder_scores = _holdout(
            "autoencoder", "none", X_first, y_first, X_second, y_second, amounts
        )

        budget = 100
        frauds = set(np.flatnonzero(y_second.to_numpy() == 1))
        top = lambda s: set(np.argsort(s)[::-1][:budget]) & frauds
        caught, caught_autoencoder = top(scores), top(autoencoder_scores)

        print(f"\n=== do they catch the same frauds? (top {budget} alerts each) ===")
        print(f"  {best['model']:20} {len(caught)}")
        print(f"  {'autoencoder':20} {len(caught_autoencoder)}")
        print(f"  {'both':20} {len(caught & caught_autoencoder)}")
        print(f"  {'autoencoder only':20} {len(caught_autoencoder - caught)}"
              f"  <- the case for keeping an unsupervised detector")
