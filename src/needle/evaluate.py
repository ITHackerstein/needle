import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, make_scorer, precision_recall_curve, roc_auc_score
from sklearn.model_selection import cross_validate
from .data import cross_validation_split
from .models import CANDIDATES, make_model

MIN_PRECISION = 0.9
REVIEW_COST = 3.0  # what one analyst investigation costs, same units as Amount

def ranking(model, X) -> np.ndarray:
    """Scores for rank-based metrics: PR-AUC, ROC-AUC, recall@precision.

    decision_function first, which is the order sklearn's own scorers use. A
    class-weighted LightGBM emits margins in the millions, and squashing those
    through a sigmoid collapses them to a handful of distinct float64 values -
    on day 2, `is_unbalance=True` gives 500 distinct probabilities for 139,490
    rows, with 15,276 tied at exactly 1.0. That destroys the ranking the margins
    hold (PR-AUC 0.0095 from probabilities, 0.62 from margins). Rank metrics do
    not need a probability, so they should not ask for one.
    """
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]

def probability(model, X) -> np.ndarray | None:
    """Probabilities for choosing an operating point, or None for score-only models.

    Precision, recall and cost are invariant under any monotone rescaling of the
    score, so this changes no metric - only the units of the threshold that gets
    reported, and a probability is the one a fraud team can reason about. The
    unsupervised detectors have no predict_proba, hence the None.
    """
    if not hasattr(model, "predict_proba"):
        return None
    return model.predict_proba(X)[:, 1]

def recall_at_precision(y_true, y_score, min_precision: float = MIN_PRECISION) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    achievable = recall[precision >= min_precision]
    return float(achievable.max()) if achievable.size else 0.0

def at_alert_budget(y_true, y_score, n_alerts: int) -> dict:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if n_alerts < 1:
        raise ValueError(f"Alert budget must be at least 1, got {n_alerts}.")

    alerts = min(n_alerts, y_score.size)
    threshold = np.sort(y_score)[::-1][:alerts].min()

    # A saturated classifier can tie thousands of rows at the threshold, so
    # `>= threshold` would blow the budget wide open. Reviewing exactly `alerts`
    # rows means taking everything strictly above plus an arbitrary slice of the
    # tied block, so count the tied block's frauds pro rata: the expected catch
    # over that arbitrary choice, rather than crediting all of them.
    above, tied = y_score > threshold, y_score == threshold
    take = min(alerts - int(above.sum()), int(tied.sum()))

    caught = float((above & (y_true == 1)).sum())
    if take > 0:
        caught += int((tied & (y_true == 1)).sum()) * take / int(tied.sum())

    return {
        "alerts": alerts,
        "precision": caught / alerts,
        "recall": caught / max((y_true == 1).sum(), 1),
        "threshold": float(threshold),
        "tied_at_threshold": int(tied.sum())
    }

def cost(y_true, y_pred, amounts, review_cost: float = REVIEW_COST) -> float:
    y_true, y_pred, amounts = np.asarray(y_true), np.asarray(y_pred), np.asarray(amounts)
    missed = amounts[(y_true == 1) & (y_pred == 0)].sum()
    to_review = ((y_true == 0) & (y_pred == 1)).sum()
    return float(missed + review_cost * to_review)

def threshold_table(
    y_true,
    y_score,
    amounts,
    days: float = 1.0,
    n_rows: int = 15,
    review_cost: float = REVIEW_COST
) -> pd.DataFrame:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    # Rows are spaced geometrically in alert count, not uniformly over the threshold
    # array. Uniform sampling there is uniform in *rank*, so the k-th row alerts on
    # roughly n - k rows: on day 2 that walked 139,490 -> 12,688 alerts and then
    # jumped to 1, leaving the entire deployable band (single digits to a few hundred
    # alerts) unsampled. Geometric spacing puts most of the resolution where an
    # alert budget actually sits while still reaching the aggressive end, which is
    # where the cost optimum lives when a missed fraud costs ~40x a review.
    descending = np.sort(y_score)[::-1]
    counts = np.unique(np.geomspace(1, y_score.size, n_rows).round().astype(int))

    rows = []
    for count in counts:
        threshold = float(descending[count - 1])
        y_pred = (y_score >= threshold).astype(np.uint8)
        alerts = int(y_pred.sum())
        caught = int(((y_pred == 1) & (y_true == 1)).sum())
        rows.append({
            "threshold": threshold,
            "alerts_per_day": alerts / days,
            "precision": caught / max(alerts, 1),
            "recall": caught / max((y_true == 1).sum(), 1),
            "cost": cost(y_true, y_pred, amounts, review_cost)
        })

    # Ties collapse distinct requested counts onto one threshold - a saturated model
    # can put thousands of rows on the same score, so `>=` overshoots the budget.
    return pd.DataFrame(rows).drop_duplicates(subset="threshold", ignore_index=True)

def summarize(
    y_true,
    y_score,
    amounts=None,
    days: float = 1.0,
    alert_budget: int = 100,
    min_precision: float = MIN_PRECISION,
    threshold_scores=None
) -> dict:
    """`y_score` drives the rank metrics; `threshold_scores` drives the budget and
    cost rows, defaulting to `y_score`. Passing probabilities for the second reports
    a cut point in units a fraud team can read, while PR-AUC still reads the
    un-squashed ranking - see `ranking` for why those can differ so violently.
    """
    threshold_scores = y_score if threshold_scores is None else threshold_scores

    summary = {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),  # secondary, for comparability only
        f"recall_at_p{int(min_precision * 100)}": recall_at_precision(y_true, y_score, min_precision)
    }
    budget = at_alert_budget(y_true, threshold_scores, int(alert_budget * days))
    for key, value in budget.items():
        summary[f"budget_{key}"] = value

    if amounts is not None:
        table = threshold_table(y_true, threshold_scores, amounts, days=days, n_rows=200)
        best = table.loc[table["cost"].idxmin()]
        summary["min_cost"] = best["cost"]
        summary["min_cost_threshold"] = best["threshold"]
        summary["min_cost_alerts"] = best["alerts_per_day"]
    return summary

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    f"recall_at_p{int(MIN_PRECISION * 100)}": make_scorer(
        # decision_function first, to match the built-in scorers above and `ranking`.
        # With predict_proba first this column read 0.0000 +/- 0.0000 for every
        # class-weighted model while pr_auc in the same row read 0.67 - the two were
        # scoring different model outputs and were never comparable.
        recall_at_precision, response_method=("decision_function", "predict_proba")
    )
}

def unpack_candidate(candidate) -> tuple[str, str, dict]:
    """A candidate is ("model", "imbalance") or ("model", "imbalance", {params})."""
    model_name, imbalance_method, *rest = candidate
    return model_name, imbalance_method, dict(rest[0]) if rest else {}

def candidate_label(candidate) -> str:
    # The suffix is what keeps a tuned entry distinguishable from the stock one it
    # shares a (model, imbalance) pair with, so labels stay unique keys.
    model_name, imbalance_method, params = unpack_candidate(candidate)
    return f"{model_name}/{imbalance_method}{' (tuned)' if params else ''}"

def cross_validate_candidates(X, y, candidates=CANDIDATES, cv=None, n_jobs: int = 1) -> pd.DataFrame:
    cv = cv if cv is not None else cross_validation_split()

    rows = []
    for candidate in candidates:
        model_name, imbalance_method, params = unpack_candidate(candidate)
        result = cross_validate(
            make_model(model_name, imbalance_method, **params),
            X, y,
            cv=cv,
            scoring=SCORING,
            n_jobs=n_jobs
        )
        row = {
            "label": candidate_label(candidate),
            "model": model_name,
            "imbalance": imbalance_method,
            "tuned": bool(params)
        }
        for metric in SCORING:
            row[f"{metric}_mean"] = result[f"test_{metric}"].mean()
            row[f"{metric}_std"] = result[f"test_{metric}"].std()
        row["fit_seconds"] = result["fit_time"].mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False, ignore_index=True)
