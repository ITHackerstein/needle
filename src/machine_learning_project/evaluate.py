import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, make_scorer, precision_recall_curve, roc_auc_score
from sklearn.model_selection import cross_validate
from .data import cross_validation_split
from .models import CANDIDATES, make_model

MIN_PRECISION = 0.9
REVIEW_COST = 3.0  # what one analyst investigation costs, same units as Amount

def response(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)

def recall_at_precision(y_true, y_score, min_precision: float = MIN_PRECISION) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    achievable = recall[precision >= min_precision]
    return float(achievable.max()) if achievable.size else 0.0

def at_alert_budget(y_true, y_score, n_alerts: int) -> dict:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if n_alerts < 1:
        raise ValueError(f"Alert budget must be at least 1, got {n_alerts}.")

    threshold = np.sort(y_score)[::-1][:n_alerts].min()
    flagged = y_score >= threshold

    caught = int((flagged & (y_true == 1)).sum())
    return {
        "alerts": int(flagged.sum()),
        "precision": caught / max(flagged.sum(), 1),
        "recall": caught / max((y_true == 1).sum(), 1),
        "threshold": float(threshold)
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
    _, _, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = thresholds[np.linspace(0, len(thresholds) - 1, n_rows).astype(int)]

    rows = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(np.uint8)
        caught = int(((y_pred == 1) & (y_true == 1)).sum())
        rows.append({
            "threshold": float(threshold),
            "alerts_per_day": y_pred.sum() / days,
            "precision": caught / max(y_pred.sum(), 1),
            "recall": caught / max((y_true == 1).sum(), 1),
            "cost": cost(y_true, y_pred, amounts, review_cost)
        })
    return pd.DataFrame(rows)

def summarize(
    y_true,
    y_score,
    amounts=None,
    days: float = 1.0,
    alert_budget: int = 100, min_precision: float = MIN_PRECISION
) -> dict:
    summary = {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),  # secondary, for comparability only
        f"recall_at_p{int(min_precision * 100)}": recall_at_precision(y_true, y_score, min_precision)
    }
    budget = at_alert_budget(y_true, y_score, int(alert_budget * days))
    for key, value in budget.items():
        summary[f"budget_{key}"] = value

    if amounts is not None:
        table = threshold_table(y_true, y_score, amounts, days=days, n_rows=200)
        best = table.loc[table["cost"].idxmin()]
        summary["min_cost"] = best["cost"]
        summary["min_cost_threshold"] = best["threshold"]
    return summary

SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    f"recall_at_p{int(MIN_PRECISION * 100)}": make_scorer(
        recall_at_precision, response_method=("predict_proba", "decision_function")
    )
}

def cross_validate_candidates(X, y, candidates=CANDIDATES, cv=None, n_jobs: int = 1) -> pd.DataFrame:
    cv = cv if cv is not None else cross_validation_split()

    rows = []
    for model_name, imbalance_method in candidates:
        result = cross_validate(
            make_model(model_name, imbalance_method),
            X, y,
            cv=cv,
            scoring=SCORING,
            n_jobs=n_jobs
        )
        row = {"model": model_name, "imbalance": imbalance_method}
        for metric in SCORING:
            row[f"{metric}_mean"] = result[f"test_{metric}"].mean()
            row[f"{metric}_std"] = result[f"test_{metric}"].std()
        row["fit_seconds"] = result["fit_time"].mean()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False, ignore_index=True)
