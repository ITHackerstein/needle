import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from .models import CandidatePipeline
from .common import SEED, probability, ranking, take
from .config import ALERT_BUDGET, MIN_PRECISION, OBJECTIVES, REVIEW_COST

def sweep_thresholds(
    y_true,
    y_score,
    transaction_amounts=None,
    days: float = 1.0,
    review_cost: float = REVIEW_COST
) -> pd.DataFrame:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)

    order = np.argsort(-y_score, stable=True) # NOTE: stable keeps the order of ties intact
    scores, labels = y_score[order], y_true[order]

    # NOTE: cuts contains the index of each tied score block
    cuts = np.flatnonzero(np.concatenate([np.diff(scores) != 0, [True]]))
    # NOTE: alerts is the number of transactions that would be reviewed at each threshold
    alerts = cuts + 1
    # NOTE: caught is the number of fraudulent transactions that would be caught at each threshold
    caught = np.cumsum(labels)[cuts]
    # NOTE: positives is the total number of fraudulent transactions in the dataset
    positives = max(int(labels.sum()), 1)

    result = pd.DataFrame({
        "threshold": scores[cuts],
        "alerts": alerts,
        "alerts_per_day": alerts / days,
        "alert_rate": alerts / scores.size,
        "precision": caught / alerts,
        "recall": caught / positives
    })

    if transaction_amounts is not None:
        transaction_amounts = np.asarray(transaction_amounts, dtype=np.float64)
        # NOTE: fraudulent_cost is non zero only for fraudulent transactions, and is the amount of the transaction
        fraudulent_cost = transaction_amounts[order] * labels
        # NOTE: missed_cost is the amount of money lost due to fraudulent transactions that were not caught
        missed_cost = fraudulent_cost.sum() - np.cumsum(fraudulent_cost)[cuts]
        # NOTE: false_positive_cost is the cost of reviewing transactions that were not fraudulent
        false_positive_cost = review_cost * (alerts - caught)
        result["cost"] = missed_cost + false_positive_cost

    return result


def threshold_table(
    y_true,
    y_score,
    transaction_amounts,
    days: float = 1.0,
    n_rows: int = 15,
    review_cost: float = REVIEW_COST
) -> pd.DataFrame:
    sweep_result = sweep_thresholds(y_true, y_score, transaction_amounts, days=days, review_cost=review_cost)
    targets = np.geomspace(1, int(sweep_result["alerts"].iat[-1]), n_rows, dtype=int)
    picks = np.clip(np.unique(np.searchsorted(sweep_result["alerts"].to_numpy(), targets)), 0, len(sweep_result) - 1)
    return sweep_result.iloc[picks].reset_index(drop=True)

def select_threshold(
    y_true,
    y_score,
    transaction_amounts=None,
    objective: str = "cost",
    days: float = 1.0,
    alert_budget: int = ALERT_BUDGET,
    min_precision: float = MIN_PRECISION,
    review_cost: float = REVIEW_COST
) -> dict:
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}, got {objective}")

    sweep_result = sweep_thresholds(y_true, y_score, transaction_amounts, days=days, review_cost=review_cost)

    if objective == "cost":
        if transaction_amounts is None:
            raise ValueError("transaction_amounts must be provided for cost objective")

        row, feasible = sweep_result.loc[sweep_result["cost"].idxmin()], True
    elif objective == "precision":
        meets_precision = sweep_result[sweep_result["precision"] >= min_precision]
        # NOTE: If meets_precision is empty, it means that no threshold meets the minimum precision requirement, so we take the row with the highest precision and mark it as infeasible
        row, feasible = (
            (meets_precision.loc[meets_precision["recall"].idxmax()], True) if len(meets_precision) > 0
            else (sweep_result.loc[sweep_result["precision"].idxmax()], False)
        )
    else:
        in_budget = sweep_result[sweep_result["alerts_per_day"] <= alert_budget]
        # NOTE: If in_budget is empty, it means that all thresholds exceed the alert budget, so we take the first row (the one with the lowest threshold) and mark it as infeasible
        row, feasible = (in_budget.iloc[-1], True) if len(in_budget) > 0 else (sweep_result.iloc[0], False)

    return {
        "objective": objective,
        "feasible": feasible,
        "threshold": float(row["threshold"]),
        "alert_rate": float(row["alert_rate"]),
        "alerts_per_day": float(row["alerts_per_day"]),
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "cost": float(row["cost"]) if "cost" in row else float("nan")
    }


def threshold_from_rate(y_score, alert_rate: float) -> float:
    y_score = np.asarray(y_score)
    if not (0.0 <= alert_rate <= 1.0):
        raise ValueError(f"alert_rate must be between 0 and 1, got {alert_rate}")

    index = max(1, min(y_score.size, int(round(alert_rate * y_score.size))))
    return float(np.sort(y_score, descending=True)[index - 1])


def apply_threshold(
    y_true,
    y_score,
    threshold: float,
    transaction_amounts=None,
    days: float = 1.0,
    review_cost: float = REVIEW_COST
) -> dict:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    alerts = int(y_pred.sum())
    caught = int(((y_pred == 1) & (y_true == 1)).sum())

    cost = float("nan")
    if transaction_amounts is not None:
        missed_cost = transaction_amounts[y_true == 1].sum() - transaction_amounts[(y_pred == 1) & (y_true == 1)].sum()
        false_positive_cost = review_cost * (alerts - caught)
        cost = missed_cost + false_positive_cost

    return {
        "threshold": threshold,
        "alert_rate": alerts / y_score.size,
        "alerts_per_day": alerts / days,
        "precision": caught / max(alerts, 1),
        "recall": caught / max(int(y_true.sum()), 1),
        "cost": cost
    }


def selection_scores(
    candidate: CandidatePipeline,
    X, y,
    cv=None,
    seed: int = SEED
) -> tuple[np.ndarray, np.ndarray | None]:
    cv = cv if cv is not None else StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    scores = np.empty(len(y), dtype=np.float64)
    probabilities = np.full(len(y), np.nan)
    for train, test in cv.split(X, y):
        model = candidate.build()
        model.fit(take(X, train), take(y, train))

        X_test = take(X, test)
        test_ranking = ranking(model, X_test)
        test_probability = probability(model, X_test)
        scores[test] = test_ranking
        if test_probability is not None:
            probabilities[test] = test_probability

    return scores, None if np.isnan(probabilities).any() else probabilities


def review_cost_sensitivity(
    y_true,
    y_score,
    transaction_amounts,
    days: float = 1.0,
    review_costs: tuple[float, ...] = (1.0, 3.0, 5.0, 10.0, 30.0, 100.0)
) -> pd.DataFrame:
    results = []
    for review_cost in review_costs:
        sweep_result = select_threshold(y_true, y_score, transaction_amounts, objective="cost", days=days, review_cost=review_cost)
        results.append({ "review_cost": review_cost, **sweep_result })
    return pd.DataFrame(results).drop(columns=["objective", "feasible"])
