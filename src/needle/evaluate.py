import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_recall_curve
from .models import CandidatePipeline, CANDIDATE_PIPELINES
from .data import cross_validation_split

MIN_PRECISION = 0.9


def _take(data, index):
    return data.iloc[index] if hasattr(data, "iloc") else np.asarray(data)[index]


def ranking(model, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def probability(model, X) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    return model.predict_proba(X)[:, 1]


def recall_at_precision(y_true, y_score, min_precision: float = MIN_PRECISION) -> float:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    at_precision = recall[precision >= min_precision]
    return float(at_precision.max()) if at_precision.size > 0 else 0.0


SCORING = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    f"recall_at_p{int(MIN_PRECISION * 100)}": make_scorer(recall_at_precision, response_method=("decision_function", "predict_proba"))
}


def cross_validate_candidates(
    X, y,
    candidates: tuple[CandidatePipeline, ...] = CANDIDATE_PIPELINES,
    cv=None,
    n_jobs: int = 1
) -> pd.DataFrame:
    cv = cv if cv is not None else cross_validation_split()

    rows = []
    for candidate in candidates:
        result = cross_validate(
            candidate.build(),
            X, y,
            cv=cv,
            scoring=SCORING,
            n_jobs=n_jobs
        )

        row = {
            "label": candidate.label(),
            "model": candidate.model_name,
            "imbalance_method": candidate.imbalance_method,
            "tuned": bool(candidate.params)
        }

        for metric in SCORING:
            row[f"{metric}_mean"] = result[f"test_{metric}"].mean()
            row[f"{metric}_std"] = result[f"test_{metric}"].std()
        row["fit_seconds"] = result["fit_time"].mean()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False, ignore_index=True)
