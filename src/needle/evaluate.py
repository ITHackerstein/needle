import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_recall_curve
from .models import CandidatePipeline, CANDIDATE_PIPELINES
from .common import cross_validation_split, recall_key
from .config import MIN_PRECISION

FOLD_COLUMN = "pr_auc_folds"


def recall_at_precision(y_true, y_score, min_precision: float = MIN_PRECISION) -> float:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    at_precision = recall[precision >= min_precision]
    return float(at_precision.max()) if at_precision.size > 0 else 0.0


def scoring(min_precision: float = MIN_PRECISION) -> dict:
    return {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        recall_key(min_precision): make_scorer(
            recall_at_precision,
            response_method=("decision_function", "predict_proba"),
            min_precision=min_precision
        )
    }


def cross_validate_candidates(
    X, y,
    candidates: tuple[CandidatePipeline, ...] = CANDIDATE_PIPELINES,
    cv=None,
    n_jobs: int = 1,
    min_precision: float = MIN_PRECISION
) -> pd.DataFrame:
    cv = cv if cv is not None else cross_validation_split()
    metrics = scoring(min_precision)

    rows = []
    for candidate in candidates:
        result = cross_validate(
            candidate.build(),
            X, y,
            cv=cv,
            scoring=metrics,
            n_jobs=n_jobs
        )

        row = {
            "label": candidate.label(),
            "model": candidate.model_name,
            "imbalance_method": candidate.imbalance_method,
            "tuned": bool(candidate.params)
        }

        for metric in metrics:
            row[f"{metric}_mean"] = result[f"test_{metric}"].mean()
            row[f"{metric}_std"] = result[f"test_{metric}"].std()
        row["fit_seconds"] = result["fit_time"].mean()
        # NOTE: the per-fold PR-AUCs survive, not just their mean and spread. Every candidate
        # here is scored on the same cv object, so these vectors pair up fold for fold and are
        # what compare.py runs its paired test on.
        row[FOLD_COLUMN] = result["test_pr_auc"].copy()

        rows.append(row)

    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False, ignore_index=True)
