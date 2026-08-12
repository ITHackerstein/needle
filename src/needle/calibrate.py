import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from .models import CandidatePipeline

def calibrated_candidate(
    candidate: CandidatePipeline,
    method: str = "sigmoid",
    cv=None
) -> CalibratedClassifierCV:
    calibrated_model = CalibratedClassifierCV(
        candidate.build(),
        method=method,
        cv=cv
    )
    return calibrated_model


def calibration_report(y_true, y_prob, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)

    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, y_prob), 1, len(edges) - 1) - 1

    table = pd.DataFrame({ "bin": bins, "score": y_prob, "label": y_true }).groupby("bin").agg(
        mean_score=("score", "mean"),
        fraction_positive=("label", "mean"),
        count=("label", "count")
    ).reset_index(drop=True)

    return table, float(brier_score_loss(y_true, y_prob))

