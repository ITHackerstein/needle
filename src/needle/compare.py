import numpy as np
import pandas as pd
from scipy import stats

from .common import CV_SPLITS
from .config import ALPHA, N_COMPARE

COLUMNS = ["label", "difference", "t", "p_value", "p_holm", "significant", "p_naive"]


def train_test_ratio(n_splits: int = CV_SPLITS) -> float:
    # NOTE: the correction needs |test| / |train|, and k-fold holds one fold out against the
    # k-1 it fits on, so the ratio is 1/(k-1) whatever the row count happens to be
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}")
    return 1.0 / (n_splits - 1)


def variance_inflation(n_folds: int, n_splits: int = CV_SPLITS) -> float:
    # NOTE: how much wider the corrected denominator is than the naive one, for the report
    # to quote rather than assert
    return 1.0 + n_folds * train_test_ratio(n_splits)


def corrected_paired_t(a, b, ratio: float | None = None) -> tuple[float, float]:
    # NOTE: Nadeau & Bengio's corrected resampled t-test. A plain paired t divides the variance
    # of the differences by n, which is the assumption that the n folds are n independent
    # measurements. Repeated k-fold is nothing of the kind: any two training sets share
    # (k-2)/(k-1) of their rows, and every repeat reuses all of them. So the naive denominator
    # is far too small and the test calls fold noise significant. The correction replaces 1/n
    # with 1/n + |test|/|train|, which is the variant Bouckaert & Frank recommend for repeated
    # k-fold specifically. Pass ratio=0 to get the naive statistic back.
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"paired scores must line up fold for fold, got {a.shape} and {b.shape}")

    differences = a - b
    n = differences.size
    if n < 2:
        raise ValueError(f"need at least two paired folds, got {n}")

    ratio = train_test_ratio() if ratio is None else ratio
    variance = float(differences.var(ddof=1))
    if variance <= 0.0:
        # NOTE: two candidates that scored identically on every fold. The statistic is 0/0, and
        # the conservative reading - no evidence of a difference - beats reporting certainty
        # from a degenerate denominator.
        return 0.0, 1.0

    t = float(differences.mean()) / np.sqrt(variance * (1.0 / n + ratio))
    return float(t), float(2.0 * stats.t.sf(abs(t), df=n - 1))


def holm(p_values) -> np.ndarray:
    # NOTE: Holm-Bonferroni, step-down. Testing the winner against m challengers on one set of
    # folds is m chances at a gap that is not there, so the smallest p is multiplied by m, the
    # next by m-1, and so on. The running maximum keeps the output monotone in the input.
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return p_values

    order = np.argsort(p_values, kind="stable")
    scaled = p_values[order] * np.arange(p_values.size, 0, -1)
    adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)

    result = np.empty_like(p_values)
    result[order] = adjusted
    return result


def against_winner(
    fold_scores: dict,
    order,
    n_compare: int = N_COMPARE,
    n_splits: int = CV_SPLITS,
    alpha: float = ALPHA
) -> pd.DataFrame:
    # NOTE: the family is fixed before any p-value is read - the winner against the challengers
    # the report already prints, and nothing else. All 120 pairs of a 16-candidate leaderboard
    # would turn up a 'significant' gap somewhere by construction.
    order = [label for label in order if label in fold_scores]
    if len(order) < 2:
        return pd.DataFrame(columns=COLUMNS)

    winner, challengers = order[0], order[1:max(n_compare, 2)]
    ratio = train_test_ratio(n_splits)

    rows = []
    for label in challengers:
        a, b = np.asarray(fold_scores[winner], dtype=float), np.asarray(fold_scores[label], dtype=float)
        t, p_value = corrected_paired_t(a, b, ratio)
        rows.append({
            "label": label,
            "difference": float(a.mean() - b.mean()),
            "t": t,
            "p_value": p_value,
            "p_naive": corrected_paired_t(a, b, 0.0)[1]
        })

    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame["p_value"])
    frame["significant"] = frame["p_holm"] < alpha
    return frame[COLUMNS]
