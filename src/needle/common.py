import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 42

def recall_key(min_precision: float) -> str:
    # NOTE: the metric's name follows its threshold, so it has one spelling everywhere
    return f"recall_at_p{int(min_precision * 100)}"


def take(data, index):
    return data.iloc[index] if hasattr(data, "iloc") else np.asarray(data)[index]


def ranking(model, X) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict_proba(X)[:, 1]


def probability(model, X) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    return model.predict_proba(X)[:, 1]


def cross_validation_split(n_splits: int = 5, n_repeats: int = 3, seed: int = SEED) -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
