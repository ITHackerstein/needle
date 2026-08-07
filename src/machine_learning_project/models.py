from .data import SEED
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import numpy as np

class NegativeOnlyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, detector=None):
        self.detector = detector

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.array([0, 1], dtype=np.uint8)
        self.detector_ = clone(self.detector).fit(X[y == 0])
        return self

    def decision_function(self, X):
        return -self.detector_.score_samples(X)  # higher = more anomalous

    def predict(self, X):
        return (self.detector_.predict(X) == -1).astype(np.uint8)

def _preprocessing():
    return ColumnTransformer(
        [("amount", RobustScaler(), ["log_amount"])],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

MODELS = {
    "dummy": lambda weighted, **kwargs: DummyClassifier(
        strategy="stratified",
        random_state=SEED,
        **kwargs
    ),
    "logistic_regression": lambda weighted, **kwargs: LogisticRegression(
        max_iter=2000,
        random_state=SEED,
        class_weight="balanced" if weighted else None,
        **kwargs
    ),
    "random_forest": lambda weighted, **kwargs: RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample" if weighted else None,
        random_state=SEED,
        n_jobs=-1,
        **kwargs
    ),
    "lgbm": lambda weighted, **kwargs: LGBMClassifier(
        is_unbalance=weighted,
        random_state=SEED,
        verbose=-1,
        **kwargs
    ),
    "isolation_forest": lambda weighted, **kwargs: NegativeOnlyDetector(
        detector=IsolationForest(
            random_state=SEED,
            n_jobs=-1,
            **kwargs
        )
    )
}

SAMPLERS = {
    "under": lambda: RandomUnderSampler(random_state=SEED),
    "smote": lambda: SMOTE(random_state=SEED),
    "weighted": lambda: None,
    "none": lambda: None
}

CANDIDATES = (
    [("dummy", "none")]
    + [(model_name, imbalance_method)
       for model_name in ("logistic_regression", "random_forest", "lgbm")
       for imbalance_method in ("weighted", "under", "smote")]
    + [("isolation_forest", "none")]
)

def make_model(model_name, imbalance_method: str = "weighted", **kwargs):
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' is not supported.")

    steps = [("preprocessing", _preprocessing())]
    if sampler := SAMPLERS[imbalance_method]():
        steps.append(("sampler", sampler))
    steps.append(("classifier", MODELS[model_name](weighted=(imbalance_method == "weighted"), **kwargs)))
    return Pipeline(steps)
