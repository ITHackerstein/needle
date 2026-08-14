import numpy as np
import pandas as pd
from dataclasses import dataclass
from .common import SEED
from .config import SHAP_FEATURES, SHAP_SAMPLE


def _correlation(values: np.ndarray, other: np.ndarray) -> float:
    if values.std() == 0.0 or other.std() == 0.0:
        return 0.0
    return float(np.corrcoef(values, other)[0, 1])


@dataclass(frozen=True)
class Explanation:
    values: np.ndarray
    features: pd.DataFrame
    base_value: float
    explainer: str
    skipped: str = ""

    def ranking(self, n_features: int | None = None) -> pd.DataFrame:
        magnitude = np.abs(self.values).mean(axis=0)
        table = pd.DataFrame({
            "feature": self.features.columns,
            "mean_abs_shap": magnitude,
            "share": magnitude / max(magnitude.sum(), 1e-12),
            "value_correlation": [
                _correlation(
                    self.features.iloc[:, column].to_numpy(dtype=np.float64),
                    self.values[:, column]
                )
                for column in range(self.values.shape[1])
            ]
        }).sort_values("mean_abs_shap", ascending=False, ignore_index=True)

        table.insert(0, "rank", np.arange(1, len(table) + 1))
        return table if n_features is None else table.head(n_features)


@dataclass(frozen=True)
class Missed:
    confusion: pd.DataFrame
    by_outcome: pd.DataFrame
    worst: pd.DataFrame
    missed_amount_share: float = 0.0


def explanation_sample(y, n_negatives: int = SHAP_SAMPLE, seed: int = SEED) -> np.ndarray:
    y = np.asarray(y)
    negatives = np.flatnonzero(y == 0)
    if negatives.size > n_negatives:
        negatives = np.random.default_rng(seed).choice(negatives, n_negatives, replace=False)
    return np.sort(np.concatenate([np.flatnonzero(y == 1), negatives]))


def _positive_class(values, base_value) -> tuple[np.ndarray, float]:
    values = np.asarray(values)
    base = np.asarray(base_value, dtype=np.float64)

    if values.ndim == 3:
        values = values[..., -1]
        base = base[..., -1] if base.ndim else base
    return values, float(np.mean(base))


def explain(
    model,
    X,
    n_features: int | None = None,
    verbose: bool = True
) -> Explanation | None:
    import shap

    preprocessing, classifier = model.named_steps["preprocessing"], model.named_steps["classifier"]
    features = pd.DataFrame(
        preprocessing.transform(X),
        columns=preprocessing.get_feature_names_out(),
        index=getattr(X, "index", None)
    )

    # An unsupervised detector wraps its estimator and negates its score, so the
    # contributions have to be negated with it to keep "positive means fraud".
    estimator = getattr(classifier, "detector_", classifier)
    orientation = -1.0 if estimator is not classifier else 1.0

    reasons = []
    for label, build in (
        ("tree", lambda: shap.TreeExplainer(estimator)),
        ("linear", lambda: shap.LinearExplainer(estimator, features.to_numpy()))
    ):
        try:
            result = build()(features.to_numpy())
        except Exception as error:  # shap raises whatever the backend raises
            reasons.append(f"{label} ({type(error).__name__})")
            continue

        values, base_value = _positive_class(result.values, result.base_values)
        explanation = Explanation(
            values=orientation * values,
            features=features,
            base_value=orientation * base_value,
            explainer=label
        )
        if verbose:
            top = explanation.ranking(n_features or SHAP_FEATURES)
            print(f"  {label} explainer on {len(features):,} rows, "
                  f"base value {explanation.base_value:.4f}")
            print(top.to_string(index=False))
        return explanation

    if verbose:
        print(f"  no SHAP explainer for {type(estimator).__name__}: tried {', '.join(reasons)}")
    return None


def confusion(y_true, y_score, threshold: float) -> pd.DataFrame:
    y_true, alerted = np.asarray(y_true), np.asarray(y_score) >= threshold

    table = pd.DataFrame(
        [
            [int(((y_true == label) & ~alerted).sum()), int(((y_true == label) & alerted).sum())]
            for label in (0, 1)
        ],
        index=["legit", "fraud"],
        columns=["cleared", "alerted"]
    )
    table.index.name = "actual"
    return table


def missed_frauds(
    y_true,
    y_score,
    threshold: float,
    transaction_amounts,
    n_rows: int = 10
) -> Missed:
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    amounts = np.asarray(transaction_amounts, dtype=np.float64)
    labels = (
        transaction_amounts.index.to_numpy() if hasattr(transaction_amounts, "index")
        else np.arange(y_true.size)
    )

    frauds = np.flatnonzero(y_true == 1)
    caught = y_score[frauds] >= threshold
    queue = np.empty(y_score.size, dtype=np.int64)
    queue[np.argsort(y_score, stable=True, descending=True)] = np.arange(1, y_score.size + 1)

    detail = pd.DataFrame({
        "row": labels[frauds],
        "amount": amounts[frauds],
        "score": y_score[frauds],
        "queue_rank": queue[frauds],
        "outcome": np.where(caught, "caught", "missed")
    })

    by_outcome = detail.groupby("outcome").agg(
        frauds=("amount", "size"),
        total_amount=("amount", "sum"),
        mean_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        max_amount=("amount", "max")
    )
    by_outcome["amount_share"] = by_outcome["total_amount"] / max(amounts[frauds].sum(), 1e-12)
    by_outcome = by_outcome.reindex(["caught", "missed"]).fillna(0.0)

    worst = detail[detail["outcome"] == "missed"].nlargest(n_rows, "amount").drop(
        columns="outcome"
    ).reset_index(drop=True)

    return Missed(
        confusion=confusion(y_true, y_score, threshold),
        by_outcome=by_outcome,
        worst=worst,
        missed_amount_share=float(by_outcome.at["missed", "amount_share"])
    )
