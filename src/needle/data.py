import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

def load_data(path: os.PathLike | str) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={
        "Time": np.float64,
        **{f"V{i}": np.float64 for i in range(1, 28 + 1)},
        "Amount": np.float64,
        "Class": np.uint8
    }).drop_duplicates(ignore_index=True)

    assert not data.isna().any().any(), "Unexpected missing values"
    return data

def extract_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    # NOTE: A cyclic sin/cos pair would be worth trying if
    # the linear baseline turns out to be competitive - the diurnal pattern is
    # roughly one sinusoid, which logistic regression can express but a raw
    # monotone hour cannot.
    hour = data["Time"] / 3600 % 24

    features = pd.DataFrame({
        "log_amount": np.log1p(data["Amount"]),
        "hour": hour,
        **{f"V{i}": data[f"V{i}"] for i in range(1, 28 + 1)}
    })
    return features, data["Class"]

@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray

    def __post_init__(self):
        assert not np.intersect1d(self.train, self.test).size, "Train and test sets must be disjoint"

def temporal_split(time, boundary: float = 60 * 60 * 24) -> Split:
    t = np.asarray(time)
    return Split(np.flatnonzero(t < boundary), np.flatnonzero(t >= boundary))
