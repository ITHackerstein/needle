from dataclasses import dataclass
from pathlib import Path

from .common import SEED

DATA_PATH = Path("dataset/creditcard.csv")
REPORTS_DIR = Path("reports")

MIN_PRECISION = 0.9
REVIEW_COST = 3.0
ALERT_BUDGET = 100
OBJECTIVES = ("cost", "precision", "budget")
OBJECTIVE = "cost"

SHAP_SAMPLE = 4000   # legitimate transactions explained; every fraud is kept regardless
SHAP_FEATURES = 15   # rows in the beeswarm and in the ranking table


@dataclass(frozen=True)
class Settings:
    # NOTE: one run's worth of choices. The constants above are only the defaults - the
    # pipeline reads its values from here, so a CLI flag reaches every module.
    data: Path = DATA_PATH
    reports_dir: Path = REPORTS_DIR
    objective: str = OBJECTIVE
    seed: int = SEED

    min_precision: float = MIN_PRECISION
    review_cost: float = REVIEW_COST
    alert_budget: int = ALERT_BUDGET
    shap_sample: int = SHAP_SAMPLE
    shap_features: int = SHAP_FEATURES

    models: tuple[str, ...] | None = None   # None -> tune.DEFAULT_TUNED
    trials: int | None = None               # None -> the per-model budget in tune.SPACES
    retune: bool = False
    plots: bool = True
    report: bool = True
    quiet: bool = False

    def __post_init__(self):
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
