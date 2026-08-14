from pathlib import Path

REPORTS_DIR = Path("reports")

MIN_PRECISION = 0.9
REVIEW_COST = 3.0
ALERT_BUDGET = 100
OBJECTIVES = ("cost", "precision", "budget")

SHAP_SAMPLE = 4000   # legitimate transactions explained; every fraud is kept regardless
SHAP_FEATURES = 15   # rows in the beeswarm and in the ranking table
