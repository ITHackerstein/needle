"""Figures for the report.

Split out of evaluate.py, which the plan puts them in, for one reason: importing
matplotlib pulls a font cache and a backend into every process that only wanted
average_precision_score - including the Optuna workers. The metrics stay importable
without a graphics stack.
"""
import matplotlib
matplotlib.use("Agg")  # written to reports/, never shown, so never require a display

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve
from .calibrate import calibration_report
from .config import REPORTS_DIR
from .threshold import sweep_thresholds


def _save(figure, name: str, directory=None) -> Path:
    path = Path(directory if directory is not None else REPORTS_DIR) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _mark(axis, x: float, y: float, text: str, fraction: float = 0.65) -> None:
    """Ring the point and label it, leaning the label back inside the axes.

    `fraction` is where along the x range the label flips to the left; without it a
    point near recall 1.0 - which is where a cost-optimal threshold lands - writes its
    caption off the right edge of the figure.
    """
    axis.plot([x], [y], marker="o", markersize=9, markerfacecolor="none",
              markeredgewidth=2, color="crimson", zorder=5)

    low, high = axis.get_xlim()
    position = x
    if axis.get_xscale() == "log":  # judge position as drawn, not as numbered
        low, high, position = np.log10(low), np.log10(high), np.log10(x)
    on_the_right = position > low + fraction * (high - low)
    axis.annotate(
        text, (x, y),
        textcoords="offset points",
        xytext=(-12, -28) if on_the_right else (12, 10),
        horizontalalignment="right" if on_the_right else "left",
        fontsize=8, color="crimson", zorder=6
    )


def precision_recall(curves: dict, chosen: dict | None = None,
                     name: str = "precision_recall.png", directory=None) -> Path:
    """PR curves with the chosen operating point marked, which is the §4 deliverable.

    `curves` maps a label to (y_true, y_score); the dashed line is the base rate, the
    PR-AUC a random ranker would get. Unlike a ROC plot, the interesting region here
    is not a corner - the whole square is in play.
    """
    figure, axis = plt.subplots(figsize=(6.5, 5))

    for label, (y_true, y_score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        axis.step(recall, precision, where="post", linewidth=1.6,
                  label=f"{label} (PR-AUC {pr_auc:.3f})")

    first = next(iter(curves.values()))
    base_rate = float(np.asarray(first[0]).mean())
    axis.axhline(base_rate, linestyle="--", linewidth=1, color="grey",
                 label=f"base rate {base_rate:.5f}")

    axis.set_xlabel("recall")
    axis.set_ylabel("precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_title("Precision-recall, with the chosen threshold")
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=0.3)

    # After the limits, because _mark reads them to decide which way to lean the label.
    if chosen is not None:
        _mark(axis, chosen["recall"], chosen["precision"],
              f"chosen: {chosen['alerts_per_day']:.0f} alerts/day\n"
              f"P={chosen['precision']:.2f} R={chosen['recall']:.2f}")
    return _save(figure, name, directory)


def cost_vs_alerts(y_true, y_score, amounts, chosen: dict | None = None, days: float = 1.0,
                   review_cost=None, name: str = "cost_vs_alerts.png", directory=None) -> Path:
    """Amount-weighted cost against queue size, log x because the deployable band
    spans three orders of magnitude and the optimum sits at the low end.
    """
    figure, axis = plt.subplots(figsize=(6.5, 5))

    kwargs = {} if review_cost is None else {"review_cost": review_cost}
    frame = sweep_thresholds(y_true, y_score, amounts, days=days, **kwargs)
    axis.plot(frame["alerts_per_day"], frame["cost"], linewidth=1.6)

    axis.set_xscale("log")
    axis.set_xlabel("alerts per day")
    axis.set_ylabel("cost (missed amount + review cost x false positives)")
    axis.set_title("Cost against queue size")
    axis.grid(alpha=0.3, which="both")

    if chosen is not None:
        _mark(axis, chosen["alerts_per_day"], chosen["cost"],
              f"chosen: {chosen['alerts_per_day']:.0f} alerts/day\ncost {chosen['cost']:,.0f}")
    return _save(figure, name, directory)


def reliability(curves: dict, n_bins: int = 10,
                name: str = "reliability.png", directory=None) -> Path:
    """Reliability curves, `curves` mapping a label to (y_true, y_prob).

    Log x because predicted probabilities here span several orders of magnitude, but
    linear y: most quantile bins contain no fraud at all, and on a log y axis every
    one of those honest zeros would have to be thrown away.
    """
    figure, axis = plt.subplots(figsize=(6.5, 5))

    limits = [1.0, 0.0]
    for label, (y_true, y_prob) in curves.items():
        table, brier = calibration_report(y_true, y_prob, n_bins=n_bins)
        drawable = table[table["mean_score"] > 0]
        axis.plot(drawable["mean_score"], drawable["fraction_positive"],
                  marker="o", markersize=4, linewidth=1.4, label=f"{label} (Brier {brier:.5f})")
        if len(drawable):
            limits = [min(limits[0], drawable["mean_score"].min()),
                      max(limits[1], drawable["mean_score"].max())]

    diagonal = np.geomspace(max(limits[0], 1e-8), max(limits[1], 1e-7), 50)
    axis.plot(diagonal, diagonal, linestyle="--", linewidth=1, color="grey",
              label="perfectly calibrated")

    axis.set_xscale("log")
    axis.set_xlabel("mean predicted probability (quantile bins)")
    axis.set_ylabel("observed fraud rate")
    axis.set_title("Reliability")
    axis.legend(loc="upper left", fontsize=8)
    axis.grid(alpha=0.3, which="both")
    return _save(figure, name, directory)
