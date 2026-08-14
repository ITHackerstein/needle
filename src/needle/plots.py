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
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve
from .calibrate import calibration_report
from .common import SEED
from .config import REPORTS_DIR, SHAP_FEATURES
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


def _percentile(values: np.ndarray) -> np.ndarray:
    """Rank-normalise to [0, 1] so a single outlier cannot own the whole colour scale."""
    order = np.argsort(values, stable=True)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size)
    return ranks / max(values.size - 1, 1)


def _swarm(values: np.ndarray, spread: float = 0.32, bins: int = 40, seed: int = SEED) -> np.ndarray:
    """Vertical offsets whose width follows how crowded that slice of the axis is.

    Uniform jitter would make a spike at zero - where nearly every point sits, this
    being a fraud model - look as wide as the tail that actually matters.
    """
    counts, edges = np.histogram(values, bins=bins)
    crowding = counts[np.clip(np.searchsorted(edges, values, side="right") - 1, 0, bins - 1)]
    scale = np.sqrt(crowding / max(counts.max(), 1))
    return np.random.default_rng(seed).uniform(-1.0, 1.0, values.size) * spread * scale


def shap_beeswarm(values, features: pd.DataFrame, n_features: int = SHAP_FEATURES,
                  name: str = "shap_beeswarm.png", directory=None) -> Path:
    """One row per feature, one dot per explained transaction, coloured by feature value.

    The §5 deliverable. The x axis is the model's own score unit (log-odds for the
    boosted trees), so distance from zero is how far that feature moved that
    transaction up or down the review queue.
    """
    values = np.asarray(values, dtype=np.float64)
    ordered = np.argsort(np.abs(values).mean(axis=0))[-n_features:]  # least important first

    figure, axis = plt.subplots(figsize=(7.5, 0.34 * len(ordered) + 1.8))
    axis.axvline(0, linewidth=1, color="grey", zorder=1)

    dots = None
    for row, column in enumerate(ordered):
        contribution = values[:, column]
        dots = axis.scatter(
            contribution, row + _swarm(contribution),
            c=_percentile(features.iloc[:, column].to_numpy(dtype=np.float64)),
            cmap="coolwarm", vmin=0.0, vmax=1.0,
            s=7, alpha=0.65, linewidths=0, zorder=2
        )

    axis.set_yticks(range(len(ordered)), [features.columns[column] for column in ordered])
    axis.set_ylim(-0.7, len(ordered) - 0.3)
    axis.set_xlabel("SHAP value (contribution towards fraud)")
    axis.set_title(f"What the model uses: top {len(ordered)} features")
    axis.grid(axis="x", alpha=0.3)

    if dots is not None:
        figure.colorbar(dots, ax=axis, pad=0.02).set_label(
            "feature value (percentile)", fontsize=8
        )
    return _save(figure, name, directory)


def missed_frauds(y_true, y_score, amounts, threshold: float,
                  name: str = "missed_frauds.png", directory=None) -> Path:
    """Every fraud placed by amount against the score it got, with the threshold drawn.

    Answers the question §5 asks about false negatives directly: if the misses sit
    in the bottom-left they are cheap, and if any sit in the bottom-right they are
    the ones worth another feature.
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    amounts = np.asarray(amounts, dtype=np.float64)

    frauds = np.flatnonzero(y_true == 1)
    caught = y_score[frauds] >= threshold

    figure, axis = plt.subplots(figsize=(6.5, 5))
    for label, mask, colour in (
        (f"caught ({int(caught.sum())})", caught, "tab:green"),
        (f"missed ({int((~caught).sum())})", ~caught, "crimson")
    ):
        selected = frauds[mask]
        axis.scatter(np.maximum(amounts[selected], 0.01), y_score[selected],
                     s=22, alpha=0.75, color=colour, linewidths=0, label=label)

    axis.axhline(threshold, linestyle="--", linewidth=1, color="black",
                 label=f"threshold {threshold:.4g}")
    axis.set_xscale("log")
    # A decision function can be negative; only probabilities earn a log score axis.
    if threshold > 0 and y_score[frauds].min() > 0:
        axis.set_yscale("log")
    axis.set_xlabel(
        "transaction amount (log, zeros pinned to 0.01)" if amounts[frauds].min() <= 0
        else "transaction amount (log)"
    )
    axis.set_ylabel("model score")
    axis.set_title("Frauds by amount and score, at the shipped threshold")
    axis.legend(loc="lower right", fontsize=8)
    axis.grid(alpha=0.3, which="both")
    return _save(figure, name, directory)
