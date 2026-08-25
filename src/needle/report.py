import numpy as np
import pandas as pd
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from .common import CV_REPEATS, CV_SPLITS, recall_key
from .compare import variance_inflation
from .config import ALERT_BUDGET, ALPHA, MIN_PRECISION, REPORTS_DIR, REVIEW_COST, SHAP_FEATURES


@dataclass(frozen=True)
class Findings:
    seed: int
    day_one: tuple[int, int]      # rows, frauds
    day_two: tuple[int, int]
    winner: str
    params: dict
    leaderboard: pd.DataFrame
    cv_pr_auc: float
    cv_pr_auc_std: float
    holdout: dict                 # metric name -> value, on day 2
    chosen: dict                  # the day-1 out-of-fold operating point
    kept_threshold: dict          # day 2, same threshold
    kept_rate: dict               # day 2, same alert rate
    confusion: pd.DataFrame
    by_outcome: pd.DataFrame
    worst_missed: pd.DataFrame
    missed_amount_share: float
    units: str
    comparisons: pd.DataFrame | None = None   # winner against the leaderboard's next rows
    n_folds: int = CV_SPLITS * CV_REPEATS
    shap_ranking: pd.DataFrame | None = None
    shap_rows: int = 0
    explainer: str = ""
    figures: list[Path] = field(default_factory=list)

    # the economics the run was scored against, so the prose quotes the run and
    # not the defaults it may have been overridden with
    min_precision: float = MIN_PRECISION
    review_cost: float = REVIEW_COST
    alert_budget: int = ALERT_BUDGET
    alpha: float = ALPHA
    shap_features: int = SHAP_FEATURES
    reports_dir: Path = REPORTS_DIR


def _cell(value, precision: int) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "-"
        if value != 0 and abs(value) < 10 ** -precision:
            return f"{value:.2e}"
        return f"{value:,.{precision}f}"
    return str(value)


def _table(
    frame: pd.DataFrame,
    index: bool = False,
    precision: int | dict[str, int] = 4
) -> str:
    """A markdown table, hand-rolled to keep `tabulate` out of the dependency list.

    `precision` may name columns individually: amounts want two decimals in the same
    table where a probability score wants six.
    """
    frame = frame.copy()
    if index:
        frame.insert(0, frame.index.name or "", frame.index)

    header = [str(column) for column in frame.columns]
    digits = [
        precision.get(name, 4) if isinstance(precision, dict) else precision for name in header
    ]
    rows = [
        [_cell(value, digits[column]) for column, value in enumerate(row)]
        for row in frame.itertuples(index=False)
    ]
    widths = [max(len(header[i]), *(len(row[i]) for row in rows or [header])) for i in range(len(header))]

    lines = [
        "| " + " | ".join(name.ljust(width) for name, width in zip(header, widths)) + " |",
        "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    ]
    lines += [
        "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join(lines)


def fill(text: str, width: int = 100, indent: str = "") -> str:
    # NOTE: assembled prose gets the same wrap as the hand-written paragraphs around it, so the
    # raw markdown stays readable in a diff. break_long_words stays off so a `back-quoted/label`
    # is never split across lines.
    return textwrap.fill(
        text, width=width, subsequent_indent=indent,
        break_long_words=False, break_on_hyphens=False
    )


def _p(value: float) -> str:
    # NOTE: these p-values span 1.0 down to 1e-16, so fixed notation would lose every convincing
    # one to '0.000' and scientific notation would print a plain 1.0 as '1e+00'
    return f"{value:.3f}" if value >= 5e-4 else f"{value:.2e}"


def _point(name: str, point: dict) -> pd.DataFrame:
    return pd.DataFrame([{"point": name, **{
        key: point[key]
        for key in ("threshold", "alerts_per_day", "alert_rate", "precision", "recall", "cost")
    }}])


def _operating_points(findings: Findings) -> pd.DataFrame:
    return pd.concat([
        _point("chosen (day 1, out-of-fold)", findings.chosen),
        _point("day 2, same threshold", findings.kept_threshold),
        _point("day 2, same alert rate", findings.kept_rate)
    ], ignore_index=True)


def _metric_choice(findings: Findings) -> str:
    roc_auc, pr_auc = findings.holdout.get("roc_auc", float("nan")), findings.holdout["pr_auc"]
    rows, frauds = findings.day_two
    recall_at_p = findings.holdout.get(recall_key(findings.min_precision), float("nan"))

    return f"""## 1. The metric, and why it is not ROC-AUC

On day 2 the winning model scores **ROC-AUC {roc_auc:.4f}** and **PR-AUC {pr_auc:.4f}**.
Both numbers describe the same {frauds} frauds among {rows:,} transactions; only one of them is usable.
ROC's x axis divides false positives by all {rows - frauds:,} negatives, which compresses the whole
range between a deployable model and an unusable one into the leftmost sliver of the curve.
Precision divides by the alerts raised instead — the quantity a review team actually pays for.

Reported here, in order of what they decide:

- **PR-AUC (average precision)** — the primary, threshold-free metric. Model selection ranks on
  it and the Optuna search optimises it.
- **Recall at precision >= {findings.min_precision:.0%}** — {recall_at_p:.4f} on day 2. What a fraud team
  buys: the share of fraud caught while most alerts are still genuine.
- **Cost = missed amount + {findings.review_cost:g} x false positives** — used to pick the threshold, and
  the only number here that treats a missed EUR 2,000 fraud differently from a missed EUR 5 one.
- **ROC-AUC** — kept for comparability with published results on this dataset, where 0.97-0.98 is
  unremarkable and says almost nothing about deployability."""


def _validation(findings: Findings) -> str:
    rows_one, frauds_one = findings.day_one
    rows_two, frauds_two = findings.day_two
    gap = findings.cv_pr_auc - findings.holdout["pr_auc"]
    within = "inside" if gap <= findings.cv_pr_auc_std else "outside"

    return f"""## 2. Splits and leakage precautions

Day 1 ({rows_one:,} rows, {frauds_one} frauds) does everything: hyperparameter search, model selection, threshold choice.
Day 2 ({rows_two:,} rows, {frauds_two} frauds) is scored once, at the end.

- **Exact duplicates dropped before splitting.** Identical rows landing on both sides of a
  split is the cheapest way to fake a good score on this dataset.
- **Resampling and scaling live inside an `imblearn.Pipeline`**, so both are fitted on the
  training part of each fold and never on the fold being scored. Oversampling before splitting
  is the usual error in published notebooks here, and it produces near-perfect fake numbers.
- **The threshold is chosen on day-1 out-of-fold scores**, not on day 2, and not on the same
  rows the model was fitted to.
- **The search never saw day 2.** Tuning ran on day-1 folds only.
- Seed {findings.seed} throughout, recorded with every tuning result under `{findings.reports_dir}/`.

### Cross-validation against the temporal holdout

Repeated stratified 5-fold on day 1 gives PR-AUC **{findings.cv_pr_auc:.4f} +/- {findings.cv_pr_auc_std:.4f}**.
The same pipeline refitted on all of day 1 and scored on day 2 gives **{findings.holdout['pr_auc']:.4f}**,
a drop of **{gap:.4f}** — {within} one standard deviation of the CV spread.

That gap is the finding, not a bug. Random folds let a model train on frauds that happened minutes
either side of the ones it is scored on, and the temporal split does not. The honest expectation
for a fresh day is the holdout number; anything quoted from cross-validation alone is optimistic."""


def _verdict(findings: Findings) -> str:
    frame = findings.comparisons
    tied = frame.loc[~frame["significant"], "label"].tolist()
    separated, total = len(frame) - len(tied), len(frame)
    runner_up = frame.iloc[0]
    listed = ", ".join(f"`{label}`" for label in tied)

    # NOTE: the leaderboard is sorted on the mean of these very folds, so the gap is normally
    # positive; the sign is read anyway rather than assumed, since the family is an argument
    gap = float(runner_up["difference"])
    lead = (
        f"The runner-up, `{runner_up['label']}`, sits {abs(gap):.4f} PR-AUC "
        f"{'below' if gap >= 0 else 'above'} the winner, and that gap "
        + ("survives" if bool(runner_up["significant"]) else "does not survive")
        + f" the test (corrected p={_p(runner_up['p_value'])}, Holm-adjusted "
          f"{_p(runner_up['p_holm'])})."
    )

    if separated == total:
        return (f"{lead} So do all {total} of the challengers tested: the top of this leaderboard "
                f"is a real ordering and not a tie.")
    if separated == 0:
        return (f"{lead} Neither does any of the other {total - 1}. On {findings.n_folds} folds of "
                f"one day's transactions this experiment cannot separate {listed} from the winner "
                f"at all - the leaderboard has an order because floats always do, not because the "
                f"data supports one.")
    return (f"{lead} {separated} of the {total} challengers "
            f"{'is' if separated == 1 else 'are'} separated from the winner; {listed} "
            f"{'is' if len(tied) == 1 else 'are'} not, and should be read as tied with it rather "
            f"than beaten by it.")


def _significance(findings: Findings) -> str:
    if findings.comparisons is None or findings.comparisons.empty:
        return """## 3. Is the leaderboard order real?

Skipped: the run passed `--no-tests`, or only one candidate was scored."""

    frame = findings.comparisons
    candidates = len(findings.leaderboard)
    inflation = variance_inflation(findings.n_folds)
    table = _table(frame, precision={
        "difference": 4, "t": 2, "p_value": 4, "p_holm": 4, "p_naive": 4
    })

    gap = abs(float(frame.iloc[0]["difference"]))
    motive = fill(
        f"The winner leads the runner-up by {gap:.4f} PR-AUC against the fold-to-fold spread of "
        f"{findings.cv_pr_auc_std:.4f} quoted above - a lead "
        + ("narrower" if gap < findings.cv_pr_auc_std else "wider")
        + " than the noise it was measured in, which is reason to test the ordering rather than "
          "assert it."
    )

    return f"""## 3. Is the leaderboard order real?

{motive}

Every candidate was scored on the *same* {findings.n_folds} folds - one `RepeatedStratifiedKFold`, built once and
handed to each pipeline - so the scores pair up fold for fold and the comparison can be paired.

{table}

`difference` is the winner's mean PR-AUC advantage over that row. `p_value` is corrected as
described below, `p_holm` is that value adjusted across the family, and `p_naive` is what an
uncorrected paired t-test would have claimed.

### Why the naive p-value is wrong

{findings.n_folds} folds are not {findings.n_folds} independent measurements. Any two of the {CV_SPLITS} training sets in a
repeat share {CV_SPLITS - 2} of their {CV_SPLITS - 1} folds, and the {CV_REPEATS} repeats reuse every row in day 1. A plain
paired t-test divides the variance of the differences by {findings.n_folds} anyway, which *is* the independence
assumption, and the denominator it produces is far too small - it reports fold noise as a finding.

The corrected resampled t-test (Nadeau & Bengio, in the form Bouckaert & Frank recommend for
repeated k-fold) replaces `1/n` with `1/n + |test|/|train|`. Holding out one fold of {CV_SPLITS} fixes that
ratio at 1/{CV_SPLITS - 1} regardless of row count, which inflates the variance by {inflation:.2f}x and shrinks every
t-statistic by {inflation ** 0.5:.2f}x. Comparing the two p-value columns is the cheapest illustration of how
much a naive test overstates on this dataset.

The family is the {len(frame)} comparisons above and was fixed before any p-value was read; the p-values
are Holm-corrected across it, because testing the winner against {len(frame)} challengers on one set of
folds is {len(frame)} chances at a gap that is not there. Every pair of the {candidates} candidates scored would
have been {candidates * (candidates - 1) // 2} tests, which turns up a "significant" difference somewhere by construction.

### What it says

{fill(_verdict(findings))}

Two things this does not establish:

- **The winner was chosen on these same folds.** Taking the highest mean and then testing it
  against the rest is post-selection inference. These p-values are optimistic in a way the
  variance correction does nothing about, and the honest version needs a held-out selection set
  the {findings.day_one[1]} day-1 frauds cannot really afford.
- **Nothing here changed what ships.** The model carried to day 2 is still the top of the
  leaderboard, not "the cheapest candidate that survives the test". That rule would be the
  better engineering call and it is a different decision from this one."""


def _operating_point_section(findings: Findings) -> str:
    chosen, kept, rate = findings.chosen, findings.kept_threshold, findings.kept_rate
    caught = int(findings.confusion.at["fraud", "alerted"])
    frauds = int(findings.confusion.loc["fraud"].sum())
    false_positives = int(findings.confusion.at["legit", "alerted"])

    return f"""## 5. The operating point

Thresholds are in {findings.units} units. The shipped one minimises cost on day-1 out-of-fold
scores — never left at 0.5 — and the other two rows are what that same decision did on day 2.

{_table(_operating_points(findings), precision={"threshold": 6, "alerts_per_day": 0, "alert_rate": 5, "cost": 2})}

Read as a review queue: **{kept['alerts_per_day']:.0f} alerts on day 2** ({kept['alert_rate'] * 100:.3f}% of transactions), {caught} of them fraud and {false_positives:,} not.
That is precision **{kept['precision']:.1%}** at recall **{kept['recall']:.1%}** of the {frauds} frauds present.
A team that can review {findings.alert_budget} transactions a day would have to cut the queue to its top {findings.alert_budget / max(kept['alerts_per_day'], 1) * 100:.0f}%.

The threshold was picked once and carried unchanged, which is the deployable move. Holding the
*alert rate* fixed instead lands at {rate['alerts_per_day']:.0f} alerts and recall {rate['recall']:.1%};
the distance between those two rows is score drift between day 1 and day 2, and it is the
argument for alarming on alert volume rather than trusting a frozen threshold forever."""


def _interpretation(findings: Findings) -> str:
    if findings.shap_ranking is None:
        return """## 4. What the model uses

Skipped: the winning model has no SHAP explainer available (an unsupervised detector with no
tree or linear structure to read)."""

    ranking = findings.shap_ranking
    top = ranking.head(findings.shap_features)
    leaders = ", ".join(f"`{name}`" for name in ranking["feature"].head(3))
    concentration = float(ranking["share"].head(5).sum())

    engineered = [
        f"`{row.feature}` ranks {row.rank} ({row.share:.1%} of the total)"
        for row in ranking[ranking["feature"].isin(["log_amount", "hour"])].itertuples()
    ]
    engineered_text = " and ".join(engineered) if engineered else "neither one appears"

    return f"""## 4. What the model uses

SHAP values from the `{findings.explainer}` explainer, over {findings.shap_rows:,} day-2 transactions: every fraud, plus a sample of legitimate ones.
Frauds are over-represented against the 0.17% base rate on purpose, so the ranking below
answers "what separates the two classes", not "what an average transaction looks like".

{_table(top, precision={"mean_abs_shap": 4, "share": 3, "value_correlation": 2})}

{leaders} lead, and the top five features carry {concentration:.1%} of all attributed magnitude.
`value_correlation` is a direction rather than a strength: negative means it is *low* values of
that feature that push a transaction towards fraud.

Of the two features that are not PCA output, {engineered_text}.

`V1`-`V28` are components of an undisclosed transform, so none of these names can be read back
to anything a fraud analyst would recognise. What the ranking is good for is narrower and still
worth having: the decision rests on a handful of components rather than being smeared across all
thirty, which is what makes it auditable at all, and a shift in this ranking between retrains is
a usable drift alarm on its own.

![SHAP beeswarm](shap_beeswarm.png)"""


def _misses(findings: Findings) -> str:
    missed = int(findings.confusion.at["fraud", "cleared"])
    caught = int(findings.confusion.at["fraud", "alerted"])
    share = findings.missed_amount_share
    # Against the share of frauds missed by count: below it, the misses are the cheap tail.
    verdict = (
        "cheaper per fraud than the ones caught, and recall understates the model"
        if share < missed / max(caught + missed, 1)
        else "worth more per fraud than the ones caught, and recall flatters the model"
    )

    alerts = findings.kept_threshold["alerts_per_day"]

    return f"""## 6. The confusion matrix, and what got through

At the shipped threshold ({findings.kept_threshold['threshold']:.6g}) on day 2 — where "alerted"
means "queued for a human", not "declined":

{_table(findings.confusion, index=True, precision=0)}

Counting the {missed} missed frauds is the wrong way to read that, because they are not worth the
same. By amount:

{_table(findings.by_outcome, index=True, precision={"total_amount": 2, "mean_amount": 2, "median_amount": 2, "max_amount": 2, "amount_share": 3})}

The misses carry **{share:.1%}** of the fraudulent money on day 2 against {missed / max(caught + missed, 1):.1%} of the fraud count: they are {verdict}.

The largest of them, with the queue position each one would have needed a reviewer to reach:

{_table(findings.worst_missed, precision={"amount": 2, "score": 6})}

`queue_rank` prices the alternative to a better model: anything ranked not far past the
{alerts:.0f} alerts already raised is caught by simply lengthening the queue, while a fraud ranked
in the thousands is a ranking failure that no threshold fixes.

![Frauds by amount and score](missed_frauds.png)"""


def _limitations(findings: Findings) -> str:
    frauds = findings.day_one[1] + findings.day_two[1]

    if findings.comparisons is None or findings.comparisons.empty:
        ranking = ("The winner beat the runner-up by less than the fold-to-fold spread, so "
                   f'"best model" here means "best on this split at seed {findings.seed}", '
                   "not a general ranking.")
    else:
        runner_up = findings.comparisons.iloc[0]
        ranking = (
            f"The winner's {abs(float(runner_up['difference'])):.4f} PR-AUC lead over "
            f"`{runner_up['label']}` "
            + ("survives" if bool(runner_up["significant"]) else "does not survive")
            + f" a corrected paired t-test (section 3, Holm-adjusted p={_p(runner_up['p_holm'])}), "
              f'and either way "best model" here means "best on this split at seed '
              f'{findings.seed}", not a general ranking.'
        )
    ranking = fill(
        f"- **One seed, one dataset.** {ranking} The test in section 3 measures how far this "
        f"split can be trusted; it cannot make the split representative.",
        indent="  "
    )

    return f"""## 7. Limitations

- **{frauds} frauds in total, {findings.day_one[1]} of them on the training side.** Every number
  here carries the variance that implies; the CV spread quoted above is not a rounding error,
  it is the honest resolution of the experiment.
- **Two days of data.** The temporal holdout is one day against one day. It shows that a gap
  exists; it cannot measure how fast the model decays, and there is no second week to check
  against.
- **Anonymised features.** `V1`-`V28` are PCA components of undisclosed inputs, which rules
  out domain sanity checks, feature repair, and any claim that the model is using something a
  fraud team would endorse.
- **The cost model is an assumption.** It prices a missed fraud at the full transaction amount
  and a review at a flat {findings.review_cost:g}. Real recovery rates, chargeback fees, and the cost of
  a wrongly blocked customer are all missing, and the cost-optimal threshold moves with them —
  the sensitivity table in the run output shows how far.
{ranking}
- **No drift validation.** Nothing here can be checked against real fraud drift, novel attack
  patterns, or an adversary reacting to the model - which is the case for keeping an
  unsupervised detector in the comparison even though it loses on PR-AUC."""


def write_summary(
    findings: Findings,
    name: str = "summary.md",
    directory=None
) -> Path:
    parameters = "\n".join(f"- `{key}`: {value}" for key, value in sorted(findings.params.items()))
    leaderboard = findings.leaderboard.head(6)[
        [column for column in ("label", "pr_auc_mean", "pr_auc_std", "roc_auc_mean", "fit_seconds")
         if column in findings.leaderboard.columns]
    ]

    document = f"""# Needle - credit card fraud detection

Written by `needle` from the run that produced the figures beside it. Seed {findings.seed}.

**Winner: `{findings.winner}`** — PR-AUC {findings.cv_pr_auc:.4f} +/- {findings.cv_pr_auc_std:.4f} in repeated cross-validation on day 1, {findings.holdout['pr_auc']:.4f} on the day-2 holdout.

{_table(leaderboard, precision={"fit_seconds": 2})}

Its tuned hyperparameters:

{parameters or "- (defaults)"}

{_metric_choice(findings)}

{_validation(findings)}

{_significance(findings)}

{_interpretation(findings)}

{_operating_point_section(findings)}

{_misses(findings)}

{_limitations(findings)}

## Figures

{chr(10).join(f"- `{Path(path).name}`" for path in findings.figures) or "- (none written)"}
"""

    path = Path(directory if directory is not None else REPORTS_DIR) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)
    return path
