import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from . import compare, interpret, models as _models, plots
from .common import cross_validation_split, probability, ranking, recall_key, take
from .config import OBJECTIVES, Settings
from .data import load_data, temporal_split, extract_features
from .models import CANDIDATE_PIPELINES
from .evaluate import FOLD_COLUMN, cross_validate_candidates, recall_at_precision
from .threshold import (
    apply_threshold,
    review_cost_sensitivity,
    select_threshold,
    selection_scores,
    threshold_from_rate,
    threshold_table
)
from .calibrate import calibrated_candidate, calibration_report
from .report import Findings, write_summary
from .tune import (
    DEFAULT_TUNED,
    load_tuning,
    result_path,
    save_tuning,
    tune_model,
    tuned_candidate
)

POINT_COLUMNS = ["threshold", "alert_rate", "alerts_per_day", "precision", "recall", "cost"]

_verbose = True

def say(*args, **kwargs) -> None:
    # NOTE: every progress line goes through here, so --quiet has a single switch
    if _verbose:
        print(*args, **kwargs)

def _prepare(settings: Settings) -> None:
    global _verbose
    _verbose = not settings.quiet

    # The model factories in models.py read this global when they build an estimator,
    # so setting it here is what makes --seed reach the estimators themselves and not
    # just the splits.
    _models.SEED = settings.seed

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

def _days(settings: Settings) -> tuple:
    df = load_data(settings.data)
    split = temporal_split(df["Time"])
    X, y = extract_features(df)

    first = (X.iloc[split.train], y.iloc[split.train], df["Amount"].iloc[split.train])
    second = (X.iloc[split.test], y.iloc[split.test], df["Amount"].iloc[split.test])
    say(f"day 1: {len(first[0]):,} rows, {int(first[1].sum())} frauds")
    say(f"day 2: {len(second[0]):,} rows, {int(second[1].sum())} frauds")
    return first, second

def _holdout(candidate, X_first, y_first, X_second, y_second, settings: Settings) -> tuple:
    say(f"\n=== temporal holdout: {candidate.label()} ===")
    model = candidate.build().fit(X_first, y_first)
    scores, probabilities = ranking(model, X_second), probability(model, X_second)

    metrics = {
        "pr_auc": float(average_precision_score(y_second, scores)),
        "roc_auc": float(roc_auc_score(y_second, scores)),  # secondary, for comparability only
        recall_key(settings.min_precision): recall_at_precision(
            y_second, scores, min_precision=settings.min_precision
        )
    }
    for name, value in metrics.items():
        say(f"  {name:22} {value:.4f}")
    return model, scores, probabilities, metrics

def _tuned_candidates(X, y, settings: Settings, revalidate: bool = False) -> list:
    candidates = []
    for model_name in settings.models or DEFAULT_TUNED:
        path = result_path(model_name, settings.reports_dir)
        if not settings.retune and path.exists():
            result = load_tuning(model_name, path)
            say(f"  {model_name:20} reusing {path} ({result.n_trials} trials, "
                f"search pr_auc={result.search_pr_auc:.4f})")
        else:
            say(f"\n  ### searching {model_name} - this is the slow part ###")
            result = tune_model(
                model_name, X, y,
                n_trials=settings.trials,
                revalidate=revalidate,
                seed=settings.seed,
                verbose=not settings.quiet
            )
            say(f"  wrote {save_tuning(result, path)}")

        candidates.append(tuned_candidate(result))
    return candidates

def _significance(fold_scores: dict, leaderboard, settings: Settings):
    if not settings.tests:
        return None

    say("\n=== is the leaderboard order real? (corrected paired t, day-1 folds) ===")
    comparisons = compare.against_winner(
        fold_scores, leaderboard["label"],
        n_compare=settings.n_compare, alpha=settings.alpha
    )
    if comparisons.empty:
        say("  only one candidate was scored, so there is nothing to compare")
        return comparisons

    say(comparisons.to_string(index=False))

    tied = comparisons.loc[~comparisons["significant"], "label"].tolist()
    say(f"\n  {len(comparisons) - len(tied)} of {len(comparisons)} separated from the winner "
        f"at alpha={settings.alpha:g}, Holm-corrected across the family")
    if tied:
        say(f"  indistinguishable on these folds: {', '.join(tied)}")
    # NOTE: reporting only, on purpose. Switching the selection rule to 'the cheapest candidate
    # that survives the test' would change what ships, which is a separate decision.
    say("  reporting only - the shipped model is still the top of the leaderboard")
    return comparisons

def _operating_point(candidate, X, y, amounts, settings: Settings) -> tuple[dict, "pd.Series"]:
    say("\n=== threshold selection (day 1, out-of-fold) ===")
    oof_ranking, oof_probabilities = selection_scores(candidate, X, y, seed=settings.seed)
    validation_scores = oof_ranking if oof_probabilities is None else oof_probabilities

    choices = pd.DataFrame(
        select_threshold(
            y, validation_scores, amounts,
            objective=objective,
            alert_budget=settings.alert_budget,
            min_precision=settings.min_precision,
            review_cost=settings.review_cost
        )
        for objective in OBJECTIVES
    )
    say(choices.to_string(index=False))

    say("\n  where the cost optimum moves with the price of one review:")
    say(review_cost_sensitivity(y, validation_scores, amounts).to_string(index=False))

    chosen = choices[choices["objective"] == settings.objective].iloc[0].to_dict()
    say(f"\n  shipping '{settings.objective}': threshold {chosen['threshold']:.6f}, "
        f"the top {chosen['alert_rate'] * 100:.3f}% of transactions")
    return chosen, validation_scores

def _transfer(chosen: dict, y_true, y_score, amounts, settings: Settings) -> tuple[dict, dict]:
    say("\n=== the chosen point, carried to day 2 ===")
    by_threshold = apply_threshold(
        y_true, y_score, chosen["threshold"], amounts, review_cost=settings.review_cost
    )
    by_rate = apply_threshold(
        y_true, y_score, threshold_from_rate(y_score, chosen["alert_rate"]), amounts,
        review_cost=settings.review_cost
    )
    rows = {
        "expected (day 1 out-of-fold)": chosen,
        "achieved (same threshold)": by_threshold,
        "achieved (same alert rate)": by_rate
    }
    say(pd.DataFrame(rows).T[POINT_COLUMNS].astype(float).to_string())
    return by_threshold, by_rate

def _interpretation(model, X, y, y_score, amounts, threshold: float, settings: Settings) -> tuple:
    say("\n=== what the model uses (SHAP, day 2) ===")
    sample = interpret.explanation_sample(y, n_negatives=settings.shap_sample, seed=settings.seed)
    explanation = interpret.explain(
        model, take(X, sample), n_features=settings.shap_features, verbose=not settings.quiet
    )

    say(f"\n=== the shipped threshold on day 2 ({threshold:.6g}) ===")
    missed = interpret.missed_frauds(y, y_score, threshold, amounts)
    say(missed.confusion.to_string())

    say("\n  the frauds, by what they were worth:")
    say(missed.by_outcome.to_string())
    say(f"\n  the misses carry {missed.missed_amount_share:.1%} of the fraudulent amount")

    say("\n  the largest ones that got through:")
    say(missed.worst.to_string(index=False))
    return explanation, missed, sample


def _calibration(candidate, X_first, y_first, X_second, y_second, raw_probabilities):
    if raw_probabilities is None:
        say("\n=== calibration: skipped, the winning model emits no probabilities ===")
        return None

    say("\n=== calibration (fit on day 1, checked on day 2) ===")
    model = calibrated_candidate(candidate).fit(X_first, y_first)
    calibrated_probabilities = model.predict_proba(X_second)[:, 1]

    for label, scores in (("raw", raw_probabilities), ("sigmoid", calibrated_probabilities)):
        table, brier = calibration_report(y_second, scores)
        say(f"\n  {label}: brier={brier:.6f} "
            f"pr_auc={average_precision_score(y_second, scores):.4f} "
            f"mean_score={scores.mean():.6f} vs base rate {y_second.mean():.6f}")
        say(table.to_string(index=False))
    return calibrated_probabilities

def _figures(settings: Settings, **parts) -> list:
    if not settings.plots:
        return []

    say("\n=== figures ===")
    figures = [
        plots.precision_recall(parts["curves"], chosen=parts["achieved"],
                               directory=settings.reports_dir),
        plots.cost_vs_alerts(parts["y"], parts["scores"], parts["amounts"],
                             chosen=parts["achieved"], review_cost=settings.review_cost,
                             directory=settings.reports_dir),
        plots.missed_frauds(parts["y"], parts["scores"], parts["amounts"], parts["threshold"],
                            directory=settings.reports_dir)
    ]
    if parts["explanation"] is not None:
        figures.append(plots.shap_beeswarm(
            parts["explanation"].values, parts["explanation"].features,
            n_features=settings.shap_features, directory=settings.reports_dir
        ))
    if parts["calibrated"] is not None:
        figures.append(plots.reliability({
            "raw": (parts["y"], parts["probabilities"]),
            "sigmoid": (parts["y"], parts["calibrated"])
        }, directory=settings.reports_dir))

    for path in figures:
        say(f"  wrote {path}")
    return figures


def search(settings: Settings, revalidate: bool = True) -> list:
    _prepare(settings)
    (X_first, y_first, _), _ = _days(settings)

    say("\n=== hyperparameter search (day 1 only) ===")
    candidates = _tuned_candidates(X_first, y_first, settings, revalidate=revalidate)

    say("\n=== tuned candidates ===")
    for candidate in candidates:
        say(f"  {candidate.label()}")
    return candidates


def run(settings: Settings | None = None) -> None:
    settings = settings if settings is not None else Settings()
    _prepare(settings)

    (X_first, y_first, amounts_first), (X_second, y_second, amounts_second) = _days(settings)
    cv = cross_validation_split(seed=settings.seed)

    say("\n=== hyperparameter search (day 1 only) ===")
    tuned = _tuned_candidates(X_first, y_first, settings)
    candidates = list(CANDIDATE_PIPELINES) + tuned
    by_label = {candidate.label(): candidate for candidate in candidates}

    say("\n=== model selection (day 1 only) ===")
    rows = []
    for candidate in candidates:
        row = cross_validate_candidates(
            X_first, y_first, (candidate,), cv=cv, min_precision=settings.min_precision
        )
        say(f"  {row.at[0, 'label']:36} "
            f"pr_auc={row.at[0, 'pr_auc_mean']:.4f} ± {row.at[0, 'pr_auc_std']:.4f} "
            f"({row.at[0, 'fit_seconds']:.1f}s/fit)")
        rows.append(row)

    leaderboard = pd.concat(rows, ignore_index=True).sort_values(
        "pr_auc_mean", ascending=False, ignore_index=True
    )
    # NOTE: the fold vectors travel out of the leaderboard before it is printed or reported -
    # they are what the paired test needs, and an ndarray column renders as noise everywhere else
    fold_scores = dict(zip(leaderboard["label"], leaderboard[FOLD_COLUMN]))
    leaderboard = leaderboard.drop(columns=[FOLD_COLUMN])
    say("")
    say(leaderboard.to_string())

    best = leaderboard.iloc[0]
    winner = by_label[best["label"]]
    comparisons = _significance(fold_scores, leaderboard, settings)
    chosen, validation_scores = _operating_point(winner, X_first, y_first, amounts_first, settings)

    model, scores, probabilities, holdout = _holdout(
        winner, X_first, y_first, X_second, y_second, settings
    )
    say(f"\n  CV pr_auc {best['pr_auc_mean']:.4f} -> holdout, gap is the finding")

    units = "probability" if probabilities is not None else "decision function"
    table_scores = scores if probabilities is None else probabilities
    kept, achieved = _transfer(chosen, y_second, table_scores, amounts_second, settings)

    say(f"\n=== operating points (day 2, thresholds in {units} units) ===")
    say(threshold_table(
        y_second, table_scores, amounts_second, n_rows=12, review_cost=settings.review_cost
    ).to_string())

    calibrated_probabilities = _calibration(
        winner, X_first, y_first, X_second, y_second, probabilities
    )

    explanation, missed, sample = _interpretation(
        model, X_second, y_second, table_scores, amounts_second, kept["threshold"], settings
    )

    figures = _figures(
        settings,
        curves={
            "day 1 (out-of-fold)": (y_first, validation_scores),
            "day 2 (holdout)": (y_second, table_scores)
        },
        y=y_second,
        scores=table_scores,
        probabilities=probabilities,
        calibrated=calibrated_probabilities,
        amounts=amounts_second,
        threshold=kept["threshold"],
        achieved=achieved,
        explanation=explanation
    )

    if not settings.report:
        return

    say("\n=== summary ===")
    summary = write_summary(Findings(
        seed=settings.seed,
        day_one=(len(X_first), int(y_first.sum())),
        day_two=(len(X_second), int(y_second.sum())),
        winner=winner.label(),
        params=winner.params,
        leaderboard=leaderboard,
        cv_pr_auc=float(best["pr_auc_mean"]),
        cv_pr_auc_std=float(best["pr_auc_std"]),
        comparisons=comparisons,
        n_folds=len(fold_scores[best["label"]]),
        alpha=settings.alpha,
        holdout=holdout,
        chosen=chosen,
        kept_threshold=kept,
        kept_rate=achieved,
        confusion=missed.confusion,
        by_outcome=missed.by_outcome,
        worst_missed=missed.worst,
        missed_amount_share=missed.missed_amount_share,
        units=units,
        shap_ranking=None if explanation is None else explanation.ranking(),
        shap_rows=len(sample),
        explainer="" if explanation is None else explanation.explainer,
        figures=figures,
        min_precision=settings.min_precision,
        review_cost=settings.review_cost,
        alert_budget=settings.alert_budget,
        shap_features=settings.shap_features,
        reports_dir=settings.reports_dir
    ), directory=settings.reports_dir)
    say(f"  wrote {summary}")
