import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

from .data import SEED, cross_validation_split
from .evaluate import ranking
from .models import make_model

SEARCH_FOLDS = 5
REPORTS_DIR = "reports"


@dataclass(frozen=True)
class TuningResult:
    model: str
    imbalance: str
    params: dict
    pr_auc_mean: float
    pr_auc_std: float
    search_pr_auc: float
    n_trials: int
    n_pruned: int
    fold_scores: list[float] = field(default_factory=list)
    seed: int = SEED


def _lgbm_space(trial: optuna.Trial, imbalance: str) -> dict:
    params = {
        "subsample_freq": 1,
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True)
    }
    if imbalance == "none":
        params["scale_pos_weight"] = trial.suggest_float("scale_pos_weight", 1.0, 1000.0, log=True)
    return params


def _logistic_regression_space(trial: optuna.Trial, imbalance: str) -> dict:
    l1_ratio = trial.suggest_categorical("l1_ratio", [0.0, 1.0])
    return {
        "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
        "l1_ratio": l1_ratio,
        "solver": "liblinear" if l1_ratio == 1.0 else "lbfgs"
    }


def _random_forest_space(trial: optuna.Trial, imbalance: str) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 4, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20, log=True),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
    }


def _isolation_forest_space(trial: optuna.Trial, imbalance: str) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_samples": trial.suggest_float("max_samples", 0.05, 1.0),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0)
    }


AUTOENCODER_SHAPES = {
    "16-4-16": (16, 4, 16),
    "20-8-20": (20, 8, 20),
    "24-12-24": (24, 12, 24),
    "28-14-7-14-28": (28, 14, 7, 14, 28)
}


def _autoencoder_space(trial: optuna.Trial, imbalance: str) -> dict:
    shape = trial.suggest_categorical("shape", sorted(AUTOENCODER_SHAPES))
    return {
        "hidden_layer_sizes": AUTOENCODER_SHAPES[shape],
        "noise_sigma": trial.suggest_float("noise_sigma", 0.0, 1.0),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "max_iter": 1000
    }


@dataclass(frozen=True)
class SearchSpace:
    arms: tuple[str, ...]
    params: Callable[[optuna.Trial, str], dict]
    n_trials: int


SPACES = {
    "logistic_regression": SearchSpace(
        ("weighted", "none", "under", "smote"), _logistic_regression_space, 20
    ),
    "random_forest": SearchSpace(
        ("weighted", "none", "under", "smote"), _random_forest_space, 20
    ),
    "lgbm": SearchSpace(
        ("none", "under", "smote"), _lgbm_space, 60
    ),
    "isolation_forest": SearchSpace(
        ("none",), _isolation_forest_space, 15
    ),
    "autoencoder": SearchSpace(
        ("none",), _autoencoder_space, 20
    )
}

DEFAULT_TUNED = tuple(SPACES)


def _suggest(trial: optuna.Trial, model_name: str) -> tuple[str, dict]:
    space = SPACES[model_name]
    imbalance = space.arms[0] if len(space.arms) == 1 else trial.suggest_categorical(
        "imbalance", list(space.arms)
    )
    return imbalance, space.params(trial, imbalance)


def _take(data, index):
    return data.iloc[index] if hasattr(data, "iloc") else np.asarray(data)[index]


def _cv_scores(
    X,
    y,
    cv,
    model_name: str,
    imbalance: str,
    params: dict,
    trial: optuna.Trial | None = None
) -> list[float]:
    scores = []
    for fold, (train, test) in enumerate(cv.split(X, y)):
        model = make_model(model_name, imbalance, **params).fit(_take(X, train), _take(y, train))
        scores.append(float(average_precision_score(_take(y, test), ranking(model, _take(X, test)))))

        if trial is not None:
            trial.report(float(np.mean(scores)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return scores


def _objective(trial: optuna.Trial, model_name: str, X, y, cv) -> float:
    imbalance, params = _suggest(trial, model_name)

    trial.set_user_attr("imbalance", imbalance)
    trial.set_user_attr("params", params)

    scores = _cv_scores(X, y, cv, model_name, imbalance, params, trial=trial)
    trial.set_user_attr("fold_scores", scores)
    return float(np.mean(scores))


def tune(
    model_name: str,
    X,
    y,
    n_trials: int | None = None,
    cv=None,
    storage: str | None = None,
    revalidate: bool = True,
    seed: int = SEED,
    verbose: bool = True
) -> TuningResult:
    if model_name not in SPACES:
        raise ValueError(f"No search space for '{model_name}'; known: {sorted(SPACES)}.")

    n_trials = SPACES[model_name].n_trials if n_trials is None else n_trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    search_cv = cv if cv is not None else StratifiedKFold(
        n_splits=SEARCH_FOLDS, shuffle=True, random_state=seed
    )

    callbacks = []
    if verbose:
        def _report(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if trial.state == optuna.trial.TrialState.PRUNED:
                print(f"  trial {trial.number:3} pruned after {len(trial.intermediate_values)} fold(s)")
            elif trial.value is not None:
                best = " *" if trial.number == study.best_trial.number else ""
                arm = trial.user_attrs.get("imbalance", "")
                print(f"  trial {trial.number:3} pr_auc={trial.value:.4f} {arm:8}{best}")

        callbacks.append(_report)

    study = optuna.create_study(
        study_name=f"{model_name}-pr-auc",
        storage=storage,
        load_if_exists=storage is not None,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2)
    )
    study.optimize(
        lambda trial: _objective(trial, model_name, X, y, search_cv),
        n_trials=n_trials,
        callbacks=callbacks
    )

    if not study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)):
        raise RuntimeError(f"No trial completed for '{model_name}' out of {n_trials}; nothing to select.")

    best = study.best_trial
    imbalance, params = best.user_attrs["imbalance"], best.user_attrs["params"]
    search_pr_auc = float(study.best_value)

    if revalidate:
        if verbose:
            print(f"\n  re-scoring the winner on the full repeated CV ({model_name} / {imbalance})")
        scores = _cv_scores(X, y, cross_validation_split(seed=seed), model_name, imbalance, params)
    else:
        scores = list(best.user_attrs.get("fold_scores", [search_pr_auc]))

    result = TuningResult(
        model=model_name,
        imbalance=imbalance,
        params=params,
        pr_auc_mean=float(np.mean(scores)),
        pr_auc_std=float(np.std(scores)),
        search_pr_auc=search_pr_auc,
        n_trials=len(study.trials),
        n_pruned=len(study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.PRUNED,))),
        fold_scores=scores,
        seed=seed
    )

    if verbose:
        source = "repeated CV" if revalidate else "search folds"
        print(f"\n=== tuned {result.model} / {result.imbalance} ===")
        print(f"  search pr_auc        {result.search_pr_auc:.4f}  (selected on, optimistic)")
        print(f"  {source} pr_auc {result.pr_auc_mean:.4f} +/- {result.pr_auc_std:.4f}")
        print(f"  trials               {result.n_trials} ({result.n_pruned} pruned)")
        for key, value in sorted(result.params.items()):
            print(f"    {key:20} {value}")

    return result


def tune_all(
    X,
    y,
    model_names: tuple[str, ...] = DEFAULT_TUNED,
    n_trials: dict[str, int] | None = None,
    verbose: bool = True,
    **kwargs
) -> dict[str, TuningResult]:
    budgets = n_trials or {}

    results = {}
    for model_name in model_names:
        if verbose:
            print(f"\n### tuning {model_name} "
                  f"({budgets.get(model_name, SPACES[model_name].n_trials)} trials) ###")
        results[model_name] = tune(
            model_name, X, y, n_trials=budgets.get(model_name), verbose=verbose, **kwargs
        )

    if verbose and len(results) > 1:
        print("\n=== tuned leaderboard (repeated CV) ===")
        ranked = sorted(results.values(), key=lambda r: r.pr_auc_mean, reverse=True)
        for result in ranked:
            print(f"  {result.model:20} {result.imbalance:8} "
                  f"pr_auc={result.pr_auc_mean:.4f} +/- {result.pr_auc_std:.4f}")

    return results


def tuned_model(result: TuningResult):
    return make_model(result.model, result.imbalance, **result.params)


def result_path(model_name: str) -> Path:
    return Path(REPORTS_DIR) / f"tuning_{model_name}.json"


def save(result: TuningResult, path: os.PathLike | str | None = None) -> Path:
    path = Path(path) if path is not None else result_path(result.model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return path


def load(model_name: str, path: os.PathLike | str | None = None) -> TuningResult:
    path = Path(path) if path is not None else result_path(model_name)
    return TuningResult(**json.loads(path.read_text()))
