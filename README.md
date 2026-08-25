# Needle

Credit card fraud detection on a 0.173% positive rate: tune, select a model, pick an operating
point, and score it once on a held-out day. The pipeline ranks on PR-AUC rather than ROC-AUC,
chooses its threshold on day-1 out-of-fold scores, and writes a report that quotes the day-2
numbers. See [PLAN.md](PLAN.md) for why those choices are made.

## Setup

```sh
devenv shell # NOTE: Only for devenv users
uv sync
```

Then put the dataset at `dataset/creditcard.csv`, download `creditcard.csv` from the Kaggle linka
at the bottom, or point `--data` somewhere else.

## Usage

```sh
uv run needle
uv run needle run --help
uv run needle tune --help
```

`needle` with no subcommand means `needle run`.

### `needle run`

Searches hyperparameters on day 1, cross-validates every candidate, picks a threshold from
out-of-fold scores, refits the winner and scores it on day 2, then writes the figures and
`summary.md`.

```sh
uv run needle run --objective precision      # ship the precision-floor point, not the cost one
uv run needle run --review-cost 7.5          # a manual review costs 7.50, not 3.00
uv run needle run --retune --models lgbm     # ignore the cache and search lgbm again
uv run needle run --no-plots --no-report     # numbers on stdout only
```

| Flag | Default | What it does |
|---|---|---|
| `--data` | `dataset/creditcard.csv` | the transactions CSV |
| `--reports-dir` | `reports` | where figures, tuning caches and the summary go |
| `--objective` | `cost` | which operating point ships: `cost`, `precision` or `budget` |
| `--min-precision` | `0.9` | precision floor for recall@p and the `precision` objective |
| `--review-cost` | `3.0` | what one manual review costs, against a missed fraud's amount |
| `--alert-budget` | `100` | reviews affordable per day, for the `budget` objective |
| `--models` | all five | which models to tune, comma separated |
| `--trials` | per-model | Optuna trials per model |
| `--retune` | off | ignore the cached tuning results and search again |
| `--shap-sample` | `4000` | legitimate transactions to explain; every fraud is kept regardless |
| `--shap-features` | `15` | features in the beeswarm and the ranking table |
| `--seed` | `42` | splits, the search, and the estimators |
| `--alpha` | `0.05` | significance level for the paired comparison of the top candidates |
| `--compare` | `6` | leaderboard rows to test the winner against |
| `--no-plots`, `--no-report`, `--no-tests` | on | skip the figures / `summary.md` / the paired test |
| `-q`, `--quiet` | off | only errors; the files are still written |

The three economic flags change what the run *decides*, not just what it prints: `--review-cost`
moves the cost-optimal threshold, and the report quotes the values the run actually used.

### `needle tune`

The hyperparameter search on its own, day 1 only. Results are cached as
`reports/tuning_<model>.json` and reused by `needle run` until `--retune`.

```sh
uv run needle tune --models lgbm --trials 60
uv run needle tune --no-revalidate           # skip re-scoring the winner on the full repeated CV
uv run needle tune --keep-cached             # reuse anything already on disk
```

The search is the slow part of a cold run. Tune once, then iterate on thresholds and reports
against the cache.

## What a run writes

Into `--reports-dir` (default `reports/`):

| File | |
|---|---|
| `summary.md` | the report: metric choice, splits, whether the leaderboard order is real, SHAP ranking, operating point, misses, limitations |
| `precision_recall.png` | day-1 out-of-fold against day-2 holdout, with the shipped point marked |
| `cost_vs_alerts.png` | amount-weighted cost against queue size |
| `missed_frauds.png` | the frauds that got through, by amount and score |
| `shap_beeswarm.png` | what the winning model uses |
| `reliability.png` | raw against sigmoid-calibrated probabilities |
| `tuning_<model>.json` | the cached search result, with its seed and fold scores |

Figures are only written for what applies: no beeswarm when the winner has no SHAP explainer,
no reliability curve when it emits no probabilities.

# Materials

* [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
