import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .config import ALERT_BUDGET, ALPHA, DATA_PATH, MIN_PRECISION, N_COMPARE, OBJECTIVE, \
    OBJECTIVES, REPORTS_DIR, REVIEW_COST, SHAP_FEATURES, SHAP_SAMPLE, Settings
from .common import SEED
from .tune import DEFAULT_TUNED

COMMANDS = ("run", "tune")


def _model_list(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in names if name not in DEFAULT_TUNED]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown model(s) {', '.join(unknown)}; known: {', '.join(DEFAULT_TUNED)}"
        )
    return names


def _shared() -> argparse.ArgumentParser:
    # NOTE: added as a parent to both subcommands, so these show up in both helps
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=Path, default=DATA_PATH,
                        help=f"the transactions CSV (default: {DATA_PATH})")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR,
                        help=f"where figures, tuning caches and the summary go (default: {REPORTS_DIR})")
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"seed for splits, the search and the estimators (default: {SEED})")
    parser.add_argument("--models", type=_model_list, default=None, metavar="A,B",
                        help=f"which models to tune (default: {','.join(DEFAULT_TUNED)})")
    parser.add_argument("--trials", type=int, default=None,
                        help="Optuna trials per model (default: the per-model budget in tune.py)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="only errors; the summary and figures are still written")
    return parser


def parser() -> argparse.ArgumentParser:
    shared = _shared()
    root = argparse.ArgumentParser(
        prog="needle",
        description="Credit card fraud detection: tune, select, threshold, and report.",
        epilog="With no subcommand, 'run' is assumed."
    )
    try:
        root.add_argument("--version", action="version", version=f"needle {version('needle')}")
    except PackageNotFoundError:  # NOTE: a source tree that was never installed
        pass

    commands = root.add_subparsers(dest="command", metavar="{run,tune}")

    run = commands.add_parser("run", parents=[shared], help="the full pipeline (default)",
                              description="Tune, select a model, pick a threshold, score day 2, "
                                          "and write the figures and summary.")
    run.add_argument("--objective", choices=OBJECTIVES, default=OBJECTIVE,
                     help=f"which operating point to ship (default: {OBJECTIVE})")
    run.add_argument("--retune", action="store_true",
                     help="ignore the cached tuning results and search again")
    run.add_argument("--min-precision", type=float, default=MIN_PRECISION, metavar="P",
                     help=f"precision floor for recall@p and the 'precision' objective "
                          f"(default: {MIN_PRECISION})")
    run.add_argument("--review-cost", type=float, default=REVIEW_COST, metavar="C",
                     help=f"what one manual review costs, against a missed fraud's amount "
                          f"(default: {REVIEW_COST:g})")
    run.add_argument("--alert-budget", type=int, default=ALERT_BUDGET, metavar="N",
                     help=f"reviews affordable per day, for the 'budget' objective "
                          f"(default: {ALERT_BUDGET})")
    run.add_argument("--shap-sample", type=int, default=SHAP_SAMPLE, metavar="N",
                     help=f"legitimate transactions to explain; every fraud is kept regardless "
                          f"(default: {SHAP_SAMPLE})")
    run.add_argument("--shap-features", type=int, default=SHAP_FEATURES, metavar="N",
                     help=f"features in the beeswarm and ranking table (default: {SHAP_FEATURES})")
    run.add_argument("--alpha", type=float, default=ALPHA, metavar="A",
                     help=f"significance level for the paired model comparison "
                          f"(default: {ALPHA:g})")
    run.add_argument("--compare", dest="n_compare", type=int, default=N_COMPARE, metavar="N",
                     help=f"leaderboard rows to test the winner against (default: {N_COMPARE})")
    run.add_argument("--no-tests", dest="tests", action="store_false",
                     help="skip the paired comparison of the leaderboard's top candidates")
    run.add_argument("--no-plots", dest="plots", action="store_false", help="skip the figures")
    run.add_argument("--no-report", dest="report", action="store_false",
                     help="skip writing summary.md")

    tune = commands.add_parser("tune", parents=[shared], help="the hyperparameter search only",
                               description="Search hyperparameters on day 1 and cache the results "
                                           "under the reports directory, without running the "
                                           "rest of the pipeline.")
    tune.add_argument("--keep-cached", dest="retune", action="store_false",
                      help="reuse any tuning result already on disk instead of searching again")
    tune.add_argument("--no-revalidate", dest="revalidate", action="store_false",
                      help="skip re-scoring the winner on the full repeated CV (faster, "
                           "and the reported PR-AUC stays the optimistic search value)")

    return root


def _settings(arguments: argparse.Namespace) -> Settings:
    fields = {
        name: value for name, value in vars(arguments).items()
        if name in Settings.__dataclass_fields__
    }
    return Settings(**fields)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # NOTE: 'needle' and 'needle --retune' both mean 'needle run ...'
    if not argv or (argv[0] not in COMMANDS and argv[0].startswith("-")
                    and argv[0] not in ("-h", "--help", "--version")):
        argv.insert(0, "run")

    arguments = parser().parse_args(argv)
    settings = _settings(arguments)

    try:
        # NOTE: imported here and not at module scope so --help and --version do not
        # pay for lightgbm, shap and matplotlib
        if arguments.command == "tune":
            from .pipeline import search
            search(settings, revalidate=arguments.revalidate)
        else:
            from .pipeline import run
            run(settings)
    except FileNotFoundError as error:
        print(f"needle: {error.filename or error}: no such file", file=sys.stderr)
        if Path(error.filename or "").name == DATA_PATH.name:
            print("       download creditcard.csv from Kaggle, or pass --data", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nneedle: interrupted", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
