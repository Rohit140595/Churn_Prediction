"""CLI entry point for training a churn prediction model.

Usage
-----
From the project root::

    python scripts/train.py --model xgboost
    python scripts/train.py --model xgboost --tune --n-trials 50
    python scripts/train.py --model xgboost --feature-selection importance
    python scripts/train.py --model xgboost --experiment-name my_exp --no-tracking
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_PARAMS
from src.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train a churn prediction model.")
    parser.add_argument(
        "--model",
        choices=list(DEFAULT_PARAMS.keys()),
        default="xgboost",
        help="Model architecture to train (default: xgboost).",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter search before fitting.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials when --tune is set (default: 50).",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["importance", "kbest"],
        default=None,
        help="Feature selection strategy (default: none).",
    )
    parser.add_argument(
        "--feature-selection-k",
        type=int,
        default=10,
        help="Features to keep when --feature-selection=kbest (default: 10).",
    )
    parser.add_argument(
        "--experiment-name",
        default="churn_prediction",
        help="MLflow experiment name (default: churn_prediction).",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable MLflow tracking.",
    )
    return parser.parse_args()


def main() -> None:
    """Run training and print results to stdout."""
    args = parse_args()

    print(f"Model           : {args.model}")
    if args.tune:
        print(f"Tuning          : Optuna ({args.n_trials} trials, 3-fold CV)")
    if args.feature_selection:
        label = args.feature_selection
        if args.feature_selection == "kbest":
            label += f" (k={args.feature_selection_k})"
        print(f"Feature selection: {label}")

    result = run_pipeline(
        model_name=args.model,
        feature_selection=args.feature_selection,
        feature_selection_k=args.feature_selection_k,
        tune=args.tune,
        n_trials=args.n_trials,
        experiment_name=args.experiment_name,
        track=not args.no_tracking,
    )

    if args.tune:
        print(f"\nBest CV ROC-AUC : {result['best_cv_score']:.4f}")
        print("Best params     :")
        for k, v in result["best_params"].items():
            print(f"  {k}: {v}")

    if result["n_features_selected"] is not None:
        print(f"\nFeatures selected: {result['n_features_selected']}")

    print("\nTest-set metrics:")
    for name, value in result["metrics"].items():
        print(f"  {name:12s}: {value:.4f}")

    rd = result["roi_default"]
    ro = result["roi_optimal"]
    print("\nCampaign ROI (assumptions: "
          f"${rd['outreach_cost'] / max(rd['tp'] + rd['fp'], 1):.0f}/outreach, "
          f"$800 revenue/retained, 30% success rate):")
    print(f"  threshold=0.50 : {rd['campaign_roi_pct']:+.1f}%  "
          f"(contacted {int(rd['tp'] + rd['fp'])}, "
          f"net ${rd['net_value'] - rd['outreach_cost']:,.0f})")
    print(f"  optimal={ro['threshold']:.2f}   : {ro['campaign_roi_pct']:+.1f}%  "
          f"(contacted {int(ro['tp'] + ro['fp'])}, "
          f"net ${ro['net_value'] - ro['outreach_cost']:,.0f})")

    if result["run_id"]:
        print(f"\nMLflow run ID : {result['run_id']}")
        print("View results  : mlflow ui")


if __name__ == "__main__":
    main()
