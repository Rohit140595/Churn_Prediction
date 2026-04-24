"""CLI entry point for training a churn prediction model.

Usage
-----
From the project root::

    python scripts/train.py --model xgboost
    python scripts/train.py --model random_forest --experiment-name my_exp
    python scripts/train.py --model logistic --no-tracking
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
        Parsed arguments with ``model``, ``experiment_name``, and ``no_tracking``.
    """
    parser = argparse.ArgumentParser(description="Train a churn prediction model.")
    parser.add_argument(
        "--model",
        choices=list(DEFAULT_PARAMS.keys()),
        default="xgboost",
        help="Model architecture to train (default: xgboost).",
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
    """Run training and print metrics to stdout."""
    args = parse_args()

    print(f"Training model: {args.model}")
    result = run_pipeline(
        model_name=args.model,
        experiment_name=args.experiment_name,
        track=not args.no_tracking,
    )

    print("\nCross-validation (5-fold, mean ± std):")
    for name, scores in result["cv_results"].items():
        print(f"  {name:12s}: {scores.mean():.4f} ± {scores.std():.4f}")

    print("\nTest-set metrics:")
    for name, value in result["metrics"].items():
        print(f"  {name:12s}: {value:.4f}")

    if result["run_id"]:
        print(f"\nMLflow run ID : {result['run_id']}")
        print("View results  : mlflow ui")


if __name__ == "__main__":
    main()
