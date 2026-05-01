"""Utilities for persisting and loading the churn prediction inference artifact.

The artifact bundles everything needed to serve predictions into a single file:
preprocessor, optional feature selector, fitted model, and metadata.  Storing
them together prevents version skew between components — if the preprocessor
and model are saved separately they can silently diverge between retraining runs.

Typical usage
-------------
After training::

    from src.serving.model_store import build_artifact, save_artifact

    result = run_pipeline(model_name="xgboost", ...)
    artifact = build_artifact(result, model_name="xgboost")
    path = save_artifact(artifact)
    print(f"Artifact saved to {path}")

At serving time::

    from src.serving.model_store import load_artifact

    artifact = load_artifact()
    preprocessor = artifact["preprocessor"]
    model        = artifact["model"]
    selector     = artifact["selector"]   # may be None
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib

from src.config import MODEL_ARTIFACT_PATH


def build_artifact(
    pipeline_result: dict[str, Any],
    model_name: str,
    version: str = "0.4.0",
) -> dict[str, Any]:
    """Assemble a serialisable inference artifact from a pipeline result dict.

    Parameters
    ----------
    pipeline_result : dict[str, Any]
        The dict returned by :func:`src.pipeline.run_pipeline`.
    model_name : str
        Registry key of the trained model (e.g. ``"xgboost"``).
    version : str, optional
        Semantic version embedded in the artifact, by default ``"0.4.0"``.

    Returns
    -------
    dict[str, Any]
        Keys:

        - ``model_name``  : str
        - ``version``     : str
        - ``trained_at``  : str — ISO-8601 UTC timestamp
        - ``model``       : fitted :class:`src.models.base.BaseModel` wrapper
        - ``preprocessor``: fitted ``ColumnTransformer``
        - ``selector``    : fitted selector or ``None``
        - ``test_metrics``: dict[str, float] — held-out evaluation metrics
    """
    return {
        "model_name": model_name,
        "version": version,
        # UTC timestamp so the value is unambiguous regardless of server timezone.
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model": pipeline_result["model"],
        "preprocessor": pipeline_result["preprocessor"],
        "selector": pipeline_result["selector"],
        "test_metrics": pipeline_result["metrics"],
    }


def save_artifact(
    artifact: dict[str, Any],
    path: Optional[Path] = None,
) -> Path:
    """Serialise the inference artifact to disk using joblib.

    Creates the parent directory automatically if it does not exist.

    Parameters
    ----------
    artifact : dict[str, Any]
        Artifact produced by :func:`build_artifact`.
    path : Path, optional
        Destination file path.  Defaults to ``MODEL_ARTIFACT_PATH`` from
        ``src.config``.

    Returns
    -------
    Path
        Absolute path of the saved file.
    """
    dest = Path(path) if path else MODEL_ARTIFACT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, dest)
    return dest.resolve()


def load_artifact(path: Optional[Path] = None) -> dict[str, Any]:
    """Load a previously saved inference artifact from disk.

    Parameters
    ----------
    path : Path, optional
        Source file path.  Defaults to ``MODEL_ARTIFACT_PATH`` from
        ``src.config``.

    Returns
    -------
    dict[str, Any]
        Artifact dict as produced by :func:`build_artifact`.

    Raises
    ------
    FileNotFoundError
        If no artifact file exists at ``path``.
    """
    src_path = Path(path) if path else MODEL_ARTIFACT_PATH
    if not src_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{src_path}'. "
            "Train and save a model first:\n"
            "  python scripts/train.py --model xgboost --save-model"
        )
    return joblib.load(src_path)
