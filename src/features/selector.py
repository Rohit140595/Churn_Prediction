from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif

from src.config import RANDOM_STATE

STRATEGIES = ("importance", "kbest")


def build_selector(
    strategy: str = "importance",
    k: int = 10,
    random_state: int = RANDOM_STATE,
) -> Any:
    """Build an unfitted sklearn feature selector.

    Two strategies are supported:

    - ``"importance"``: ``SelectFromModel`` backed by a ``RandomForestClassifier``.
      Keeps features whose importance exceeds the mean importance across all features.
      The number of selected features is determined by the data.
    - ``"kbest"``: ``SelectKBest`` using ANOVA F-scores. Keeps the top ``k`` features.

    The returned object is a standard sklearn transformer with ``fit``,
    ``transform``, and ``get_support`` methods, so it can be dropped into
    an sklearn ``Pipeline``.

    Parameters
    ----------
    strategy : str, optional
        One of ``"importance"`` or ``"kbest"``, by default ``"importance"``.
    k : int, optional
        Number of features to keep when ``strategy="kbest"``, by default 10.
    random_state : int, optional
        Random seed passed to the internal ``RandomForestClassifier`` when
        ``strategy="importance"``.

    Returns
    -------
    SelectFromModel or SelectKBest
        Unfitted sklearn feature selector.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of the supported values.
    """
    if strategy == "importance":
        return SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=random_state),
            threshold="mean",
        )
    if strategy == "kbest":
        return SelectKBest(f_classif, k=k)
    raise ValueError(f"Unknown strategy '{strategy}'. Choose one of {STRATEGIES}.")
