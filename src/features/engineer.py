import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the churn dataset.

    Creates two new columns:

    - ``BalancePerProduct``: customer balance divided by number of products
      (denominator floored at 1 to avoid division by zero).
    - ``ActiveProducts``: interaction term — ``IsActiveMember * NumOfProducts``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or minimally cleaned DataFrame. Must contain columns:
        ``Balance``, ``NumOfProducts``, ``IsActiveMember``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with added feature columns. Input is not modified.
    """
    df = df.copy()
    df["BalancePerProduct"] = df["Balance"] / df["NumOfProducts"].clip(lower=1)
    df["ActiveProducts"] = df["IsActiveMember"] * df["NumOfProducts"]
    return df
