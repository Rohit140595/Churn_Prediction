from pathlib import Path

import pandas as pd

from src.config import DROP_COLS, RAW_DATA_PATH


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and minimally clean the raw churn dataset.

    Reads the CSV and drops identifier columns that carry no predictive signal.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file, by default ``RAW_DATA_PATH``.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``RowNumber``, ``CustomerId``, and ``Surname`` removed.
    """
    df = pd.read_csv(path)
    return df.drop(columns=DROP_COLS)
