from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_optional_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    dataframe = pd.read_csv(csv_path)
    return dataframe.dropna(how="all")
