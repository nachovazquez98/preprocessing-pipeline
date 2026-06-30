import os

import pandas as pd


def read_input_data(path: str) -> pd.DataFrame:
    if not path:
        raise ValueError("No input data path was provided.")
    if path.endswith(".csv"):
        return pd.read_csv(path)
    if path.endswith(".parquet"):
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError("Reading parquet requires pyarrow or fastparquet.") from exc
    if path.endswith(".pkl") or path.endswith(".pickle"):
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported input format for path '{path}'. Use csv, parquet, or pickle.")


def write_output_data(df: pd.DataFrame, path: str):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if path.endswith(".csv"):
        df.to_csv(path, index=False)
        return
    if path.endswith(".parquet"):
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            raise ImportError("Writing parquet requires pyarrow or fastparquet.") from exc
        return
    if path.endswith(".pkl") or path.endswith(".pickle"):
        df.to_pickle(path)
        return
    raise ValueError(f"Unsupported output format for path '{path}'. Use csv, parquet, or pickle.")
