from __future__ import annotations
import os
import pandas as pd

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def read_csv_ts(path: str, ts_col: str = "ts") -> pd.DataFrame:
    df = pd.read_csv(path)
    if ts_col not in df.columns:
        raise ValueError(f"Missing '{ts_col}' column in {path}")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    if df[ts_col].isna().any():
        bad = df[df[ts_col].isna()].head(5)
        raise ValueError(f"Unparseable timestamps in {path}. Examples:\n{bad}")
    df = df.sort_values(ts_col).reset_index(drop=True)
    return df

def relpath(from_dir: str, to_path: str) -> str:
    return os.path.relpath(to_path, from_dir).replace(os.sep, "/")

def utc_now_stamp() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
