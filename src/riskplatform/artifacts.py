"""Read/write helpers for committed data artifacts.

All tabular artifacts are parquet with natural keys; `upsert` is the single write
path so every writer gets the same idempotence guarantee: re-running a day (or the
7-day ingestion overlap window) overwrites rows in place rather than duplicating them.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

PRICES = "prices.parquet"
RETURNS = "returns.parquet"
RISK_METRICS = "risk_metrics.parquet"
BACKTEST = "backtest_results.parquet"
ALERTS = "alerts_history.parquet"
MONITOR_STATUS = "monitor_status.json"


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def upsert(path: Path, new_rows: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Merge new_rows into the parquet file at path, keyed on `keys` (last write wins)."""
    if new_rows.empty and not path.exists():
        return new_rows
    old = pd.read_parquet(path) if path.exists() else new_rows.iloc[0:0]
    out = (
        pd.concat([old, new_rows], ignore_index=True)
        .drop_duplicates(subset=keys, keep="last")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    _atomic_write_parquet(out, path)
    return out


def overwrite(path: Path, df: pd.DataFrame) -> None:
    """Full rewrite for artifacts rebuilt from scratch each run (returns, backtest)."""
    _atomic_write_parquet(df, path)


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_parquet(path) if path.exists() else None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False
    ) as tmp:
        json.dump(payload, tmp, indent=2, default=str)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None
