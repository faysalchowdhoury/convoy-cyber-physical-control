"""Write results tables to CSV + a combined metrics.json."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: str | Path, index: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def write_metrics_json(metrics: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
