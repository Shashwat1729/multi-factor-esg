from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QualityReport:
    rows: int
    cols: int
    missing_by_col: dict[str, float]
    overall_missing: float
    outlier_counts_by_col: dict[str, int]


def missingness_report(df: pd.DataFrame) -> QualityReport:
    """Compute missingness and basic outlier counts for numeric columns.

    Outliers are defined via a simple z-score rule (|z| > 4). This is a *screening*
    diagnostic (not a final statistical test).
    """
    if df.empty:
        return QualityReport(
            rows=0,
            cols=0,
            missing_by_col={},
            overall_missing=float("nan"),
            outlier_counts_by_col={},
        )

    missing_by_col = (df.isna().mean()).to_dict()
    overall_missing = float(df.isna().mean().mean())

    outlier_counts: dict[str, int] = {}
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        s = df[c]
        mu = s.mean(skipna=True)
        sig = s.std(skipna=True)
        if pd.isna(sig) or sig == 0:
            outlier_counts[str(c)] = 0
            continue
        z = (s - mu) / sig
        outlier_counts[str(c)] = int((z.abs() > 4).sum(skipna=True))

    return QualityReport(
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        missing_by_col={str(k): float(v) for k, v in missing_by_col.items()},
        overall_missing=overall_missing,
        outlier_counts_by_col=outlier_counts,
    )


def enforce_min_coverage(
    df: pd.DataFrame,
    *,
    required_cols: Iterable[str],
    min_coverage: float,
    id_col: str = "ticker",
) -> pd.DataFrame:
    """Filter entities failing indicator coverage threshold.

    Coverage = non-missing required indicators / number of required indicators.
    """
    required_cols = list(required_cols)
    if not required_cols:
        return df
    present = [c for c in required_cols if c in df.columns]
    if not present:
        return df

    coverage = 1.0 - df[present].isna().mean(axis=1)
    keep = coverage >= float(min_coverage)
    out = df.loc[keep].copy()
    if id_col in out.columns:
        out[id_col] = out[id_col].astype(str)
    return out


def group_median_impute(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    numeric_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Impute numeric columns using within-group medians."""
    out = df.copy()
    if numeric_cols is None:
        numeric_cols = list(out.select_dtypes(include=[np.number]).columns)
    for c in numeric_cols:
        if c not in out.columns:
            continue
        if not group_cols:
            out[c] = out[c].fillna(out[c].median(skipna=True))
        else:
            out[c] = out[c].fillna(out.groupby(group_cols)[c].transform("median"))
            out[c] = out[c].fillna(out[c].median(skipna=True))
    return out

