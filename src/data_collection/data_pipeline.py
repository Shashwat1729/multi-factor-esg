from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from .data_quality import enforce_min_coverage, group_median_impute, missingness_report


@dataclass(frozen=True)
class PipelinePaths:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    external_dir: Path = Path("data/external")


def load_yaml(path: str | os.PathLike[str]) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs(paths: PipelinePaths) -> None:
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)
    paths.external_dir.mkdir(parents=True, exist_ok=True)


def _try_import_yfinance():
    try:
        import yfinance as yf  # type: ignore

        return yf
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "yfinance is required for Yahoo Finance ingestion. Install with `pip install yfinance`."
        ) from e


def fetch_yahoo_financials(
    tickers: list[str],
    *,
    data_sources_cfg: dict[str, Any],
    paths: PipelinePaths = PipelinePaths(),
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch a standardized set of financial fields for a ticker list (public source).

    Output columns (subset depending on availability):
      - ticker, market_cap, total_revenue, ebitda, net_income, roa, roe,
        debt_to_equity, free_cashflow, trailing_pe, price_to_book
    """
    ensure_dirs(paths)
    enabled = (
        data_sources_cfg.get("data_sources", {})
        .get("public", {})
        .get("financial", {})
        .get("yahoo_finance", {})
        .get("enabled", False)
    )
    if not enabled:
        raise RuntimeError("Yahoo Finance source is disabled in config/data_sources.yaml")

    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        return pd.DataFrame(columns=["ticker"])

    cache_path = paths.raw_dir / "yahoo_financials.json"
    cache_obj: dict[str, Any] = {}
    if cache and cache_path.exists():
        cache_obj = json.loads(cache_path.read_text(encoding="utf-8") or "{}")

    yf = _try_import_yfinance()

    rows: list[dict[str, Any]] = []
    updated = False
    for t in tickers:
        if cache and t in cache_obj:
            info = cache_obj[t]
        else:
            info = yf.Ticker(t).info
            cache_obj[t] = info
            updated = True

        rows.append(
            {
                "ticker": t,
                "market_cap": info.get("marketCap"),
                "total_revenue": info.get("totalRevenue"),
                "ebitda": info.get("ebitda"),
                "net_income": info.get("netIncomeToCommon"),
                "roa": info.get("returnOnAssets"),
                "roe": info.get("returnOnEquity"),
                "debt_to_equity": info.get("debtToEquity"),
                "free_cashflow": info.get("freeCashflow"),
                "trailing_pe": info.get("trailingPE"),
                "price_to_book": info.get("priceToBook"),
                "currency": info.get("currency"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "country": info.get("country"),
            }
        )

    if cache and updated:
        cache_path.write_text(json.dumps(cache_obj, indent=2), encoding="utf-8")

    return pd.DataFrame(rows)


def _sec_headers(data_sources_cfg: dict[str, Any]) -> dict[str, str]:
    ua_env = (
        data_sources_cfg.get("data_sources", {})
        .get("public", {})
        .get("filings", {})
        .get("sec_edgar", {})
        .get("user_agent_env", "SEC_EDGAR_USER_AGENT")
    )
    ua = os.getenv(ua_env)
    if not ua:
        raise RuntimeError(
            f"SEC EDGAR requires a user-agent. Set env var {ua_env} to something like "
            "'YourName your.email@domain.com'."
        )
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}


def fetch_sec_company_facts(
    cik: str,
    *,
    data_sources_cfg: dict[str, Any],
    paths: PipelinePaths = PipelinePaths(),
    cache: bool = True,
) -> dict[str, Any]:
    """Fetch SEC companyfacts JSON for a given CIK (public source)."""
    ensure_dirs(paths)
    enabled = (
        data_sources_cfg.get("data_sources", {})
        .get("public", {})
        .get("filings", {})
        .get("sec_edgar", {})
        .get("enabled", False)
    )
    if not enabled:
        raise RuntimeError("SEC EDGAR source is disabled in config/data_sources.yaml")

    cik_num = str(cik).strip()
    cik_padded = cik_num.zfill(10)

    cache_path = paths.raw_dir / f"sec_companyfacts_{cik_padded}.json"
    if cache and cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8") or "{}")

    import requests  # local import: optional dependency for pure notebook use

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json"
    r = requests.get(url, headers=_sec_headers(data_sources_cfg), timeout=60)
    r.raise_for_status()
    obj = r.json()
    if cache:
        cache_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return obj


def standardize_and_clean(
    df: pd.DataFrame,
    *,
    index_cfg: dict[str, Any],
    required_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply basic missing-data rules from config and return (df, diagnostics)."""
    required_cols = required_cols or []

    esg_cfg = index_cfg.get("esg_index", {})
    missing_cfg = esg_cfg.get("missing_data", {})
    min_cov = float(missing_cfg.get("min_indicator_coverage", 0.60))
    impute_cfg = missing_cfg.get("imputation", {})
    impute_method = impute_cfg.get("method", "group_median")
    group_keys = list(impute_cfg.get("group_keys", ["sector"]))

    diag: dict[str, Any] = {}
    diag["missingness_before"] = missingness_report(df).__dict__

    if required_cols:
        df = enforce_min_coverage(df, required_cols=required_cols, min_coverage=min_cov, id_col="ticker")

    if impute_method == "group_median":
        df = group_median_impute(df, group_cols=group_keys)
    elif impute_method == "none":
        pass
    else:
        # Placeholder: KNN/iterative can be added once a stable feature matrix is defined
        df = group_median_impute(df, group_cols=group_keys)

    diag["missingness_after"] = missingness_report(df).__dict__
    return df, diag


def load_configs(
    *,
    index_config_path: str | os.PathLike[str] | None = None,
    data_sources_path: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load index and data source configurations.
    
    If paths are not provided, searches for config files relative to project root.
    """
    if index_config_path is None:
        # Try to find project root (look for config/ directory)
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            config_dir = parent / "config"
            if config_dir.exists() and (config_dir / "index_config.yaml").exists():
                index_config_path = config_dir / "index_config.yaml"
                break
        else:
            # Fallback: try relative to this file's location
            this_file = Path(__file__)
            project_root = this_file.parent.parent.parent
            config_dir = project_root / "config"
            if config_dir.exists() and (config_dir / "index_config.yaml").exists():
                index_config_path = config_dir / "index_config.yaml"
            else:
                # Last fallback to relative path
                index_config_path = "config/index_config.yaml"
    
    if data_sources_path is None:
        # Use same project root logic
        if isinstance(index_config_path, Path):
            data_sources_path = index_config_path.parent / "data_sources.yaml"
        else:
            # Try to find project root
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                config_dir = parent / "config"
                if config_dir.exists() and (config_dir / "data_sources.yaml").exists():
                    data_sources_path = config_dir / "data_sources.yaml"
                    break
            else:
                # Fallback: try relative to this file's location
                this_file = Path(__file__)
                project_root = this_file.parent.parent.parent
                config_dir = project_root / "config"
                if config_dir.exists() and (config_dir / "data_sources.yaml").exists():
                    data_sources_path = config_dir / "data_sources.yaml"
                else:
                    data_sources_path = "config/data_sources.yaml"
    
    return load_yaml(index_config_path), load_yaml(data_sources_path)

