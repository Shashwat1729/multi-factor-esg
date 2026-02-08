"""Composite ESG Index Construction.

Implements three-level framework: Scope → Measurement → Weighting & Aggregation.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def normalize_indicators(
    df: pd.DataFrame,
    indicator_cols: list[str],
    *,
    method: str = "zscore",
    by_group: dict[str, bool] | None = None,
    winsorize: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Normalize ESG indicators using specified method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with indicator columns
    indicator_cols : list[str]
        Column names to normalize
    method : str, default "zscore"
        Normalization method: "zscore", "minmax", "percentile", "robust_zscore"
    by_group : dict[str, bool] | None, optional
        Group-by keys for within-group normalization (e.g., {"sector": True})
    winsorize : dict[str, Any] | None, optional
        Winsorization config: {"enabled": bool, "lower_quantile": float, "upper_quantile": float}

    Returns
    -------
    pd.DataFrame
        DataFrame with normalized indicator columns (suffix "_norm")
    """
    by_group = by_group or {}
    winsorize = winsorize or {}
    df = df.copy()

    # Filter to only numeric columns
    numeric_cols = []
    for col in indicator_cols:
        if col not in df.columns:
            warnings.warn(f"Indicator column '{col}' not found, skipping", UserWarning)
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            warnings.warn(f"Indicator column '{col}' is not numeric, skipping", UserWarning)
            continue
        numeric_cols.append(col)
    
    if not numeric_cols:
        warnings.warn("No numeric indicator columns found for normalization", UserWarning)
        return df

    # Winsorize if enabled
    if winsorize.get("enabled", False):
        lower_q = winsorize.get("lower_quantile", 0.01)
        upper_q = winsorize.get("upper_quantile", 0.99)
        for col in numeric_cols:
            if col in df.columns:
                lower = df[col].quantile(lower_q)
                upper = df[col].quantile(upper_q)
                df[col] = df[col].clip(lower=lower, upper=upper)

    # Apply normalization
    norm_cols = []
    for col in numeric_cols:

        if method == "zscore":
            if by_group.get("sector", False) and "sector" in df.columns:
                df[f"{col}_norm"] = df.groupby("sector")[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-10)
                )
            else:
                df[f"{col}_norm"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-10)
        elif method == "minmax":
            if by_group.get("sector", False) and "sector" in df.columns:
                df[f"{col}_norm"] = df.groupby("sector")[col].transform(
                    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10)
                )
            else:
                df[f"{col}_norm"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-10)
        elif method == "percentile":
            if by_group.get("sector", False) and "sector" in df.columns:
                df[f"{col}_norm"] = df.groupby("sector")[col].transform(
                    lambda x: x.rank(pct=True) * 100
                )
            else:
                df[f"{col}_norm"] = df[col].rank(pct=True) * 100
        elif method == "robust_zscore":
            if by_group.get("sector", False) and "sector" in df.columns:
                df[f"{col}_norm"] = df.groupby("sector")[col].transform(
                    lambda x: (x - x.median()) / (x.mad() + 1e-10)
                )
            else:
                df[f"{col}_norm"] = (df[col] - df[col].median()) / (df[col].mad() + 1e-10)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        norm_cols.append(f"{col}_norm")

    return df


def compute_pillar_scores(
    df: pd.DataFrame,
    pillar_config: dict[str, dict[str, float]],
    *,
    pillar_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute ESG pillar scores (E, S, G) from normalized indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with normalized indicator columns (suffix "_norm")
    pillar_config : dict[str, dict[str, float]]
        Category weights within each pillar, e.g.:
        {
            "E": {"emissions": 0.40, "energy": 0.25, "water": 0.20, "waste": 0.15},
            "S": {"labor": 0.35, "diversity": 0.30, "health_safety": 0.25, "community": 0.10},
            "G": {"board": 0.35, "comp": 0.25, "shareholder_rights": 0.20, "ethics": 0.20}
        }
    pillar_weights : dict[str, float] | None, optional
        Final pillar weights for composite score (default: equal weights)

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: "E_score", "S_score", "G_score", "ESG_composite"
    """
    df = df.copy()
    pillar_weights = pillar_weights or {"E": 1/3, "S": 1/3, "G": 1/3}

    for pillar, cat_weights in pillar_config.items():
        score_col = f"{pillar}_score"
        df[score_col] = 0.0

        total_weight = sum(cat_weights.values())
        if total_weight == 0:
            warnings.warn(f"No weights defined for pillar {pillar}, skipping", UserWarning)
            continue

        for category, weight in cat_weights.items():
            # Look for normalized indicator columns matching this category
            # Pattern: {category}_{indicator}_norm or {category}_norm or indicator names containing category
            category_lower = category.lower()
            matching_cols = []
            
            # Try multiple patterns
            patterns = [
                f"{category}_norm",  # Exact match
                f"{category_lower}_norm",  # Lowercase
            ]
            
            # Also search for columns containing category keywords
            category_keywords = {
                "emissions": ["emission", "scope1", "scope2", "carbon", "ghg"],
                "energy": ["energy", "renewable", "efficiency"],
                "water": ["water"],
                "waste": ["waste", "diversion"],
                "labor": ["labor", "turnover", "employee"],
                "diversity": ["diversity", "gender", "women"],
                "health_safety": ["injury", "safety", "health"],
                "community": ["community"],
                "board": ["board", "independence"],
                "comp": ["comp", "pay", "ceo", "exec"],
                "shareholder_rights": ["shareholder", "rights"],
                "ethics": ["ethics", "compliance"],
            }
            
            # Search for normalized columns
            for col in df.columns:
                if col.endswith("_norm"):
                    col_lower = col.lower()
                    # Check exact match
                    if category_lower in col_lower or any(kw in col_lower for kw in category_keywords.get(category, [category_lower])):
                        matching_cols.append(col)
            
            if not matching_cols:
                warnings.warn(f"No indicator found for category '{category}' in pillar {pillar}", UserWarning)
                continue

            # Use first matching column (or average if multiple)
            if len(matching_cols) == 1:
                df[score_col] += (weight / total_weight) * df[matching_cols[0]].fillna(0)
            else:
                df[score_col] += (weight / total_weight) * df[matching_cols].mean(axis=1).fillna(0)

    # Compute composite ESG score
    total_pillar_weight = sum(pillar_weights.values())
    df["ESG_composite"] = 0.0
    for pillar, weight in pillar_weights.items():
        score_col = f"{pillar}_score"
        if score_col in df.columns:
            df["ESG_composite"] += (weight / total_pillar_weight) * df[score_col].fillna(0)

    # Scale to 0-100 for interpretability
    for col in ["E_score", "S_score", "G_score", "ESG_composite"]:
        if col in df.columns:
            # Rescale from z-score-like to 0-100 (assuming roughly normal distribution)
            df[col] = 50 + (df[col] * 20)  # Center at 50, scale by 20
            df[col] = df[col].clip(0, 100)

    return df


class CompositeIndexBuilder:
    """Build composite ESG index from raw indicators."""

    def __init__(self, config: dict[str, Any]):
        """Initialize with index configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Index configuration dict (from index_config.yaml)
        """
        self.config = config
        self.esg_cfg = config.get("esg_index", {})

    def build(
        self,
        df: pd.DataFrame,
        indicator_cols: list[str],
        *,
        id_col: str = "ticker",
    ) -> pd.DataFrame:
        """Build composite ESG index.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with indicator columns
        indicator_cols : list[str]
            List of indicator column names to include
        id_col : str, default "ticker"
            Identifier column name

        Returns
        -------
        pd.DataFrame
            DataFrame with added normalized indicators, pillar scores, and composite score
        """
        # Step 1: Normalize indicators
        norm_cfg = self.esg_cfg.get("normalization", {})
        df = normalize_indicators(
            df,
            indicator_cols,
            method=norm_cfg.get("method", "zscore"),
            by_group=norm_cfg.get("by_group", {}),
            winsorize=norm_cfg.get("winsorize", {}),
        )

        # Step 2: Compute pillar scores
        category_weights_raw = self.esg_cfg.get("category_weights", {})
        pillar_weights_raw = self.esg_cfg.get("pillar_weights", {})
        
        # Extract default weights from weight ranges (if they're dicts with min/max/default)
        def extract_default_weight(weight_value):
            """Extract default weight from weight range dict or return value directly."""
            if isinstance(weight_value, dict):
                return weight_value.get("default", weight_value.get("max", 0.0))
            return weight_value
        
        # Process category weights
        category_weights = {}
        for pillar, cat_dict in category_weights_raw.items():
            if isinstance(cat_dict, dict):
                category_weights[pillar] = {
                    cat: extract_default_weight(weight_val)
                    for cat, weight_val in cat_dict.items()
                    if cat != "constraint"  # Skip constraint keys
                }
            else:
                category_weights[pillar] = cat_dict
        
        # Process pillar weights
        pillar_weights = {}
        for pillar, weight_val in pillar_weights_raw.items():
            if pillar != "constraint":  # Skip constraint keys
                pillar_weights[pillar] = extract_default_weight(weight_val)
        
        df = compute_pillar_scores(df, category_weights, pillar_weights=pillar_weights)

        return df
