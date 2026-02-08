"""Investment Preference Scoring Module.

Combines ESG scores, financial metrics, market factors, operational quality,
risk-adjusted returns, growth, value, stability, similarity, and sector position
into a composite investment preference score using 10 sub-factors.

Weight rationale (from literature and empirical analysis):
- Financial quality factors explain most cross-sectional return variation (Fama-French)
- ESG integration adds risk mitigation and downside protection (Giese et al. 2019)
- Market momentum captures short-term price dynamics (Jegadeesh & Titman 1993)
- Operational quality is a proxy for sustainable competitive advantage (Novy-Marx 2013)
- Risk-adjusted metrics (Sharpe/Sortino) reward efficient return generation
- Similarity and sector position add peer-relative context
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# All 10 score components used in preference scoring
ALL_SCORE_COMPONENTS = [
    "esg_score", "financial_score", "market_score", "operational_score",
    "risk_adjusted_score", "growth_score", "value_score", "stability_score",
    "similarity_rank", "sector_position",
]

# Mapping from config key -> DataFrame column name
SCORE_COLUMN_MAP = {
    "esg_score": "ESG_composite",
    "financial_score": "financial_score",
    "market_score": "market_score",
    "operational_score": "operational_score",
    "risk_adjusted_score": "risk_adjusted_score",
    "growth_score": "growth_score",
    "value_score": "value_score",
    "stability_score": "stability_score",
    "similarity_rank": "similarity_rank",
    "sector_position": "sector_position",
}


class PreferenceScorer:
    """Compute investment preference scores from 10 sub-factor components."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        pref_cfg = config.get("preference_scoring", {})
        self.profiles = pref_cfg.get("investor_profiles", {})

    def compute_preference_score(
        self,
        df: pd.DataFrame,
        *,
        esg_score_col: str = "ESG_composite",
        financial_score_col: str | None = None,
        similarity_rank_col: str | None = None,
        sector_position_col: str | None = None,
        investor_profile: str = "balanced",
    ) -> pd.Series:
        """Compute composite investment preference score.

        Uses ALL 10 sub-factor components with profile-specific weights.
        Components on 0-100 scale are used directly.
        Components on 0-1 scale (similarity_rank, sector_position) are rescaled.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with component scores
        investor_profile : str
            Investor profile: "esg_first", "balanced", "financial_first"

        Returns
        -------
        pd.Series
            Investment preference scores (0-100 scale)
        """
        # Get weights for profile
        if investor_profile in self.profiles:
            weights = self.profiles[investor_profile].copy()
        else:
            # Default balanced weights
            weights = {
                "esg_score": 0.20, "financial_score": 0.25,
                "market_score": 0.10, "operational_score": 0.10,
                "risk_adjusted_score": 0.10, "growth_score": 0.08,
                "value_score": 0.07, "stability_score": 0.05,
                "similarity_rank": 0.03, "sector_position": 0.02,
            }

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        score = pd.Series(0.0, index=df.index)

        for component, weight in weights.items():
            if weight <= 0:
                continue

            col = SCORE_COLUMN_MAP.get(component, component)

            if col in df.columns:
                vals = df[col].fillna(df[col].median() if df[col].notna().any() else 50)

                # similarity_rank and sector_position are on 0-1 scale -> rescale
                if component in ("similarity_rank", "sector_position"):
                    if vals.max() <= 1.0:
                        vals = vals * 100

                score += weight * vals

        return score.clip(0, 100)

    def rank_companies(
        self,
        df: pd.DataFrame,
        *,
        preference_score_col: str = "preference_score",
        top_n: int | None = None,
    ) -> pd.DataFrame:
        """Rank companies by investment preference score."""
        if preference_score_col not in df.columns:
            raise ValueError(f"Preference score column '{preference_score_col}' not found")

        ranked = df.sort_values(preference_score_col, ascending=False)
        if top_n:
            ranked = ranked.head(top_n)

        ranked["preference_rank"] = range(1, len(ranked) + 1)
        return ranked
