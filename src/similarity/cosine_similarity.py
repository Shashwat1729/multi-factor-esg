"""Cosine Similarity Implementation for ESG Profiles."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray, eps: float = 1e-12) -> float:
    """Compute cosine similarity between two vectors.

    Parameters
    ----------
    vec_a : np.ndarray
        First vector
    vec_b : np.ndarray
        Second vector
    eps : float, default 1e-12
        Small epsilon to avoid division by zero

    Returns
    -------
    float
        Cosine similarity in [0, 1] range (for non-negative vectors)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a < eps or norm_b < eps:
        return 0.0

    dot_product = np.dot(vec_a, vec_b)
    return dot_product / (norm_a * norm_b + eps)


def compute_similarity_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    id_col: str = "ticker",
    metric: str = "cosine",
    feature_weights: dict[str, float] | None = None,
    output_scale: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute pairwise similarity matrix for companies based on ESG profiles.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns and id_col
    feature_cols : list[str]
        Column names to use as feature vector
    id_col : str, default "ticker"
        Identifier column name
    metric : str, default "cosine"
        Similarity metric: "cosine", "euclidean", "jaccard"
    feature_weights : dict[str, float] | None, optional
        Optional weights for each feature column
    output_scale : dict[str, float] | None, optional
        Scale output to [min, max] range (default: [0, 1])

    Returns
    -------
    pd.DataFrame
        Similarity matrix with id_col as index and columns
    """
    output_scale = output_scale or {"min": 0.0, "max": 1.0}

    # Extract feature vectors
    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        warnings.warn(f"Missing feature columns: {missing}", UserWarning)

    if not available_cols:
        raise ValueError("No feature columns available")

    X = df[available_cols].fillna(0).values
    ids = df[id_col].values

    # Apply feature weights if provided
    if feature_weights:
        weights = np.array([feature_weights.get(col, 1.0) for col in available_cols])
        X = X * weights[np.newaxis, :]

    # Compute similarity matrix
    n = len(ids)
    sim_matrix = np.zeros((n, n))

    if metric == "cosine":
        # Normalize vectors
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / (norms + 1e-12)
        sim_matrix = X_norm @ X_norm.T
    elif metric == "euclidean":
        # Convert to similarity (1 / (1 + distance))
        from scipy.spatial.distance import pdist, squareform

        distances = squareform(pdist(X, metric="euclidean"))
        sim_matrix = 1 / (1 + distances)
    elif metric == "jaccard":
        # For binary/categorical features
        # Convert to binary if not already
        X_binary = (X > 0).astype(float)
        for i in range(n):
            for j in range(n):
                intersection = np.sum(X_binary[i] * X_binary[j])
                union = np.sum((X_binary[i] + X_binary[j]) > 0)
                sim_matrix[i, j] = intersection / (union + 1e-12) if union > 0 else 0.0
    else:
        raise ValueError(f"Unknown similarity metric: {metric}")

    # Scale to output range
    if output_scale["min"] != 0.0 or output_scale["max"] != 1.0:
        min_val = sim_matrix.min()
        max_val = sim_matrix.max()
        if max_val > min_val:
            sim_matrix = (sim_matrix - min_val) / (max_val - min_val)
            sim_matrix = sim_matrix * (output_scale["max"] - output_scale["min"]) + output_scale["min"]

    # Create DataFrame
    result = pd.DataFrame(sim_matrix, index=ids, columns=ids)
    return result


def rank_by_similarity(
    similarity_matrix: pd.DataFrame,
    target_id: str,
    *,
    top_n: int = 10,
    exclude_self: bool = True,
) -> pd.Series:
    """Rank companies by similarity to a target company.

    Parameters
    ----------
    similarity_matrix : pd.DataFrame
        Pairwise similarity matrix
    target_id : str
        Target company identifier
    top_n : int, default 10
        Number of top similar companies to return
    exclude_self : bool, default True
        Exclude the target company from results

    Returns
    -------
    pd.Series
        Ranked similarity scores (descending order)
    """
    if target_id not in similarity_matrix.index:
        raise ValueError(f"Target ID '{target_id}' not found in similarity matrix")

    similarities = similarity_matrix.loc[target_id].sort_values(ascending=False)

    if exclude_self:
        similarities = similarities.drop(target_id)

    return similarities.head(top_n)
