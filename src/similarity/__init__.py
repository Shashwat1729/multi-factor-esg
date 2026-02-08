"""Company Similarity Metrics Module."""

from .cosine_similarity import compute_similarity_matrix, cosine_similarity
from .preference_scoring import PreferenceScorer

__all__ = [
    "cosine_similarity",
    "compute_similarity_matrix",
    "PreferenceScorer",
]
