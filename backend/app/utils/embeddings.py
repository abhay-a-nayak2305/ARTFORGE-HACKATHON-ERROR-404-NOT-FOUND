"""
Singleton wrapper around SentenceTransformer.
Loads once at startup; provides cosine similarity helpers.
"""
import numpy as np
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

_MODEL_NAME = "all-MiniLM-L6-v2"  # 80 MB
class EmbeddingService:
    """
    Singleton SentenceTransformer wrapper.
    Call EmbeddingService.get() to retrieve the shared instance.
    """
    _instance: "EmbeddingService | None" = None
    _model: SentenceTransformer | None = None

    def __init__(self):
        logger.info("Loading SentenceTransformer: %s", _MODEL_NAME)
        self._model = SentenceTransformer(_MODEL_NAME)
        logger.info("SentenceTransformer loaded.")

    @classmethod
    def get(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (N, 384) float32 array."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def similarity(self, a: list[str], b: list[str]) -> np.ndarray:
        """Return (len(a), len(b)) cosine similarity matrix."""
        emb_a = self.embed(a)
        emb_b = self.embed(b)
        if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
            return np.zeros((len(a), len(b)), dtype=np.float32)
        return cosine_similarity(emb_a, emb_b)

    def best_match_score(self, skill: str, candidates: list[str]) -> float:
        """Return highest cosine similarity between skill and any candidate."""
        if not candidates:
            return 0.0
        matrix = self.similarity([skill], candidates)
        return float(matrix[0].max())