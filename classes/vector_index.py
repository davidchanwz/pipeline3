import numpy as np
from typing import List, Dict

from services.embedder import EmbeddingService


class VectorIndex:
    """
    Minimal in-memory vector index.
    Stores code embeddings and supports cosine similarity search.
    Relies on Codebook as the source of truth for all metadata.
    """

    def __init__(self, codebook, embed_service: EmbeddingService):
        """
        codebook: your Codebook instance
        embed_service: EmbeddingService instance (required).

        VectorIndex requires an EmbeddingService which owns how embeddings
        are constructed from Code objects (name + description).
        """
        self.codebook = codebook
        self.embed_service = embed_service
        self.index: Dict[int, List[float]] = {}  # code_id -> embedding

    # ------------------------------------------------------------
    # Add / Update
    # ------------------------------------------------------------
    def add_or_update(self, code):
        """
        Update the index entry for this code.
        If code.embedding is missing the method will compute it using the
        configured EmbeddingService (preferred) or the legacy embed_fn.
        """
        if code.embedding is None:
            if self.embed_service is not None:
                # EmbeddingService operates on Code objects and assigns back
                # to code.embedding.
                self.embed_service.embed_code(code)
            else:
                # Leave None (or you may choose to raise)
                raise RuntimeError("No EmbeddingService configured for VectorIndex")

        self.index[code.code_id] = code.embedding

    def remove(self, code_id: int):
        """Remove an entry from the index."""
        if code_id in self.index:
            del self.index[code_id]

    # ------------------------------------------------------------
    # Search
    # ------------------------------------------------------------
    def search(self, embedding, top_k=5, min_score=0.6, function_filter=None):
        """
        embedding: list[float]
        Returns top-k similar Code objects with similarity >= min_score.
        """
        if not self.index:
            return []

        query = np.array(embedding)
        results = []

        # Normalize function_filter to a comparable string if provided
        target_func = None
        if function_filter is not None:
            # Accept either Enum or string
            try:
                target_func = function_filter.value  # type: ignore
            except Exception:
                target_func = str(function_filter)

        for code_id, vec in self.index.items():
            # If a function filter is provided, skip codes that don't match
            if target_func is not None:
                code_obj = self.codebook.codes.get(code_id)
                if not code_obj:
                    continue
                code_func = getattr(code_obj.function, "value", str(code_obj.function))
                if str(code_func) != str(target_func):
                    continue

            sim = self._cosine(query, np.array(vec))
            if sim >= min_score:
                results.append((sim, code_id))

        results.sort(key=lambda x: x[0], reverse=True)
        top = results[:top_k]
        

        # Return actual Code objects
# Return actual Code objects - filter out any missing codes
        return [
            self.codebook.codes[cid] 
            for _, cid in top 
            if cid in self.codebook.codes  # Add this check
        ]
    # ------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------
    @staticmethod
    def _cosine(v1, v2):
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def embed_text(self, text: str):
        """Convenience wrapper for embedding raw text."""

        # Create a minimal temporary object with name/description so we can
        # reuse EmbeddingService.embed_codes which will choose provider or
        # fallback as configured.
        class _Tmp:
            def __init__(self, name, description):
                self.name = name
                self.description = description
                self.code_id = None
                self.embedding = None

        tmp = _Tmp("", text)
        # embed_codes will populate tmp.embedding
        self.embed_service.embed_codes([tmp])
        return tmp.embedding
