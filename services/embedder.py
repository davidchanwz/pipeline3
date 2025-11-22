from typing import Iterable, List, Dict, Optional
import hashlib
import math
import logging

from classes.dataclasses import Code

try:
    import openai

    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service to produce vector embeddings for Code objects.

    Behavior:
    - Embeddings are generated from the concatenation of code.name + code.description
      (only these two fields are used).
    - If OpenAI is available, and an `openai_model` is provided, the service will
      call the OpenAI embeddings API. Otherwise it falls back to a deterministic
      local function that produces a fixed-size vector derived from a hash.

    The deterministic fallback ensures the repository can run without network
    access or API keys; it is stable and repeatable for the same input.
    """

    def __init__(self, openai_model: Optional[str] = None, dim: int = 1536):
        """Create an EmbeddingService.

        Args:
            openai_model: If provided and the openai package is present, used
                as the model name for the OpenAI embeddings API (e.g. "text-embedding-ada-002").
            dim: dimensionality for the fallback deterministic embeddings.
        """
        self.openai_model = openai_model if _HAS_OPENAI and openai_model else None
        self.dim = dim

    @staticmethod
    def _text_for_code(code: Code) -> str:
        """Build the text used for embedding from a Code: name + description.

        This ensures only the requested fields are used.
        """
        name = (code.name or "").strip()
        desc = (code.description or "").strip()
        if name and desc:
            return f"{name}\n\n{desc}"
        return name or desc

    def _fallback_embed(self, text: str) -> List[float]:
        """Deterministic hash-based embedding fallback.

        Produces a vector of length `self.dim` by repeatedly hashing the input
        with a counter and mapping bytes into floats in [-1, 1]. This is
        deterministic and stable across runs for the same input.
        """
        if not text:
            return [0.0] * self.dim

        out: List[float] = []
        counter = 0
        # Keep hashing until we have enough bytes to fill self.dim floats
        while len(out) < self.dim:
            h = hashlib.sha256()
            h.update(text.encode("utf-8"))
            h.update(counter.to_bytes(4, "big"))
            digest = h.digest()
            # convert each byte to a float in [-1,1]
            for b in digest:
                if len(out) >= self.dim:
                    break
                val = (b / 255.0) * 2.0 - 1.0
                out.append(val)
            counter += 1

        # Normalize vector to unit length to be well-behaved for similarity
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]

    def _openai_embed(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embeddings API for a batch of texts.

        Returns a list of embedding vectors. Raises RuntimeError if OpenAI is
        not available or the call fails.
        """
        if not _HAS_OPENAI or not self.openai_model:
            raise RuntimeError("OpenAI not configured or model not provided")

        try:
            # Use OpenAI v1.0+ API
            from openai import OpenAI

            client = OpenAI()  # Uses OPENAI_API_KEY env var
            resp = client.embeddings.create(model=self.openai_model, input=texts)
        except Exception as exc:
            logger.exception("OpenAI embedding call failed")
            raise RuntimeError("OpenAI embedding request failed") from exc

        embeddings: List[List[float]] = []
        for item in resp.data:
            embeddings.append(item.embedding)
        return embeddings

    def embed_code(self, code: Code) -> List[float]:
        """Generate an embedding for a single Code and assign it to code.embedding.

        Only `code.name` and `code.description` are used.
        Returns the embedding vector.
        """
        text = self._text_for_code(code)
        if self.openai_model:
            try:
                emb = self._openai_embed([text])[0]
                code.embedding = emb
                return emb
            except Exception:
                logger.info(
                    "Falling back to local embedding for code id=%s",
                    getattr(code, "code_id", None),
                )

        emb = self._fallback_embed(text)
        code.embedding = emb
        return emb

    def embed_codes(self, codes: Iterable[Code]) -> Dict[int, List[float]]:
        """Batch embed an iterable of Code objects.

        Attempts to use OpenAI in batch if configured; otherwise generates
        deterministic fallback embeddings per code. The method assigns
        embeddings back onto each Code.embedding and returns a mapping of
        code_id -> embedding.
        """
        codes_list = list(codes)
        texts = [self._text_for_code(c) for c in codes_list]

        results: List[List[float]] = []
        if self.openai_model:
            try:
                results = self._openai_embed(texts)
            except Exception:
                logger.info("OpenAI embedding failed; using fallback for whole batch")
                results = [self._fallback_embed(t) for t in texts]
        else:
            results = [self._fallback_embed(t) for t in texts]

        mapping: Dict[int, List[float]] = {}
        for code, emb in zip(codes_list, results):
            code.embedding = emb
            mapping[getattr(code, "code_id", None)] = emb

        return mapping
