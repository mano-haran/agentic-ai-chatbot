"""
Reranker implementations for the RAG pipeline.

Three providers (selected via RERANKER_PROVIDER):
  none              — pass-through; returns candidates in original order (default)
  openai_compatible — POST to {RERANKER_BASE_URL}/v1/rerank using httpx
  local             — sentence-transformers CrossEncoder loaded from a local path

Air-gapped safety:
  - TRANSFORMERS_OFFLINE / HF_DATASETS_OFFLINE are set before any HuggingFace import
  - openai_compatible uses httpx directly — no SDK, only the internal gateway reachable
  - local model must exist under models/ (or any path set in RERANKER_LOCAL_MODEL)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from functools import lru_cache


class BaseReranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[int]:
        """
        Rerank documents relative to a query.

        Args:
            query:     The search query.
            documents: Candidate document texts.
            top_n:     Number of results to return.

        Returns:
            List of original *indices* into ``documents``, sorted best-first,
            length <= top_n.
        """


class NoopReranker(BaseReranker):
    """Pass-through: returns documents in original order (no reranking)."""

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[int]:
        return list(range(min(top_n, len(documents))))


class OpenAICompatibleReranker(BaseReranker):
    """
    Reranker via an OpenAI-compatible ``/v1/rerank`` endpoint.

    Compatible with: mxbai-rerank-large-v1 served by vLLM or Infinity,
    Jina Reranker API, Cohere via proxy, etc.

    Uses httpx directly — no SDK dependency.  Only the internal gateway needs
    to be reachable; no internet access required.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self.endpoint = base_url.rstrip("/") + "/v1/rerank"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[int]:
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for RERANKER_PROVIDER=openai_compatible. "
                "Run: pip install httpx"
            )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }

        try:
            resp = httpx.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"[reranker] OpenAI-compatible rerank request to {self.endpoint} failed: {exc}"
            ) from exc

        # Standard response: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        results = data.get("results", [])
        results.sort(key=lambda r: r.get("relevance_score", 0.0), reverse=True)
        return [r["index"] for r in results[:top_n]]


class LocalCrossEncoderReranker(BaseReranker):
    """
    Local reranker using sentence-transformers CrossEncoder.

    Air-gapped safe:
    - Set TRANSFORMERS_OFFLINE=1 and HF_DATASETS_OFFLINE=1 in the environment.
    - Set RERANKER_LOCAL_MODEL to a local directory path (e.g. models/mxbai-rerank-large-v1).
    """

    def __init__(self, model_name_or_path: str) -> None:
        # Force offline mode before importing HuggingFace libraries
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for RERANKER_PROVIDER=local. "
                "Run: pip install sentence-transformers"
            )

        self._model = CrossEncoder(model_name_or_path)

    def rerank(self, query: str, documents: list[str], top_n: int) -> list[int]:
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        indexed = sorted(enumerate(scores), key=lambda x: float(x[1]), reverse=True)
        return [idx for idx, _ in indexed[:top_n]]


@lru_cache(maxsize=1)
def get_reranker() -> BaseReranker:
    """
    Factory: instantiate and cache the reranker based on RERANKER_PROVIDER config.

    Returns a NoopReranker when RERANKER_PROVIDER=none (the default), so callers
    never need to branch on whether reranking is configured.
    """
    import config

    provider = config.RERANKER_PROVIDER.lower()

    if provider == "openai_compatible":
        if not config.RERANKER_BASE_URL:
            raise ValueError(
                "RERANKER_BASE_URL must be set when RERANKER_PROVIDER=openai_compatible. "
                "Example: RERANKER_BASE_URL=https://your-gateway/v1"
            )
        return OpenAICompatibleReranker(
            base_url=config.RERANKER_BASE_URL,
            api_key=config.RERANKER_API_KEY,
            model=config.RERANKER_MODEL,
        )

    if provider == "local":
        return LocalCrossEncoderReranker(
            model_name_or_path=config.RERANKER_LOCAL_MODEL,
        )

    # Default: no reranking
    return NoopReranker()
