"""
BM25 keyword search index for hybrid RAG retrieval.

Provides:
  BM25Index        — wraps rank_bm25.BM25Okapi with tokenisation and metadata storage
  rrf_merge        — Reciprocal Rank Fusion over multiple ranked lists
  get_bm25_index   — lazy loader from BM25_INDEX_PATH (returns None when disabled)
  save_bm25_index  — serialise index to disk (called by ingest_confluence.py)

Air-gapped safe: rank_bm25 is pure Python with no network dependency.
"""

from __future__ import annotations

import pickle
import re
from functools import lru_cache
from pathlib import Path


def _tokenise(text: str) -> list[str]:
    """Lowercase word tokeniser — splits on non-alphanumeric characters."""
    return re.findall(r"\b\w+\b", text.lower())


class BM25Index:
    """
    Thin wrapper around rank_bm25.BM25Okapi that stores document texts and
    per-document metadata so callers can look up page information from a hit index.

    Args:
        documents: List of chunk text strings (same order as metadata).
        metadata:  Per-document dicts — should include at minimum ``page_id``.
                   Defaults to empty dicts if omitted.
    """

    def __init__(
        self,
        documents: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for BM25 hybrid search. "
                "Run: pip install rank-bm25"
            )

        self._documents = documents
        self._metadata: list[dict] = metadata if metadata is not None else [{} for _ in documents]
        tokenised = [_tokenise(doc) for doc in documents]
        self._bm25 = BM25Okapi(tokenised)

    @property
    def document_count(self) -> int:
        return len(self._documents)

    def get_document(self, idx: int) -> str:
        return self._documents[idx]

    def get_metadata(self, idx: int) -> dict:
        return self._metadata[idx]

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Keyword search over the index.

        Returns:
            List of ``(document_index, score)`` tuples sorted by score descending,
            limited to top_k non-zero results.
        """
        tokens = _tokenise(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        indexed = [(i, float(s)) for i, s in enumerate(scores) if s > 0.0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]


def rrf_merge(ranked_lists: list[list[str]], k: int = 60) -> list[str]:
    """
    Reciprocal Rank Fusion — merge multiple ranked lists of string IDs.

    Args:
        ranked_lists: Each inner list is a sequence of IDs (e.g. page_ids) sorted
                      best-first.  An ID may appear in multiple lists.
        k:            Smoothing constant (standard value from paper: 60).

    Returns:
        Merged list of IDs sorted by RRF score descending.

    Formula: score(d) = Σ_i  1 / (k + rank_i(d))   where rank is 1-based.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank_zero, item_id in enumerate(ranked):
            rank = rank_zero + 1  # 1-based
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


def save_bm25_index(index: BM25Index, path: str) -> None:
    """Serialise a BM25Index to disk using pickle."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(index, f)
    print(f"[bm25] index saved → {path} ({index.document_count} documents)")


@lru_cache(maxsize=1)
def get_bm25_index() -> "BM25Index | None":
    """
    Lazy-load and cache the BM25 index from BM25_INDEX_PATH.

    Returns None when BM25 is disabled (ENABLE_BM25=false) or the index file
    does not exist — callers fall back to vector-only search automatically.
    """
    import config

    if not config.ENABLE_BM25:
        return None

    path = Path(config.BM25_INDEX_PATH)
    if not path.exists():
        print(
            f"[bm25] WARNING: ENABLE_BM25=true but index not found at {path}. "
            "Run scripts/ingest_confluence.py to build the BM25 index. "
            "Falling back to vector-only search."
        )
        return None

    try:
        with open(path, "rb") as f:
            index: BM25Index = pickle.load(f)
        print(f"[bm25] loaded index from {path} ({index.document_count} documents)")
        return index
    except Exception as exc:
        print(
            f"[bm25] WARNING: failed to load index from {path}: {exc}. "
            "Falling back to vector-only search."
        )
        return None
