"""
Confluence RAG search tool.

Uses a Chroma vector store populated by scripts/ingest_confluence.py.
The store is lazy-loaded on first call so startup cost is zero when the
confluence_qa workflow is not used.

Tool: search_confluence(query, top_k=5)
  Returns the top-k most relevant Confluence page chunks, formatted with
  page title, section heading, source URL, and relevance score.
  Results below the relevance threshold (0.30) are filtered out.
"""

from __future__ import annotations
from functools import lru_cache
from langchain_core.documents import Document

import config


# Minimum cosine-similarity score to include a chunk in results.
# Chroma distance is 1 - cosine_similarity, so threshold=0.30 means similarity >= 0.70.
_RELEVANCE_THRESHOLD = 0.30


@lru_cache(maxsize=1)
def _get_store():
    """Lazy-load and cache the Chroma vector store."""
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    from framework.core.embeddings import get_embeddings

    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_PATH,
    )


def search_confluence(query: str, top_k: int = 5) -> str:
    """
    Search the Confluence knowledge base for content relevant to the query.

    Args:
        query: Natural-language question or search phrase.
        top_k: Maximum number of document chunks to return (default 5).

    Returns:
        Formatted string with relevant chunks, each showing page title,
        section, URL, relevance score, and content.  Returns a "no results"
        message if nothing meets the relevance threshold.
    """
    try:
        store = _get_store()
    except Exception as exc:
        return (
            f"[search_confluence] Vector store unavailable: {exc}\n"
            "Run scripts/ingest_confluence.py to populate the knowledge base."
        )

    try:
        results: list[tuple[Document, float]] = store.similarity_search_with_relevance_scores(
            query, k=top_k
        )
    except Exception as exc:
        return f"[search_confluence] Search failed: {exc}"

    # Filter by relevance threshold
    relevant = [(doc, score) for doc, score in results if score >= _RELEVANCE_THRESHOLD]

    if not relevant:
        return (
            f"No relevant results found for: '{query}'\n"
            "The knowledge base may not contain information on this topic, "
            "or you may need to rephrase the query."
        )

    parts: list[str] = []
    for i, (doc, score) in enumerate(relevant, 1):
        meta = doc.metadata
        title = meta.get("title", "Unknown page")
        section = meta.get("section", "")
        url = meta.get("url", "")
        section_label = f" › {section}" if section else ""
        url_label = f"\n   URL: {url}" if url else ""
        parts.append(
            f"[{i}] {title}{section_label}  (relevance: {score:.2f}){url_label}\n"
            f"{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(parts)
