"""
Confluence tools for the DevOps Knowledgebase workflow.

Two tools are exposed:

  search_confluence(query, top_k=5)
      Vector-similarity search over the pre-ingested Chroma store.
      Fast, covers all ingested pages. Use this first.

  fetch_confluence_page(query)
      Full-text CQL search against the live Confluence Data Center API,
      then returns the complete plain-text body of the best-matching page.
      Use as a fallback when search_confluence returns no results or the
      retrieved chunks do not contain enough information to answer the question.

The Chroma store is lazy-loaded and cached on first use (zero startup cost
when the workflow is not active).
"""

from __future__ import annotations
from functools import lru_cache

import config

# Minimum cosine-similarity score to include a chunk in results.
# Lowered from 0.30 to 0.20 so domain-specific content that may embed with
# moderate similarity is not silently filtered out.
_RELEVANCE_THRESHOLD = 0.20


# ── Vector store (RAG) ─────────────────────────────────────────────────────────

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
    Search the Confluence knowledge base using vector similarity (RAG).

    Searches the pre-ingested Chroma vector store for chunks semantically
    similar to the query.  Each result includes the page title, section,
    source URL, relevance score, and content excerpt.

    Args:
        query: Natural-language question or search phrase.
        top_k: Maximum number of document chunks to return (default 5).

    Returns:
        Formatted string of matching chunks.  Each chunk is numbered [1], [2]
        etc. and includes the exact URL stored during ingestion — cite these
        URLs verbatim in the final answer.
        Returns a "no results" message when nothing meets the relevance threshold.
    """
    try:
        store = _get_store()
    except Exception as exc:
        return (
            f"[search_confluence] Vector store unavailable: {exc}\n"
            "Run scripts/ingest_confluence.py to populate the knowledge base."
        )

    try:
        from langchain_core.documents import Document
        results: list[tuple[Document, float]] = store.similarity_search_with_relevance_scores(
            query, k=top_k
        )
    except Exception as exc:
        return f"[search_confluence] Search failed: {exc}"

    relevant = [(doc, score) for doc, score in results if score >= _RELEVANCE_THRESHOLD]

    if not relevant:
        return (
            f"[NO RESULTS] No relevant chunks found in the vector store for: '{query}'\n"
            "Suggestion: call fetch_confluence_page with the same query to search "
            "Confluence directly."
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


# ── Direct Confluence API (fallback) ───────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """Strip HTML tags and return clean plain text."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        # Remove script/style noise
        for tag in soup(["script", "style", "ac:structured-macro"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        # Fallback: naive tag stripping
        import re
        return re.sub(r"<[^>]+>", " ", html).strip()


def fetch_confluence_page(query: str) -> str:
    """
    Search Confluence Data Center using CQL full-text search and return the
    complete content of the best-matching page(s).

    Use this as a fallback when search_confluence returns [NO RESULTS] or when
    the retrieved vector chunks do not contain sufficient information to answer
    the question.

    Args:
        query: Natural-language question or topic to search for.

    Returns:
        Full plain-text content of the most relevant Confluence page(s),
        each prefixed with the page title and its exact URL.
        Returns an error message if Confluence is unreachable or no pages match.
    """
    try:
        from atlassian import Confluence
    except ImportError:
        return (
            "[fetch_confluence_page] atlassian-python-api is not installed.\n"
            "Run: pip install atlassian-python-api"
        )

    if not config.CONFLUENCE_URL or not config.CONFLUENCE_TOKEN:
        return (
            "[fetch_confluence_page] CONFLUENCE_URL or CONFLUENCE_TOKEN is not configured.\n"
            "Set both variables in your .env file."
        )

    cf = Confluence(
        url=config.CONFLUENCE_URL,
        token=config.CONFLUENCE_TOKEN,
        cloud=False,
    )

    # Build a CQL query.  Escape double-quotes inside the search phrase.
    escaped = query.replace('"', '\\"')
    cql = f'text ~ "{escaped}" AND type = "page" ORDER BY score DESC'

    try:
        response = cf.cql(cql, limit=2, expand="body.storage")
    except Exception as exc:
        return f"[fetch_confluence_page] Confluence CQL search failed: {exc}"

    pages = response.get("results", []) if isinstance(response, dict) else []

    if not pages:
        return (
            f"[fetch_confluence_page] No pages found in Confluence matching: '{query}'\n"
            "The knowledge base may not contain information on this topic."
        )

    parts: list[str] = []
    base = config.CONFLUENCE_URL.rstrip("/")

    for page in pages:
        title = page.get("title", "Untitled")
        page_id = page.get("id", "")
        url = (
            f"{base}/pages/viewpage.action?pageId={page_id}"
            if page_id else base
        )
        html = page.get("body", {}).get("storage", {}).get("value", "")
        text = _html_to_text(html) if html else "(no content available)"

        # Cap per-page content to avoid blowing the context window.
        # ~4 000 chars ≈ ~1 000 tokens, enough for a detailed section.
        if len(text) > 4000:
            text = text[:4000] + "\n...[content truncated]"

        parts.append(f"**{title}**\nURL: {url}\n\n{text}")

    return "\n\n---\n\n".join(parts)
