"""
Confluence tools for the DevOps Knowledgebase workflow.

Four tools are exposed:

  find_confluence_page_ids(query, top_k=5)
      Vector-similarity search over the pre-ingested Chroma store.
      Returns page IDs, titles, relevance scores, and matched section names —
      NOT the chunk text itself.  Pass the returned page IDs to fetch_page_by_id.

  fetch_page_by_id(page_id)
      Fetch the full content of a Confluence page by its exact numeric page ID.
      Returns the COMPLETE plain-text page content — no truncation, no
      compression.  The full page is handed to the LLM so it can answer any
      question against it, including follow-up questions about sections the
      original query didn't surface.

  fetch_confluence_page(query)
      Full-text CQL search against the live Confluence Data Center API.
      Use as a fallback when find_confluence_page_ids returns [NO PAGES FOUND].

  search_confluence(query, top_k=5)
      Legacy chunk-based vector search — kept for backward compatibility.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import config


# ── Mock mode helpers ──────────────────────────────────────────────────────────

def _mock_confluence_dir() -> Path:
    return Path(config.MOCK_DATA_DIR) / "confluence"


def _mock_fetch_page_by_id(page_id: str) -> str:
    """
    Read a mock page from tests/mock_data/confluence/<page_id>.html and return
    the full plain-text content.  Mirrors the real fetch_page_by_id exactly:
    no compression, no truncation — the whole page is sent to the LLM.
    """
    d = _mock_confluence_dir()
    path = d / f"{page_id}.html"
    if not path.exists():
        return (
            f"[MOCK] Page {page_id} not found.\n"
            f"Create {path} to add a mock page for this page ID."
        )

    html = path.read_text(encoding="utf-8")
    # Extract title from <title> tag or fall back to page_id
    title_m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    title = title_m.group(1).strip() if title_m else f"Page {page_id}"
    url = f"file://{path.resolve()}"

    full = _html_to_text(html)
    return f"**{title}**\nURL: {url}\n\n{full}"


def _mock_fetch_confluence_page(query: str) -> str:
    """
    Simple keyword search across mock HTML files — returns up to 2 best matches.
    Used as the CQL fallback when MOCK_CONFLUENCE=true.
    """
    d = _mock_confluence_dir()
    if not d.exists():
        return (
            f"[MOCK] Mock Confluence directory not found: {d}\n"
            "Create tests/mock_data/confluence/ with .html files."
        )

    query_terms = set(re.findall(r"\w+", query.lower())) - {
        "the", "a", "an", "is", "are", "how", "what", "where", "to", "for",
        "of", "in", "do", "i", "my", "can", "get", "set", "use", "with"
    }

    scored: list[tuple[float, Path, str]] = []
    for path in sorted(d.glob("*.html")):
        html = path.read_text(encoding="utf-8", errors="ignore")
        text = _html_to_text(html).lower()
        title_m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
        title = title_m.group(1).strip() if title_m else path.stem
        score = sum(1 for t in query_terms if t in text) / max(len(query_terms), 1)
        if score > 0:
            scored.append((score, path, title))

    scored.sort(key=lambda x: x[0], reverse=True)

    if not scored:
        return (
            f"[MOCK] No matching pages found for: '{query}'\n"
            "The mock knowledge base may not contain information on this topic."
        )

    parts: list[str] = []
    for _, path, title in scored[:2]:
        html = path.read_text(encoding="utf-8", errors="ignore")
        text = _html_to_text(html)
        page_id = path.stem
        url = f"file://{path.resolve()}"
        parts.append(f"**{title}**\nURL: {url}\n\n{text}")

    return "\n\n---\n\n".join(parts)

# Minimum cosine-similarity score to include a result in page lookups.
# Configurable via config.SIMILARITY_THRESHOLD (env: SIMILARITY_THRESHOLD).
# This module reads config lazily at call time so test suites / runtime
# reloads pick up changes without reimporting.


# ── Vector store ───────────────────────────────────────────────────────────────

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


# ── HTML parsing helpers ───────────────────────────────────────────────────────

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
        return re.sub(r"<[^>]+>", " ", html).strip()


# ── Tool: find_confluence_page_ids ────────────────────────────────────────────

def _extract_page_id(meta: dict) -> str:
    """Extract page_id from chunk metadata — explicit field or parsed from URL."""
    page_id = meta.get("page_id", "")
    if not page_id:
        url = meta.get("url", "")
        m = re.search(r"pageId=(\d+)", url)
        if m:
            page_id = m.group(1)
    return page_id


def _vector_search_pages(query: str, candidate_k: int) -> tuple[list[str], dict[str, dict]]:
    """
    Run vector similarity search and group results by page_id.

    Returns:
        ranked_page_ids: page IDs sorted by best chunk score descending.
        page_map:        dict mapping page_id → page info (title, url, best_score,
                         best_chunk_text, matched_sections).
    """
    from langchain_core.documents import Document

    store = _get_store()
    results: list[tuple[Document, float]] = store.similarity_search_with_relevance_scores(
        query, k=candidate_k
    )

    page_map: dict[str, dict] = {}
    threshold = config.SIMILARITY_THRESHOLD

    for doc, score in results:
        if score < threshold:
            continue

        meta = doc.metadata
        page_id = _extract_page_id(meta)
        if not page_id:
            continue

        section = meta.get("section", "")
        title = meta.get("title", "Unknown")
        url = meta.get("url", "")

        if page_id not in page_map:
            page_map[page_id] = {
                "title": title,
                "url": url,
                "best_score": score,
                "best_chunk_text": doc.page_content,
                "matched_sections": [],
            }
        else:
            if score > page_map[page_id]["best_score"]:
                page_map[page_id]["best_score"] = score
                page_map[page_id]["best_chunk_text"] = doc.page_content

        if section and section not in page_map[page_id]["matched_sections"]:
            if len(page_map[page_id]["matched_sections"]) < 5:
                page_map[page_id]["matched_sections"].append(section)

    ranked = sorted(page_map.keys(), key=lambda p: page_map[p]["best_score"], reverse=True)
    return ranked, page_map


def _bm25_search_pages(query: str, candidate_k: int, page_map: dict[str, dict]) -> list[str]:
    """
    Run BM25 keyword search and group results by page_id.

    Augments page_map in-place with metadata for pages not already present
    (pages found only by BM25, not by vector search).

    Returns:
        ranked_page_ids: page IDs sorted by best BM25 chunk score descending.
    """
    from framework.core.bm25_index import get_bm25_index

    index = get_bm25_index()
    if index is None:
        return []

    hits = index.search(query, top_k=candidate_k)
    page_scores: dict[str, float] = {}

    for doc_idx, score in hits:
        meta = index.get_metadata(doc_idx)
        page_id = meta.get("page_id", "")
        if not page_id:
            continue

        if page_id not in page_scores or score > page_scores[page_id]:
            page_scores[page_id] = score

        # Add to page_map if not yet present (BM25-only hit)
        if page_id not in page_map:
            page_map[page_id] = {
                "title": meta.get("title", "Unknown"),
                "url": meta.get("url", ""),
                "best_score": score,
                "best_chunk_text": index.get_document(doc_idx),
                "matched_sections": [meta.get("section", "")] if meta.get("section") else [],
            }

    return sorted(page_scores.keys(), key=lambda p: page_scores[p], reverse=True)


def find_confluence_page_ids(query: str, top_k: int = 5) -> str:
    """
    Search the Chroma knowledge base for Confluence pages relevant to the query.

    Retrieval pipeline (layers enabled via env vars):
      1. Vector similarity search (always active)
      2. BM25 keyword search + RRF merge  (ENABLE_BM25=true)
      3. Cross-encoder reranking          (RERANKER_PROVIDER != none)

    Returns page IDs, titles, relevance scores, and matched section names —
    NOT the chunk text itself.  Pass the returned page IDs to fetch_page_by_id.

    Args:
        query:  Natural-language question or search phrase.
        top_k:  Maximum number of distinct pages to return (default 5).

    Returns:
        A structured PAGE_RESULTS block, or [NO PAGES FOUND] if nothing meets
        the relevance threshold, with a suggestion to use fetch_confluence_page.
    """
    try:
        store = _get_store()  # noqa: F841 — validates store is accessible
    except Exception as exc:
        return (
            f"[find_confluence_page_ids] Vector store unavailable: {exc}\n"
            "Run scripts/ingest_confluence.py to populate the knowledge base."
        )

    # Determine candidate pool size — fetch more when reranker will narrow it down
    candidate_k = (
        config.RERANKER_CANDIDATE_K
        if config.RERANKER_PROVIDER.lower() != "none"
        else top_k * 4
    )
    # A candidate pool of ~2× is enough recall for page-level grouping; pages
    # are deduped and the best chunk per page wins.  Going higher wastes
    # embedding compute without changing top_k results.
    vector_fetch_k = max(candidate_k * 2, top_k * 4)

    try:
        vector_ranked, page_map = _vector_search_pages(query, vector_fetch_k)
    except Exception as exc:
        return f"[find_confluence_page_ids] Vector search failed: {exc}"

    # ── BM25 + RRF ────────────────────────────────────────────────────────────
    if config.ENABLE_BM25:
        try:
            bm25_ranked = _bm25_search_pages(query, vector_fetch_k, page_map)
        except Exception as exc:
            print(f"[confluence] BM25 search failed: {exc}. Using vector-only ranking.")
            bm25_ranked = []

        if bm25_ranked:
            from framework.core.bm25_index import rrf_merge
            merged = rrf_merge([vector_ranked, bm25_ranked], k=config.RRF_K)
        else:
            merged = vector_ranked
    else:
        merged = vector_ranked

    if not merged:
        return (
            f"[NO PAGES FOUND] No relevant pages found in the knowledge base for: '{query}'\n"
            "Suggestion: call fetch_confluence_page with the same query to search "
            "Confluence directly via CQL."
        )

    # ── Reranker ──────────────────────────────────────────────────────────────
    if config.RERANKER_PROVIDER.lower() != "none":
        candidates = merged[:candidate_k]
        doc_texts = [page_map[p]["best_chunk_text"] for p in candidates]
        try:
            from framework.core.reranker import get_reranker
            reranker = get_reranker()
            reranked_indices = reranker.rerank(query, doc_texts, top_n=top_k)
            top_pages = [candidates[i] for i in reranked_indices]
        except Exception as exc:
            print(f"[confluence] Reranker failed: {exc}. Falling back to RRF/vector ranking.")
            top_pages = candidates[:top_k]
    else:
        top_pages = merged[:top_k]

    if not top_pages:
        return (
            f"[NO PAGES FOUND] Chunks were found but none had a resolvable page ID "
            f"for query: '{query}'\n"
            "Suggestion: call fetch_confluence_page with the same query."
        )

    lines = ["PAGE_RESULTS:"]
    for pid in top_pages:
        info = page_map[pid]
        sections_label = ", ".join(info["matched_sections"]) if info["matched_sections"] else ""
        lines.append(
            f'page_id={pid} | title="{info["title"]}" | '
            f'score={info["best_score"]:.2f} | matched_sections=[{sections_label}]'
        )

    return "\n".join(lines)


# ── Page content cache ────────────────────────────────────────────────────────


def _page_cache_path(page_id: str) -> Path:
    cache_dir = Path(config.PAGE_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{page_id}.json"


def _load_page_cache(page_id: str) -> dict | None:
    """Return the cached entry for *page_id*, or None if it does not exist."""
    path = _page_cache_path(page_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_page_cache(
    page_id: str, title: str, url: str, html: str, confluence_version: int
) -> None:
    """Write a cache entry to disk using an atomic rename so reads never see
    a partially-written file."""
    data = {
        "page_id": page_id,
        "title": title,
        "url": url,
        "html": html,
        "confluence_version": confluence_version,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    path = _page_cache_path(page_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


# ── Tool: fetch_page_by_id ────────────────────────────────────────────────────

def fetch_page_by_id(page_id: str) -> str:
    """
    Fetch the complete content of a Confluence page by its exact page ID.

    Returns the FULL plain-text content of the page — no truncation, no
    compression, no query-guided section filtering.  The entire page is
    handed to the LLM so the user can ask any question about any part of it,
    including follow-up questions about sections the initial query did not
    surface.

    Args:
        page_id: Numeric Confluence page ID (string).

    Returns:
        Formatted string: title, URL, and the full plain-text page content.
    """
    if config.MOCK_CONFLUENCE:
        return _mock_fetch_page_by_id(page_id)

    try:
        from atlassian import Confluence
    except ImportError:
        return (
            "[fetch_page_by_id] atlassian-python-api is not installed.\n"
            "Run: pip install atlassian-python-api"
        )

    if not config.CONFLUENCE_URL or not config.CONFLUENCE_TOKEN:
        return (
            "[fetch_page_by_id] CONFLUENCE_URL or CONFLUENCE_TOKEN is not configured.\n"
            "Set both variables in your .env file."
        )

    try:
        cf = Confluence(
            url=config.CONFLUENCE_URL,
            token=config.CONFLUENCE_TOKEN,
            cloud=False,
        )

        # ── Cache lookup ────────────────────────────────────────────────────────
        # Strategy:
        #   • Cache fresh (< TTL) AND Confluence version unchanged → serve cache
        #   • Cache fresh but Confluence version changed → full fetch + update cache
        #   • Cache stale (≥ TTL) or missing → full fetch + update cache
        ttl_seconds = config.PAGE_CACHE_TTL_HOURS * 3600
        now = datetime.now(timezone.utc)
        cached = _load_page_cache(page_id)
        html = title = url = None
        confluence_version = 0

        if cached:
            cached_at = datetime.fromisoformat(cached["cached_at"])
            age_seconds = (now - cached_at).total_seconds()

            if age_seconds < ttl_seconds:
                # Cache is within TTL — do a lightweight version check.
                # expand="version" fetches only metadata, not the HTML body.
                version_resp = cf.get_page_by_id(page_id, expand="version")
                current_version = (
                    (version_resp or {}).get("version", {}).get("number", 0)
                )
                if current_version == cached["confluence_version"]:
                    # Page unchanged: serve from cache; skip full fetch.
                    html = cached["html"]
                    title = cached["title"]
                    url = cached["url"]
                    confluence_version = current_version

        if html is None:
            # Cache miss, stale, or Confluence page updated — fetch full content.
            page = cf.get_page_by_id(page_id, expand="body.storage,version")

            title = page.get("title", "Untitled")
            base = config.CONFLUENCE_URL.rstrip("/")
            url = f"{base}/pages/viewpage.action?pageId={page_id}"
            html = page.get("body", {}).get("storage", {}).get("value", "")
            confluence_version = (page.get("version") or {}).get("number", 0)

            _save_page_cache(page_id, title, url, html, confluence_version)

        # ── Content rendering ───────────────────────────────────────────────────
        if not html:
            return f"**{title}**\nURL: {url}\n\n(No content available)"

        # Return the complete page as plain text — no compression, no truncation.
        # The LLM gets the whole page so it can answer any question about any
        # section, including follow-ups on material the search query did not hit.
        full = _html_to_text(html)
        return f"**{title}**\nURL: {url}\n\n{full}"

    except Exception as exc:
        return f"[fetch_page_by_id] Failed to fetch page {page_id}: {exc}"


# ── Tool: fetch_confluence_page (CQL fallback) ────────────────────────────────

def fetch_confluence_page(query: str) -> str:
    """
    Search Confluence Data Center using CQL full-text search and return the
    complete content of the best-matching page(s).

    Use this as a fallback when find_confluence_page_ids returns [NO PAGES FOUND]
    or when the retrieved vector chunks do not contain sufficient information to
    answer the question.

    Args:
        query: Natural-language question or topic to search for.

    Returns:
        Full plain-text content of the most relevant Confluence page(s),
        each prefixed with the page title and its exact URL.
        Returns an error message if Confluence is unreachable or no pages match.
    """
    if config.MOCK_CONFLUENCE:
        return _mock_fetch_confluence_page(query)

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

        # Return full plain-text content — no truncation.  The LLM receives
        # the complete page so the user can ask any follow-up question about it.
        parts.append(f"**{title}**\nURL: {url}\n\n{text}")

    return "\n\n---\n\n".join(parts)


# ── Tool: search_and_fetch_pages (atomic retrieve + fetch) ────────────────────

def search_and_fetch_pages(query: str, top_k: int = 1) -> str:
    """
    Atomic RAG retrieval: find the best-matching Confluence page IDs via
    vector search (plus optional BM25 + reranker) AND fetch the FULL page
    content from Confluence for each, all in a single tool call.

    Pipeline:
      1. Vector search on Chroma (+ optional BM25 + RRF, optional reranker)
         → best-matching page IDs
      2. Parallel fetch of the top-k pages from Confluence via page ID
         (threaded, with per-page cache) — full page content, no compression
      3. CQL full-text fallback when vector search returns [NO PAGES FOUND]

    Args:
        query:  Natural-language question or search phrase.
        top_k:  Number of matching pages to fetch (default 1 — return only
                the single best-matching page so the LLM can answer any
                question about it.  Raise if you need more than one page).

    Returns:
        A single formatted block ready to be consumed by the answer agent:

            PAGES_FETCHED: <n>
            ─────────────────────────────
            **Page Title**
            URL: https://...

            <full page content>
            ─────────────────────────────

        Or the [NO PAGES FOUND] / [NO PAGES PARSED] sentinel when nothing
        is retrievable.
    """
    # ── Step 1: locate page IDs via the existing retrieval pipeline ─────────
    located = find_confluence_page_ids(query, top_k=top_k)

    if located.startswith("[NO PAGES FOUND]") or located.startswith("[find_confluence_page_ids]"):
        # Try CQL fallback against live Confluence.
        cql_result = fetch_confluence_page(query)
        if cql_result.startswith("[fetch_confluence_page]"):
            return (
                "[NO PAGES FOUND] No relevant pages in the knowledge base or via "
                f"CQL search for: '{query}'\n\n{cql_result}"
            )
        return (
            "PAGES_FETCHED: 1 (CQL fallback)\n"
            "─────────────────────────────\n"
            f"{cql_result}"
        )

    # Extract page_id values from the PAGE_RESULTS block.  The format is:
    #   page_id=<id> | title="<title>" | score=<0.00> | matched_sections=[...]
    page_ids: list[str] = []
    for line in located.splitlines():
        m = re.match(r"\s*page_id=(\d+)\s*\|", line)
        if m:
            page_ids.append(m.group(1))

    if not page_ids:
        # Defensive: find_confluence_page_ids returned a well-formed block but
        # we couldn't parse any IDs.  Surface the raw block so the LLM can see.
        return f"[NO PAGES PARSED]\n{located}"

    # ── Step 2: fetch full pages in parallel ────────────────────────────────
    # Confluence fetches are I/O-bound (HTTPS + cache read).  A small thread
    # pool matches or beats sequential by 2–3× when top_k > 1.
    from concurrent.futures import ThreadPoolExecutor

    def _safe_fetch(pid: str) -> tuple[str, str]:
        try:
            return pid, fetch_page_by_id(pid)
        except Exception as exc:
            return pid, f"[fetch_page_by_id] Failed to fetch page {pid}: {exc}"

    # Preserve the retrieval order (best score first) in the output.
    max_workers = min(len(page_ids[:top_k]), 4)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fetched = list(ex.map(_safe_fetch, page_ids[:top_k]))

    # ── Step 3: assemble the response block ────────────────────────────────
    separator = "─────────────────────────────"
    parts = [f"PAGES_FETCHED: {len(fetched)}", separator]
    for _, content in fetched:
        parts.append(content)
        parts.append(separator)
    return "\n".join(parts)


# ── Tool: search_confluence (backward compatibility) ──────────────────────────

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

    relevant = [(doc, score) for doc, score in results if score >= config.SIMILARITY_THRESHOLD]

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
