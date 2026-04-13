"""
Confluence tools for the DevOps Knowledgebase workflow.

Four tools are exposed:

  find_confluence_page_ids(query, top_k=5)
      Vector-similarity search over the pre-ingested Chroma store.
      Returns page IDs, titles, relevance scores, and matched section names —
      NOT the chunk text itself.  Pass the returned page IDs to fetch_page_by_id.

  fetch_page_by_id(page_id, query="")
      Fetch the full content of a Confluence page by its exact numeric page ID.
      For short pages (< 4 000 chars) the full text is returned.
      For larger pages, query-guided section extraction (Strategy 1) is applied:
      highly relevant sections are returned in full, moderately relevant sections
      are truncated to 3 sentences, and low-relevance sections show heading only.

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


def _mock_fetch_page_by_id(page_id: str, query: str) -> str:
    """
    Read a mock page from tests/mock_data/confluence/<page_id>.html and apply
    the same parsing and compression logic as the real fetch_page_by_id.
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

    sections = _parse_html_sections(html)
    raw_text_len = sum(len(s["content"]) for s in sections)

    if raw_text_len <= _FULL_PAGE_THRESHOLD or not query:
        full = _html_to_text(html)
        return f"**{title}**\nURL: {url}\n\n{full}"
    else:
        compressed = _compress_page_content(sections, query, max_chars=8000)
        return (
            f"**{title}**\nURL: {url}\n\n"
            "[Content compressed — query-guided section extraction applied]\n\n"
            f"{compressed}"
        )


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
        if len(text) > 4000:
            text = text[:4000] + "\n...[content truncated]"
        parts.append(f"**{title}**\nURL: {url}\n\n{text}")

    return "\n\n---\n\n".join(parts)

# Minimum cosine-similarity score to include a result in page lookups.
_RELEVANCE_THRESHOLD = 0.20

# Pages with total plain-text content below this threshold are returned in
# full without compression.  Above it, query-guided section extraction applies.
_FULL_PAGE_THRESHOLD = 4000   # chars


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


def _parse_html_sections(html: str) -> list[dict]:
    """
    Parse Confluence HTML into structured sections preserving heading hierarchy.

    Returns a list of dicts: {heading: str, level: int, content: str}
    Each dict represents one heading plus all body text until the next heading.
    The first dict may have an empty heading (page introduction before any heading).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # bs4 not available — return single section with plain-text content
        return [{"heading": "", "level": 1, "content": _html_to_text(html)}]

    soup = BeautifulSoup(html, "html.parser")

    # Remove noise tags
    for tag in soup(["script", "style", "ac:structured-macro"]):
        tag.decompose()

    sections: list[dict] = []
    current_heading = ""
    current_level = 1
    current_lines: list[str] = []

    def _flush() -> None:
        content = "\n".join(current_lines).strip()
        if current_heading or content:
            sections.append({
                "heading": current_heading,
                "level": current_level,
                "content": content,
            })

    # Iterate all relevant elements in document order
    for tag in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "code", "td"],
        recursive=True,
    ):
        if tag.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            _flush()
            current_heading = tag.get_text(strip=True)
            current_level = int(tag.name[1])
            current_lines = []
        else:
            text = tag.get_text(" ", strip=True)
            if text:
                current_lines.append(text)

    _flush()

    # Filter out completely empty entries
    return [s for s in sections if s["heading"] or s["content"]]


# ── Section scoring and compression ───────────────────────────────────────────

def _score_section_relevance(section: dict, query: str) -> float:
    """
    Score a section's relevance to the query using keyword overlap.

    Heading matches are weighted 2x (headings are concise topic labels).
    Common stopwords are excluded from query terms.
    Returns a float 0.0–1.0.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "how", "what", "where",
        "when", "why", "who", "which", "to", "for", "of", "in", "on", "at",
        "by", "from", "up", "about", "into", "i", "you", "we", "my", "our",
    }

    query_terms = {
        w for w in re.findall(r"\w+", query.lower())
        if w not in stopwords and len(w) > 2
    }

    if not query_terms:
        return 0.5  # no query terms → treat all sections equally

    heading_text = section["heading"].lower()
    content_text = section["content"].lower()

    heading_hits = sum(1 for t in query_terms if t in heading_text)
    content_hits = sum(1 for t in query_terms if t in content_text)

    score = (heading_hits * 2 + content_hits) / (len(query_terms) * 3)
    return min(score, 1.0)


def _compress_page_content(sections: list[dict], query: str, max_chars: int = 8000) -> str:
    """
    Apply query-guided section extraction (Strategy 1).

    Scoring tiers (relative to the best-scoring section in the page):
      HIGH   >= 60% of top score  → full section content included
      MEDIUM >= 20% of top score  → first 3 sentences + '...'
      LOW    <  20% of top score  → heading only (structure preserved)

    Always prepends a TOC skeleton listing every heading so the user can see
    the full page structure and ask follow-up questions about any section, even
    ones that were trimmed.

    Stops adding content once max_chars is reached; appends a count of omitted
    sections so the LLM knows more content exists.
    """
    if not sections:
        return ""

    # Score all sections
    scores = [_score_section_relevance(s, query) for s in sections]
    max_score = max(scores) if scores else 0.0

    high_threshold = max(max_score * 0.6, 0.15)
    medium_threshold = max(max_score * 0.2, 0.05)

    # Build TOC
    toc_lines = ["**Table of Contents**"]
    for section in sections:
        if section["heading"]:
            indent = " " * ((section["level"] - 1) * 2)
            toc_lines.append(f"{indent}• {section['heading']}")
    toc = "\n".join(toc_lines)

    # Build content
    content_parts: list[str] = []
    total_chars = len(toc)
    omitted = 0

    for section, score in zip(sections, scores):
        level = max(1, section["level"])
        heading_prefix = "#" * level + (" " if section["heading"] else "")
        heading_line = heading_prefix + section["heading"] if section["heading"] else ""

        if score >= high_threshold:
            body = section["content"]
            part = (heading_line + "\n" + body).strip() if heading_line else body.strip()
        elif score >= medium_threshold:
            # First 3 sentences
            sentences = re.split(r"(?<=[.!?])\s+", section["content"])
            truncated = " ".join(sentences[:3])
            if len(sentences) > 3:
                truncated += " ..."
            part = (heading_line + "\n" + truncated).strip() if heading_line else truncated.strip()
        else:
            # Heading only
            part = heading_line.strip()

        if not part:
            continue

        if total_chars + len(part) + 2 > max_chars:
            omitted += 1
            continue

        content_parts.append(part)
        total_chars += len(part) + 2  # +2 for the separator

    result = toc + "\n\n" + "\n\n".join(content_parts)

    if omitted:
        result += f"\n\n... [{omitted} more section(s) omitted]"

    return result


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

    for doc, score in results:
        if score < _RELEVANCE_THRESHOLD:
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
        else top_k * 6
    )
    # Always fetch at least top_k * 4 vector candidates for adequate recall
    vector_fetch_k = max(candidate_k * 4, top_k * 4)

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

def fetch_page_by_id(page_id: str, query: str = "") -> str:
    """
    Fetch the full content of a Confluence page by its exact page ID.

    For pages under 4 000 characters: returns the complete plain-text content.
    For larger pages: applies query-guided section extraction (Strategy 1):
      • Sections highly relevant to the query receive full content
      • Sections moderately relevant receive the first 3 sentences
      • Sections with low relevance show heading only (structure preserved)
    A table of contents is always prepended so every section is visible,
    enabling accurate follow-up Q&A on any part of the page.

    Args:
        page_id: Numeric Confluence page ID (string).
        query:   User's original search query for section scoring.
                 Pass empty string to disable scoring (equal weight, all sections).

    Returns:
        Formatted string: title, URL, optional compression notice, content.
    """
    if config.MOCK_CONFLUENCE:
        return _mock_fetch_page_by_id(page_id, query)

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

        sections = _parse_html_sections(html)
        raw_text_len = sum(len(s["content"]) for s in sections)

        if raw_text_len <= _FULL_PAGE_THRESHOLD or not query:
            full = _html_to_text(html)
            return f"**{title}**\nURL: {url}\n\n{full}"
        else:
            compressed = _compress_page_content(sections, query, max_chars=8000)
            return (
                f"**{title}**\nURL: {url}\n\n"
                "[Content compressed — query-guided section extraction applied]\n\n"
                f"{compressed}"
            )

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

        # Cap per-page content to avoid blowing the context window.
        # ~4 000 chars ≈ ~1 000 tokens, enough for a detailed section.
        if len(text) > 4000:
            text = text[:4000] + "\n...[content truncated]"

        parts.append(f"**{title}**\nURL: {url}\n\n{text}")

    return "\n\n---\n\n".join(parts)


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
