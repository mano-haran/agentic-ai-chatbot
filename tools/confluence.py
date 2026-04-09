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

import re
from functools import lru_cache

import config

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

def find_confluence_page_ids(query: str, top_k: int = 5) -> str:
    """
    Search the Chroma knowledge base for Confluence pages relevant to the query.

    Uses vector similarity search to identify which pages contain content
    matching the query. Returns page IDs, titles, relevance scores, and the
    section names that matched — NOT the chunk text itself.

    The page IDs returned should be passed to fetch_page_by_id to retrieve
    full page content for answering the user's question.

    Args:
        query:  Natural-language question or search phrase.
        top_k:  Maximum number of distinct pages to return (default 5).

    Returns:
        A structured PAGE_RESULTS block, or [NO PAGES FOUND] if nothing meets
        the relevance threshold, with a suggestion to use fetch_confluence_page.
    """
    try:
        store = _get_store()
    except Exception as exc:
        return (
            f"[find_confluence_page_ids] Vector store unavailable: {exc}\n"
            "Run scripts/ingest_confluence.py to populate the knowledge base."
        )

    try:
        from langchain_core.documents import Document
        results: list[tuple[Document, float]] = store.similarity_search_with_relevance_scores(
            query, k=top_k * 4
        )
    except Exception as exc:
        return f"[find_confluence_page_ids] Search failed: {exc}"

    # Filter by relevance threshold
    relevant = [(doc, score) for doc, score in results if score >= _RELEVANCE_THRESHOLD]

    if not relevant:
        return (
            f"[NO PAGES FOUND] No relevant pages found in the vector store for: '{query}'\n"
            "Suggestion: call fetch_confluence_page with the same query to search "
            "Confluence directly via CQL."
        )

    # Group by page_id
    page_map: dict[str, dict] = {}

    for doc, score in relevant:
        meta = doc.metadata

        # Extract page_id from metadata or parse from URL
        page_id = meta.get("page_id", "")
        if not page_id:
            url = meta.get("url", "")
            m = re.search(r"pageId=(\d+)", url)
            if m:
                page_id = m.group(1)

        if not page_id:
            continue  # skip results with no identifiable page_id

        section = meta.get("section", "")
        title = meta.get("title", "Unknown")
        url = meta.get("url", "")

        if page_id not in page_map:
            page_map[page_id] = {
                "page_id": page_id,
                "title": title,
                "url": url,
                "best_score": score,
                "matched_sections": [],
            }
        else:
            if score > page_map[page_id]["best_score"]:
                page_map[page_id]["best_score"] = score

        if section and section not in page_map[page_id]["matched_sections"]:
            if len(page_map[page_id]["matched_sections"]) < 5:
                page_map[page_id]["matched_sections"].append(section)

    if not page_map:
        return (
            f"[NO PAGES FOUND] Chunks were found but none had a resolvable page ID "
            f"for query: '{query}'\n"
            "Suggestion: call fetch_confluence_page with the same query."
        )

    # Sort by best_score descending, return top_k
    sorted_pages = sorted(page_map.values(), key=lambda p: p["best_score"], reverse=True)
    top_pages = sorted_pages[:top_k]

    lines = ["PAGE_RESULTS:"]
    for p in top_pages:
        sections_label = ", ".join(p["matched_sections"]) if p["matched_sections"] else ""
        lines.append(
            f'page_id={p["page_id"]} | title="{p["title"]}" | '
            f'score={p["best_score"]:.2f} | matched_sections=[{sections_label}]'
        )

    return "\n".join(lines)


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

        page = cf.get_page_by_id(page_id, expand="body.storage")

        title = page.get("title", "Untitled")
        base = config.CONFLUENCE_URL.rstrip("/")
        url = f"{base}/pages/viewpage.action?pageId={page_id}"

        html = page.get("body", {}).get("storage", {}).get("value", "")

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
