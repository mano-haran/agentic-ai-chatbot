"""
Confluence ingestion script.

Fetches Confluence pages, splits them into sections, embeds, and stores in Chroma.

Usage examples:
  # Ingest an entire Confluence space
  python scripts/ingest_confluence.py --space DEV

  # Ingest specific pages by ID
  python scripts/ingest_confluence.py --pages 123456 789012

  # Ingest local HTML/Markdown files (useful for testing without Confluence access)
  python scripts/ingest_confluence.py --dir docs/

  # Clear the existing store and rebuild from scratch
  python scripts/ingest_confluence.py --space DEV --reset

Environment variables (set in .env):
  CONFLUENCE_URL    Base URL, e.g. https://confluence.your-company.com
  CONFLUENCE_TOKEN  Personal Access Token (create in Confluence → Profile → Personal Access Tokens)
  CHROMA_PATH       Path for the Chroma DB (default: data/chroma)
  CHROMA_COLLECTION Collection name (default: confluence)
  EMBEDDING_PROVIDER  local | openai  (default: local)
  EMBEDDING_MODEL   Model name override
"""

import argparse
import re
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv
load_dotenv()

import config
from framework.core.embeddings import get_embeddings


# ── Chunking helpers ────────────────────────────────────────────────────────────

def _html_to_sections(html: str, title: str, url: str) -> list[dict]:
    """
    Parse HTML into section-level chunks.

    Each chunk represents the content under one heading (h1–h3).  Content
    before the first heading is kept as the page introduction.
    Returns a list of dicts: {title, section, url, text}
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 is required: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")
    chunks: list[dict] = []
    current_section = ""
    current_lines: list[str] = []

    def _flush():
        text = " ".join(current_lines).strip()
        if text:
            chunks.append({
                "title": title,
                "section": current_section,
                "url": url,
                "text": text,
            })

    for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "code", "pre"]):
        if tag.name in ("h1", "h2", "h3"):
            _flush()
            current_section = tag.get_text(strip=True)
            current_lines = []
        else:
            text = tag.get_text(" ", strip=True)
            if text:
                current_lines.append(text)

    _flush()
    return chunks


def _markdown_to_sections(md: str, title: str, url: str) -> list[dict]:
    """
    Split Markdown into sections by heading lines (# / ## / ###).
    """
    chunks: list[dict] = []
    current_section = ""
    current_lines: list[str] = []

    def _flush():
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append({"title": title, "section": current_section, "url": url, "text": text})

    for line in md.splitlines():
        m = re.match(r"^#{1,3}\s+(.+)", line)
        if m:
            _flush()
            current_section = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    _flush()
    return chunks


def _plain_to_sections(text: str, title: str, url: str) -> list[dict]:
    """Treat each non-empty paragraph as a chunk."""
    chunks = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if para:
            chunks.append({"title": title, "section": "", "url": url, "text": para})
    return chunks


# ── Confluence API helpers ──────────────────────────────────────────────────────

def _fetch_space_pages(space_key: str) -> list[dict]:
    """Return all pages in a Confluence space using the REST API."""
    try:
        from atlassian import Confluence
    except ImportError:
        raise ImportError(
            "atlassian-python-api is required: pip install atlassian-python-api"
        )

    cf = Confluence(
        url=config.CONFLUENCE_URL,
        token=config.CONFLUENCE_TOKEN,
        cloud=False,
    )

    pages = []
    start = 0
    limit = 50
    while True:
        batch = cf.get_all_pages_from_space(
            space_key, start=start, limit=limit, expand="body.storage,metadata.labels"
        )
        pages.extend(batch)
        if len(batch) < limit:
            break
        start += limit

    print(f"[ingest] fetched {len(pages)} pages from space {space_key}")
    return pages


def _fetch_page_by_id(page_id: str) -> dict:
    try:
        from atlassian import Confluence
    except ImportError:
        raise ImportError(
            "atlassian-python-api is required: pip install atlassian-python-api"
        )

    cf = Confluence(
        url=config.CONFLUENCE_URL,
        token=config.CONFLUENCE_TOKEN,
        cloud=False,
    )
    return cf.get_page_by_id(page_id, expand="body.storage")


def _page_to_chunks(page: dict) -> list[dict]:
    title = page.get("title", "Untitled")
    base_url = config.CONFLUENCE_URL.rstrip("/")
    page_id = page.get("id", "")
    url = f"{base_url}/pages/{page_id}" if page_id else base_url
    html = page.get("body", {}).get("storage", {}).get("value", "")
    if html:
        return _html_to_sections(html, title, url)
    return []


# ── Local file helpers ──────────────────────────────────────────────────────────

def _ingest_dir(directory: str) -> list[dict]:
    """Load all .html, .md, .txt files under a directory."""
    chunks: list[dict] = []
    for path in Path(directory).rglob("*"):
        if path.suffix.lower() in (".html", ".htm"):
            chunks.extend(_html_to_sections(path.read_text(errors="ignore"), path.stem, str(path)))
        elif path.suffix.lower() == ".md":
            chunks.extend(_markdown_to_sections(path.read_text(errors="ignore"), path.stem, str(path)))
        elif path.suffix.lower() == ".txt":
            chunks.extend(_plain_to_sections(path.read_text(errors="ignore"), path.stem, str(path)))
    return chunks


# ── Chroma store helpers ────────────────────────────────────────────────────────

def _get_store(reset: bool = False):
    try:
        from langchain_chroma import Chroma
    except ImportError:
        from langchain_community.vectorstores import Chroma

    if reset:
        import shutil
        chroma_path = Path(config.CHROMA_PATH)
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
            print(f"[ingest] removed existing store at {chroma_path}")

    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=config.CHROMA_PATH,
    )


def _upsert_chunks(store, chunks: list[dict]) -> int:
    """Add chunks to the vector store, deduplicating by (title, section)."""
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content=c["text"],
            metadata={"title": c["title"], "section": c["section"], "url": c["url"]},
        )
        for c in chunks
        if c.get("text", "").strip()
    ]

    if not docs:
        return 0

    # Batch in groups of 100 to avoid exceeding embedding API limits
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        store.add_documents(docs[i : i + batch_size])
        print(f"[ingest] embedded {min(i + batch_size, len(docs))} / {len(docs)} chunks", end="\r")

    print()
    return len(docs)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Confluence pages into Chroma.")
    parser.add_argument("--space", help="Confluence space key (e.g. DEV)")
    parser.add_argument("--pages", nargs="+", metavar="PAGE_ID", help="Specific page IDs")
    parser.add_argument("--dir", metavar="DIRECTORY", help="Local directory with HTML/MD/TXT files")
    parser.add_argument("--reset", action="store_true", help="Clear the store before ingesting")
    args = parser.parse_args()

    if not (args.space or args.pages or args.dir):
        parser.print_help()
        sys.exit(1)

    all_chunks: list[dict] = []

    if args.space:
        for page in _fetch_space_pages(args.space):
            all_chunks.extend(_page_to_chunks(page))

    if args.pages:
        for pid in args.pages:
            page = _fetch_page_by_id(pid)
            all_chunks.extend(_page_to_chunks(page))

    if args.dir:
        dir_chunks = _ingest_dir(args.dir)
        all_chunks.extend(dir_chunks)
        print(f"[ingest] found {len(dir_chunks)} chunks from {args.dir}")

    if not all_chunks:
        print("[ingest] no content found — nothing to store.")
        sys.exit(0)

    print(f"[ingest] total chunks to embed: {len(all_chunks)}")
    store = _get_store(reset=args.reset)
    n = _upsert_chunks(store, all_chunks)
    print(f"[ingest] done — {n} chunks stored in {config.CHROMA_PATH} (collection: {config.CHROMA_COLLECTION})")


if __name__ == "__main__":
    main()
