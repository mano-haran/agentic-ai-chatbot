"""
Confluence ingestion script.

Fetches Confluence pages, splits them into sections, embeds, and stores in Chroma.
Optionally builds a BM25 keyword index for hybrid retrieval.

Usage examples:
  # Ingest an entire Confluence space
  python scripts/ingest_confluence.py --space DEV

  # Ingest specific pages by ID
  python scripts/ingest_confluence.py --pages 56408673 99568862

  # Ingest local HTML/Markdown files (useful for testing without Confluence access)
  python scripts/ingest_confluence.py --dir docs/

  # Clear the existing store and rebuild from scratch
  python scripts/ingest_confluence.py --space DEV --reset

  # Use docling context-aware chunking (requires: pip install docling)
  CHUNKING_STRATEGY=docling python scripts/ingest_confluence.py --space DEV

  # Build BM25 index for hybrid retrieval (requires: pip install rank-bm25)
  ENABLE_BM25=true python scripts/ingest_confluence.py --space DEV

Environment variables (set in .env):
  CONFLUENCE_URL       Base URL, e.g. https://confluence.your-company.com
  CONFLUENCE_TOKEN     Personal Access Token
  CHROMA_PATH          Path for the Chroma DB (default: data/chroma)
  CHROMA_COLLECTION    Collection name (default: confluence)
  EMBEDDING_PROVIDER   local | openai  (default: local)
  EMBEDDING_MODEL      Model name override
  CHUNKING_STRATEGY         html (default) | docling
  DOCLING_TOKENIZER_PROVIDER  local (default) | openai_compatible
  DOCLING_TOKENIZER_MODEL   Local tokenizer path for docling (air-gapped, used when provider=local)
  ENABLE_BM25          true | false (default: false)
  BM25_INDEX_PATH      Path to save BM25 index (default: data/bm25.pkl)
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


# ── Docling chunking ───────────────────────────────────────────────────────────

def _make_tiktoken_wrapper_class():
    """Build a TiktokenWrapper class that properly subclasses docling's BaseTokenizer.

    In docling >=2.x, HybridChunker is a Pydantic v2 model and its ``tokenizer``
    field is typed as ``BaseTokenizer`` (itself a Pydantic BaseModel).  Passing a
    plain Python object fails Pydantic validation with a ``model_type`` error.
    This factory imports ``BaseTokenizer`` at runtime (after docling is on
    sys.path) and builds a compliant subclass.  If no matching base class is
    found (very old docling), it falls back to a plain Python class so the
    caller can still attempt construction and let the outer try/except handle it.

    The returned class constructor accepts ``model_name`` as its first keyword
    argument and is safe to instantiate before the docling import happens.
    """
    try:
        import tiktoken as _tiktoken
    except ImportError as exc:
        raise ImportError(
            "tiktoken is required for DOCLING_TOKENIZER_PROVIDER=openai_compatible. "
            "It is bundled with langchain-openai: pip install langchain-openai"
        ) from exc

    # Probe known locations for BaseTokenizer across docling / docling-core versions.
    _DoclingBase = None
    for _mod_path, _attr in [
        ("docling_core.transforms.chunker.tokenizer_utils", "BaseTokenizer"),
        ("docling_core.transforms.chunker",                 "BaseTokenizer"),
        ("docling.chunking",                                "BaseTokenizer"),
    ]:
        try:
            import importlib
            _mod = importlib.import_module(_mod_path)
            _candidate = getattr(_mod, _attr, None)
            if _candidate is not None:
                _DoclingBase = _candidate
                break
        except ImportError:
            continue

    if _DoclingBase is not None:
        # Pydantic-aware subclass — satisfies HybridChunker's type validator.
        try:
            from pydantic import PrivateAttr, ConfigDict

            class TiktokenWrapper(_DoclingBase):  # type: ignore[valid-type]
                model_config = ConfigDict(arbitrary_types_allowed=True)

                # Declared Pydantic fields (serialisable)
                model_name: str = "text-embedding-3-small"
                model_max_length: int = 8191

                # Private runtime state — not part of the Pydantic schema
                _enc: object = PrivateAttr(default=None)

                def model_post_init(self, __context: object) -> None:
                    try:
                        self._enc = _tiktoken.encoding_for_model(self.model_name)
                    except KeyError:
                        self._enc = _tiktoken.get_encoding("cl100k_base")

                # ── docling >=2.x interface ─────────────────────────────────
                def count_tokens(self, text: str) -> int:
                    return len(self._enc.encode(text))

                @property
                def max_tokens(self) -> int:
                    return self.model_max_length

                # ── legacy / fallback interface (older docling) ─────────────
                def encode(self, text: str, **kwargs) -> list[int]:  # noqa: ARG002
                    return self._enc.encode(text)

                def decode(self, token_ids: list[int], **kwargs) -> str:  # noqa: ARG002
                    return self._enc.decode(token_ids)

            return TiktokenWrapper

        except Exception:
            pass  # fall through to plain-class fallback below

    # Plain-class fallback for very old docling (no Pydantic validation).
    class TiktokenWrapper:  # type: ignore[no-redef]
        def __init__(self, model_name: str = "text-embedding-3-small", **_: object) -> None:
            try:
                self._enc = _tiktoken.encoding_for_model(model_name)
            except KeyError:
                self._enc = _tiktoken.get_encoding("cl100k_base")
            self.model_max_length: int = 8191

        def count_tokens(self, text: str) -> int:
            return len(self._enc.encode(text))

        @property
        def max_tokens(self) -> int:
            return self.model_max_length

        def encode(self, text: str, **kwargs) -> list[int]:  # noqa: ARG002
            return self._enc.encode(text)

        def decode(self, token_ids: list[int], **kwargs) -> str:  # noqa: ARG002
            return self._enc.decode(token_ids)

    return TiktokenWrapper


def _build_docling_tokenizer():
    """Return the tokenizer (or path string) to pass to HybridChunker.

    - provider=local            → path string from DOCLING_TOKENIZER_MODEL, or None
                                  (None means docling uses its own default tokenizer)
    - provider=openai_compatible → TiktokenWrapper instance (proper BaseTokenizer
                                  subclass for Pydantic v2 compatibility)
    """
    provider = config.DOCLING_TOKENIZER_PROVIDER

    if provider == "openai_compatible":
        model_name = config.EMBEDDING_MODEL or "text-embedding-3-small"
        TiktokenWrapper = _make_tiktoken_wrapper_class()
        return TiktokenWrapper(model_name=model_name)

    # Default: local path (may be empty → None → docling uses its own default)
    return config.DOCLING_TOKENIZER_MODEL or None


def _docling_to_chunks(html: str, title: str, url: str, page_id: str) -> list[dict]:
    """
    Parse HTML using docling's HybridChunker for context-aware chunking.

    Requires: pip install 'docling>=2.0.0'

    Advantages over BeautifulSoup HTML chunking:
    - Tables preserved as atomic units (not split mid-row)
    - Code blocks preserved as atomic units
    - Heading breadcrumbs embedded in every chunk for context
    - Token-aware chunk sizing (avoids very large or very small chunks)

    Air-gapped: set TRANSFORMERS_OFFLINE=1 and DOCLING_TOKENIZER_MODEL to a
    local tokenizer path before running.  HTML input does NOT require ML models
    for document conversion itself — only HybridChunker uses a tokenizer.

    Falls back to _html_to_sections on any error so ingestion never fails silently.
    """
    import os
    import tempfile

    # Ensure HuggingFace libraries stay offline before any import
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    try:
        from docling.document_converter import DocumentConverter
        from docling.chunking import HybridChunker
    except ImportError:
        raise ImportError(
            "docling is required for CHUNKING_STRATEGY=docling. "
            "Run: pip install 'docling>=2.0.0'"
        )

    # Wrap bare HTML bodies so docling's HTML backend handles them correctly
    if not html.strip().lower().startswith("<html"):
        html = f"<html><head><title>{title}</title></head><body>{html}</body></html>"

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
        tmp_path = f.name

    try:
        converter = DocumentConverter()
        result = converter.convert(tmp_path)
        dl_doc = result.document

        # HybridChunker: build the tokenizer based on DOCLING_TOKENIZER_PROVIDER.
        # - local            → path string (air-gapped HuggingFace model) or None
        # - openai_compatible → _TiktokenWrapper (tiktoken, no local model needed)
        tokenizer = _build_docling_tokenizer()
        try:
            chunker = (
                HybridChunker(tokenizer=tokenizer)
                if tokenizer is not None
                else HybridChunker()
            )
        except Exception:
            # Older docling: no `tokenizer` kwarg (TypeError).
            # Newer docling with unrecognised tokenizer type: Pydantic ValidationError.
            # Either way fall back to the default tokenizer.
            chunker = HybridChunker()

        chunks: list[dict] = []
        for chunk in chunker.chunk(dl_doc):
            text = (chunk.text if hasattr(chunk, "text") else str(chunk)).strip()
            if not text:
                continue

            # Extract section heading breadcrumb (e.g. "Intro > Setup > Install")
            section = ""
            meta = getattr(chunk, "meta", None)
            if meta is not None:
                headings = getattr(meta, "headings", None) or []
                section = " > ".join(str(h) for h in headings if h)

            chunks.append({
                "title": title,
                "section": section,
                "url": url,
                "page_id": page_id,
                "text": text,
            })

        return chunks

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── HTML chunking helpers ───────────────────────────────────────────────────────

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
    # Data Center URL format — /pages/viewpage.action?pageId=<id>
    url = f"{base_url}/pages/viewpage.action?pageId={page_id}" if page_id else base_url
    html = page.get("body", {}).get("storage", {}).get("value", "")
    if not html:
        return []

    strategy = config.CHUNKING_STRATEGY.lower()

    if strategy == "docling":
        try:
            return _docling_to_chunks(html, title, url, page_id)
        except Exception as exc:
            print(f"[ingest] docling chunking failed for '{title}' ({page_id}): {exc}")
            print("[ingest] falling back to html chunking for this page")
            chunks = _html_to_sections(html, title, url)
    else:
        chunks = _html_to_sections(html, title, url)

    for chunk in chunks:
        chunk["page_id"] = page_id

    return chunks


# ── Local file helpers ──────────────────────────────────────────────────────────

def _ingest_dir(directory: str) -> list[dict]:
    """Load all .html, .md, .txt files under a directory."""
    chunks: list[dict] = []
    strategy = config.CHUNKING_STRATEGY.lower()

    for path in Path(directory).rglob("*"):
        if path.suffix.lower() in (".html", ".htm"):
            html = path.read_text(errors="ignore")
            page_id = path.stem  # use filename as page_id for local files
            url = str(path)
            title = path.stem

            if strategy == "docling":
                try:
                    chunks.extend(_docling_to_chunks(html, title, url, page_id))
                    continue
                except Exception as exc:
                    print(f"[ingest] docling failed for {path}: {exc}. Using html chunker.")

            raw = _html_to_sections(html, title, url)
            for c in raw:
                c.setdefault("page_id", page_id)
            chunks.extend(raw)

        elif path.suffix.lower() == ".md":
            raw = _markdown_to_sections(path.read_text(errors="ignore"), path.stem, str(path))
            for c in raw:
                c.setdefault("page_id", path.stem)
            chunks.extend(raw)

        elif path.suffix.lower() == ".txt":
            raw = _plain_to_sections(path.read_text(errors="ignore"), path.stem, str(path))
            for c in raw:
                c.setdefault("page_id", path.stem)
            chunks.extend(raw)

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
            metadata={
                "title": c["title"],
                "section": c["section"],
                "url": c["url"],
                "page_id": c.get("page_id", ""),  # ← ADD THIS
            },
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

    strategy = config.CHUNKING_STRATEGY.lower()
    print(f"[ingest] total chunks to embed: {len(all_chunks)} (chunking_strategy={strategy})")

    store = _get_store(reset=args.reset)
    n = _upsert_chunks(store, all_chunks)
    print(f"[ingest] done — {n} chunks stored in {config.CHROMA_PATH} (collection: {config.CHROMA_COLLECTION})")

    # ── BM25 index ────────────────────────────────────────────────────────────
    if config.ENABLE_BM25:
        print("[ingest] building BM25 index …")
        try:
            from framework.core.bm25_index import BM25Index, save_bm25_index

            valid_chunks = [c for c in all_chunks if c.get("text", "").strip()]
            documents = [c["text"] for c in valid_chunks]
            metadata = [
                {
                    "page_id": c.get("page_id", ""),
                    "title": c.get("title", ""),
                    "section": c.get("section", ""),
                    "url": c.get("url", ""),
                }
                for c in valid_chunks
            ]

            bm25 = BM25Index(documents, metadata)
            save_bm25_index(bm25, config.BM25_INDEX_PATH)
        except ImportError as exc:
            print(
                f"[ingest] BM25 index skipped — {exc}\n"
                "         Install rank-bm25 to enable: pip install rank-bm25"
            )
        except Exception as exc:
            print(f"[ingest] BM25 index build failed: {exc}")
    else:
        print("[ingest] BM25 index skipped (ENABLE_BM25=false)")


if __name__ == "__main__":
    main()
