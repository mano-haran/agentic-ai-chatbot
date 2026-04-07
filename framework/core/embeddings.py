"""
Embedding provider factory.

Controlled by config.EMBEDDING_PROVIDER:
  local  — Loads SentenceTransformer directly from a local models/ directory (air-gapped safe)
  openai — OpenAIEmbeddings (requires OPENAI_API_KEY)

Why we load directly via SentenceTransformer instead of HuggingFaceEmbeddings
-------------------------------------------------------------------------------
LangChain's HuggingFaceEmbeddings passes the model_name string through
huggingface_hub's repo-ID validator before checking the filesystem.  That
validator rejects absolute paths (they contain '/', ':', or '\') with the error:
  "repo id must use alphanumeric chars ... max length is 96"
Loading via SentenceTransformer(path) bypasses this validation entirely — it
accepts any filesystem path, relative or absolute, on both Linux and Windows.

Local model setup (air-gapped / offline environments):
  1. On a connected machine, download and save the model:
       pip install sentence-transformers
       python -c "
       from sentence_transformers import SentenceTransformer
       SentenceTransformer('all-MiniLM-L6-v2').save('models/all-MiniLM-L6-v2')
       "
  2. Copy the models/ directory to the air-gapped machine.
  3. Set EMBEDDING_MODEL=all-MiniLM-L6-v2 in .env (folder name under models/).
  4. Set HF_DATASETS_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 in .env.

Path resolution (cross-platform):
  Model path is resolved as <project_root>/models/<model_name>/ using this
  file's location, so it works on Linux and Windows regardless of the CWD.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_core.embeddings import Embeddings

import config

# Project root: two levels up from framework/core/embeddings.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class _LocalSentenceTransformerEmbeddings(Embeddings):
    """
    Thin LangChain Embeddings wrapper around a locally loaded SentenceTransformer.

    Accepts any filesystem path (absolute or relative) without going through
    HuggingFace Hub's repo-ID validation, making it safe for air-gapped environments
    and for paths that contain OS-specific characters (e.g. Windows drive letters).
    """

    def __init__(self, model_path: str) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_path)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode(text, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached embeddings instance for the configured provider."""
    provider = config.EMBEDDING_PROVIDER.lower()
    model = config.EMBEDDING_MODEL

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")

    if provider == "local":
        model_name = model or "all-MiniLM-L6-v2"

        # Build the absolute path to the model directory.
        # Using pathlib ensures the correct separator on each OS
        # (forward slash on Linux/macOS, backslash on Windows).
        model_path = _PROJECT_ROOT / "models" / model_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model not found at: {model_path}\n"
                "Download the model and place it under the models/ directory.\n"
                "See the docstring in framework/core/embeddings.py for instructions."
            )

        # Block any outbound HuggingFace network calls as a safety net.
        # os.environ.setdefault does not override values already set in .env.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        # Convert Path to a plain string; SentenceTransformer accepts this
        # on both Linux (/path/to/model) and Windows (C:\path\to\model).
        model_path_str = str(model_path)

        print(f"[embeddings] loading local model from: {model_path_str}", flush=True)
        return _LocalSentenceTransformerEmbeddings(model_path_str)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER='{provider}'. "
        "Valid values: local, openai"
    )
