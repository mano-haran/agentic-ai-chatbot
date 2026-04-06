"""
Embedding provider factory.

Controlled by config.EMBEDDING_PROVIDER:
  local  — HuggingFaceEmbeddings (sentence-transformers, no API key required)
  openai — OpenAIEmbeddings (requires OPENAI_API_KEY)

Default model per provider:
  local  → all-MiniLM-L6-v2  (fast, 384-dim, good general-purpose quality)
  openai → text-embedding-3-small

Override with config.EMBEDDING_MODEL.
"""

from functools import lru_cache
from langchain_core.embeddings import Embeddings

import config


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached embeddings instance for the configured provider."""
    provider = config.EMBEDDING_PROVIDER.lower()
    model = config.EMBEDDING_MODEL

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model or "text-embedding-3-small")

    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=model or "all-MiniLM-L6-v2")

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER='{provider}'. "
        "Valid values: local, openai"
    )
