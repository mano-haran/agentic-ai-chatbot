"""
Embedding provider factory.

Controlled by config.EMBEDDING_PROVIDER:
  local  — HuggingFaceEmbeddings loaded from a local model directory (air-gapped safe)
  openai — OpenAIEmbeddings (requires OPENAI_API_KEY)

Local model setup (air-gapped / offline environments):
  1. Download the model on a connected machine:
       pip install sentence-transformers
       python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('models/all-MiniLM-L6-v2')"
  2. Copy the models/ directory to the target machine.
  3. Set EMBEDDING_MODEL=all-MiniLM-L6-v2 in .env (must match the folder name under models/).
  4. Set HF_DATASETS_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 to prevent any outbound calls.

Path resolution (cross-platform):
  Models are stored under <project_root>/models/<model_name>/.
  The project root is derived from this file's location (framework/core/embeddings.py),
  so the path works correctly on both Linux and Windows regardless of the working directory.
"""

import os
from functools import lru_cache
from pathlib import Path
from langchain_core.embeddings import Embeddings

import config

# Project root: two levels up from framework/core/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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

        # Resolve the model path from the local models/ directory.
        # Path() handles the separator difference between Linux (/) and Windows (\)
        # automatically, so no manual os.sep handling is needed.
        model_path = _PROJECT_ROOT / "models" / model_name

        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model not found at: {model_path}\n"
                "Download the model and place it under the models/ directory.\n"
                "See the docstring in framework/core/embeddings.py for instructions."
            )

        # Ensure HuggingFace libraries do not attempt any network calls.
        # These are set here as a safety net in addition to .env so the
        # application stays air-gapped even if .env is not configured.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        # str() converts the Path to the OS-native format (backslashes on Windows)
        # which sentence-transformers expects.
        local_path_str = str(model_path)

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        print(f"[embeddings] loading local model from: {local_path_str}", flush=True)
        return HuggingFaceEmbeddings(model_name=local_path_str)

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER='{provider}'. "
        "Valid values: local, openai"
    )
