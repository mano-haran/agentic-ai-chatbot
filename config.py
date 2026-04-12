import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Defaults (model ids — must match an id in llm_config.yaml) ────────────────
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "")
DEFAULT_ROUTING_MODEL: str = os.getenv("DEFAULT_ROUTING_MODEL", "")

# ── Token limits ───────────────────────────────────────────────────────────────
# Hard ceiling applied to every agent regardless of per-agent configuration.
# Any per-agent max_tokens value that exceeds this will be silently clamped.
# Adjust to control worst-case token spend across all agents.
MAX_TOKENS_HARD_LIMIT: int = int(os.getenv("MAX_TOKENS_HARD_LIMIT", "4096"))


def clamp_tokens(requested: int) -> int:
    """Return requested tokens clamped to MAX_TOKENS_HARD_LIMIT."""
    return min(requested, MAX_TOKENS_HARD_LIMIT)

# ── Checkpointer ───────────────────────────────────────────────────────────────
APP_ENV: str = os.getenv("APP_ENV", "dev")          # "dev" | "prod"
POSTGRES_URL: str = os.getenv("POSTGRES_URL", "")   # only needed in prod

# ── History compaction ─────────────────────────────────────────────────────────
# Controls how the conversation history is trimmed after each workflow run.
#
#   none    — no trimming; history grows unbounded (default)
#   window  — keep only the last HISTORY_WINDOW_SIZE messages
#   summary — condense older messages into a single summary using an LLM call,
#             then prepend the summary as context for the next turn
#
# "none" is the safe default: it matches the original behaviour and avoids any
# risk of losing context on short conversations.  Switch to "window" or "summary"
# when conversations grow long enough to approach the model's context limit.
HISTORY_STRATEGY: str = os.getenv("HISTORY_STRATEGY", "none")   # none | window | summary
HISTORY_WINDOW_SIZE: int = int(os.getenv("HISTORY_WINDOW_SIZE", "20"))
HISTORY_SUMMARY_MODEL: str = os.getenv("HISTORY_SUMMARY_MODEL", DEFAULT_ROUTING_MODEL)

# ── Logging ────────────────────────────────────────────────────────────────────
# LOG_LEVEL controls which messages are written to LOG_FILE.
#   OFF     — no logging (default)
#   ERROR   — errors only
#   WARNING — warnings + errors
#   INFO    — info + warnings + errors
#   DEBUG   — everything (routing decisions, agent steps, LLM calls, responses)
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "OFF").upper()
LOG_FILE: str = os.getenv("LOG_FILE", "app.log")

# ── Inline agent step visibility ───────────────────────────────────────────────
# Controls whether tool calls are surfaced as collapsible cl.Step blocks in the
# Chainlit chat window (in addition to the high-level TaskList sidebar).
#
#   off     — TaskList only; no inline steps shown (default, cleanest for end users)
#   tools   — show a collapsible step for each tool call (name only, no I/O detail)
#             good for users who want to see what's happening without being overwhelmed
#   verbose — show tool name + truncated input/output; use when debugging workflows
AGENT_STEPS: str = os.getenv("AGENT_STEPS", "off").lower()

# ── Tool integrations ──────────────────────────────────────────────────────────
JENKINS_URL: str = os.getenv("JENKINS_URL", "http://localhost:8080")
JENKINS_USER: str = os.getenv("JENKINS_USER", "")
JENKINS_TOKEN: str = os.getenv("JENKINS_TOKEN", "")

# Confluence Data Center (used by confluence_qa workflow)
CONFLUENCE_URL: str = os.getenv("CONFLUENCE_URL", "")
CONFLUENCE_TOKEN: str = os.getenv("CONFLUENCE_TOKEN", "")   # Personal Access Token

# Vector store (Chroma) for Confluence RAG
CHROMA_PATH: str = os.getenv("CHROMA_PATH", "data/chroma")
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "confluence")

# Embedding provider: "local" (sentence-transformers, no API key needed) or "openai"
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")        # provider-specific default applied in embeddings.py
EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")  # custom API endpoint for OpenAI-compatible providers
EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")    # API key if different from OPENAI_API_KEY

# ── Testing / mock mode ────────────────────────────────────────────────────────
# Set MOCK_JENKINS=true to make Jenkins tools read from tests/mock_data/jenkins/
# instead of calling a real Jenkins server.  No Jenkins server needed.
MOCK_JENKINS: bool = os.getenv("MOCK_JENKINS", "false").lower() == "true"
# Set MOCK_CONFLUENCE=true to make fetch_page_by_id and fetch_confluence_page
# read from tests/mock_data/confluence/ instead of calling the Confluence API.
# find_confluence_page_ids still queries Chroma — run ingest first (see README).
MOCK_CONFLUENCE: bool = os.getenv("MOCK_CONFLUENCE", "false").lower() == "true"
# Base directory for all mock data files (relative to project root or absolute).
MOCK_DATA_DIR: str = os.getenv("MOCK_DATA_DIR", "tests/mock_data")

# ── RAG improvements ───────────────────────────────────────────────────────────
# Chunking strategy for Confluence ingestion (scripts/ingest_confluence.py):
#   html     — BeautifulSoup heading-based sections (default, no extra deps)
#   docling  — docling HybridChunker: context-aware, tables/code as atomic units
CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "html")
# Local tokenizer path for docling HybridChunker.  Leave empty to use docling's
# default; set to a local model directory for air-gapped environments.
DOCLING_TOKENIZER_MODEL: str = os.getenv("DOCLING_TOKENIZER_MODEL", "")

# BM25 hybrid retrieval — adds keyword search alongside vector search.
# Results from both are merged via Reciprocal Rank Fusion (RRF).
# Requires: pip install rank-bm25  (build the index with ingest_confluence.py)
ENABLE_BM25: bool = os.getenv("ENABLE_BM25", "false").lower() == "true"
BM25_INDEX_PATH: str = os.getenv("BM25_INDEX_PATH", "data/bm25.pkl")
RRF_K: int = int(os.getenv("RRF_K", "60"))  # RRF smoothing constant (paper default: 60)

# Reranker — cross-encoder reranking of candidate pages after retrieval:
#   none              — no reranking; use RRF/vector order as-is (default)
#   openai_compatible — POST to RERANKER_BASE_URL/v1/rerank (httpx, no SDK dep)
#   local             — sentence-transformers CrossEncoder (air-gapped safe)
RERANKER_PROVIDER: str = os.getenv("RERANKER_PROVIDER", "none")
RERANKER_BASE_URL: str = os.getenv("RERANKER_BASE_URL", "")
RERANKER_API_KEY: str = os.getenv("RERANKER_API_KEY", "")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "mxbai-rerank-large-v1")
RERANKER_LOCAL_MODEL: str = os.getenv("RERANKER_LOCAL_MODEL", "mixedbread-ai/mxbai-rerank-large-v1")
# Candidate pool size before reranking — more candidates = better recall, slower
RERANKER_CANDIDATE_K: int = int(os.getenv("RERANKER_CANDIDATE_K", "30"))
# Final number of pages returned after reranking
RERANKER_TOP_N: int = int(os.getenv("RERANKER_TOP_N", "5"))


# ── LLM config (llm_config.yaml) ──────────────────────────────────────────────

@dataclass
class ModelResolution:
    """Fully resolved model config ready for the provider factory."""
    model_id: str           # id from llm_config.yaml (used in workflow configs)
    name: str               # actual model name sent to the API
    provider: str           # provider name (e.g. "anthropic")
    client_type: str        # LangChain SDK to use: anthropic | openai | azure_openai | google
    api_key: str            # resolved from .env
    base_url: str           # resolved from .env (may be empty)
    extra: dict = field(default_factory=dict)  # provider-specific extras (e.g. api_version)


def _load_llm_config() -> dict:
    path = Path(__file__).parent / "llm_config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"llm_config.yaml not found at {path}. "
            "Create it from the documented template to register providers and models."
        )
    with open(path) as f:
        return yaml.safe_load(f)


_llm_cfg = _load_llm_config()
_providers: dict[str, dict] = {p["name"]: p for p in _llm_cfg.get("providers", [])}
_models: dict[str, dict] = {m["id"]: m for m in _llm_cfg.get("models", [])}


def resolve_model(model_id: str) -> ModelResolution:
    """
    Look up a model id in llm_config.yaml and return fully resolved credentials.

    Raises ValueError with the list of available ids if the id is not found.
    """
    if model_id not in _models:
        available = ", ".join(sorted(_models.keys()))
        raise ValueError(
            f"Model id '{model_id}' not found in llm_config.yaml. "
            f"Available ids: {available}"
        )

    model_entry = _models[model_id]
    provider_name = model_entry["provider"]

    if provider_name not in _providers:
        raise ValueError(
            f"Provider '{provider_name}' referenced by model '{model_id}' "
            f"is not defined in llm_config.yaml providers list."
        )

    provider_entry = _providers[provider_name]
    api_key  = os.getenv(provider_entry.get("api_key_env", ""), "")
    base_url = os.getenv(provider_entry.get("base_url_env", ""), "")

    extra = {
        key: os.getenv(env_var, "")
        for key, env_var in provider_entry.get("extra_env", {}).items()
    }

    return ModelResolution(
        model_id=model_id,
        name=model_entry["name"],
        provider=provider_name,
        client_type=provider_entry["client_type"],
        api_key=api_key,
        base_url=base_url,
        extra=extra,
    )
