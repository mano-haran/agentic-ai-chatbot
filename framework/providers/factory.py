"""
LLM provider factory.

All agents resolve a model id through llm_config.yaml, which maps each id
to a provider and its credentials.  The factory dispatches to the correct
LangChain SDK based on the provider's client_type.

Supported client_types
-----------------------
anthropic    ChatAnthropic
openai       ChatOpenAI  (also covers any OpenAI-compatible endpoint via base_url)
azure_openai AzureChatOpenAI
google       ChatGoogleGenerativeAI
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

import config
from config import ModelResolution
from framework.providers.token_logger import TokenUsageLogger

# Single shared instance — stateless, safe to reuse across all LLM calls.
_token_logger = TokenUsageLogger()


def get_llm(
    model_id: str,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
) -> BaseChatModel:
    """
    Return a configured LangChain chat model for the given model id.

    The model id must match an entry in llm_config.yaml.  Provider credentials
    are resolved automatically from the .env variables named in that file.

    Args:
        model_id:    Id from llm_config.yaml (e.g. "claude-sonnet-4-6", "gpt-4o").
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens:  Maximum tokens in the response.
        **kwargs:    Extra kwargs forwarded to the underlying LangChain class.

    Raises:
        ValueError: If model_id is not in llm_config.yaml, with a list of
                    available ids to help the user correct the config.
    """
    resolution = config.resolve_model(model_id)

    match resolution.client_type:
        case "anthropic":
            return _build_anthropic(resolution, temperature, max_tokens, **kwargs)
        case "openai":
            return _build_openai(resolution, temperature, max_tokens, **kwargs)
        case "azure_openai":
            return _build_azure(resolution, temperature, max_tokens, **kwargs)
        case "google":
            return _build_google(resolution, temperature, max_tokens, **kwargs)
        case _:
            raise ValueError(
                f"Unknown client_type '{resolution.client_type}' for provider "
                f"'{resolution.provider}'. "
                f"Valid values: anthropic, openai, azure_openai, google."
            )


# ── Per-client-type builders ───────────────────────────────────────────────────

def _build_anthropic(r: ModelResolution, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise ImportError("Run: pip install langchain-anthropic") from e

    init: dict = dict(model=r.name, temperature=temperature, max_tokens=max_tokens,
                      callbacks=[_token_logger], **kwargs)
    if r.api_key:
        init["api_key"] = r.api_key
    if r.base_url:
        init["anthropic_api_url"] = r.base_url

    return ChatAnthropic(**init)


def _build_openai(r: ModelResolution, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        raise ImportError("Run: pip install langchain-openai") from e

    init: dict = dict(model=r.name, temperature=temperature, max_tokens=max_tokens,
                      callbacks=[_token_logger], **kwargs)
    if r.api_key:
        init["api_key"] = r.api_key
    if r.base_url:
        init["base_url"] = r.base_url

    return ChatOpenAI(**init)


def _build_azure(r: ModelResolution, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
    """
    Azure OpenAI Service.
    Required .env vars (referenced via llm_config.yaml):
        api_key_env  → AZURE_OPENAI_API_KEY
        base_url_env → AZURE_OPENAI_ENDPOINT  (e.g. https://<resource>.openai.azure.com/)
        extra_env.api_version → AZURE_OPENAI_API_VERSION  (e.g. 2024-02-01)
    `name` in llm_config.yaml is the deployment name in Azure.
    """
    try:
        from langchain_openai import AzureChatOpenAI
    except ImportError as e:
        raise ImportError("Run: pip install langchain-openai") from e

    return AzureChatOpenAI(
        azure_deployment=r.name,
        azure_endpoint=r.base_url,
        api_key=r.api_key,
        api_version=r.extra.get("api_version", "2024-02-01"),
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=[_token_logger],
        **kwargs,
    )


def _build_google(r: ModelResolution, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise ImportError("Run: pip install langchain-google-genai") from e

    init: dict = dict(model=r.name, temperature=temperature, max_output_tokens=max_tokens,
                      callbacks=[_token_logger], **kwargs)
    if r.api_key:
        init["google_api_key"] = r.api_key
    if r.base_url:
        init["base_url"] = r.base_url

    return ChatGoogleGenerativeAI(**init)
