"""
Token usage logger — LangChain callback that prints token counts to stdout
after every LLM call, across all supported providers.

Attached automatically to every LLM created by factory.get_llm().

Output format (one line per call):
    [token_usage] model=claude-sonnet-4-6 input=312 output=87 total=399
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class TokenUsageLogger(BaseCallbackHandler):
    """
    Logs token usage to stdout after every LLM invocation.

    Provider-specific response shapes handled:

      OpenAI / Azure OpenAI
        llm_output = {"token_usage": {"prompt_tokens": N, "completion_tokens": N,
                                       "total_tokens": N}, "model_name": "..."}

      Anthropic
        llm_output = {"usage": {"input_tokens": N, "output_tokens": N},
                      "model": "..."}

      Google Gemini
        llm_output = {"usage_metadata": {"prompt_token_count": N,
                                         "candidates_token_count": N,
                                         "total_token_count": N}}

    All shapes are normalised to input/output/total before printing.
    """

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        llm_output = response.llm_output or {}

        # ── Extract usage dict (varies by provider) ────────────────────────
        # OpenAI: {"token_usage": {...}}
        # Anthropic: {"usage": {...}}
        # Google: {"usage_metadata": {...}}
        usage = (
            llm_output.get("token_usage")
            or llm_output.get("usage")
            or llm_output.get("usage_metadata")
            or {}
        )

        if not usage:
            return

        # ── Normalise field names ──────────────────────────────────────────
        # OpenAI uses prompt_tokens / completion_tokens
        # Anthropic uses input_tokens / output_tokens
        # Google uses prompt_token_count / candidates_token_count
        input_tokens = (
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("prompt_token_count")
            or "?"
        )
        output_tokens = (
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidates_token_count")
            or "?"
        )
        total_tokens = (
            usage.get("total_tokens")
            or usage.get("total_token_count")
            or (
                input_tokens + output_tokens
                if isinstance(input_tokens, int) and isinstance(output_tokens, int)
                else "?"
            )
        )

        # ── Extract model name (varies by provider) ────────────────────────
        model = (
            llm_output.get("model_name")   # OpenAI
            or llm_output.get("model")     # Anthropic
            or "unknown"
        )

        print(
            f"[token_usage] model={model} "
            f"input={input_tokens} output={output_tokens} total={total_tokens}",
            flush=True,
        )
