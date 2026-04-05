from typing import Callable, Any
from langchain_core.tools import StructuredTool

# Global registry: tool_name → StructuredTool
_tool_registry: dict[str, StructuredTool] = {}


def tool(name: str = "", description: str = "") -> Callable:
    """
    Decorator that registers a function as a named, LangChain-compatible tool.
    The function remains importable; the module-level name becomes a StructuredTool.

    Usage (Python DSL):
        @tool(description="Fetch Jenkins build logs")
        def fetch_build_log(job_name: str, build_number: int) -> str:
            ...

    Usage in YAML:
        tools:
          - name: fetch_build_log
            module: tools.jenkins
            function: fetch_build_log
    """
    def decorator(fn: Callable) -> StructuredTool:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip()

        structured = StructuredTool.from_function(
            func=fn,
            name=tool_name,
            description=tool_desc,
        )
        _tool_registry[tool_name] = structured
        return structured

    return decorator


def get_tool(name: str) -> StructuredTool:
    if name not in _tool_registry:
        raise KeyError(
            f"Tool '{name}' not registered. Available: {sorted(_tool_registry)}"
        )
    return _tool_registry[name]


def all_tools() -> dict[str, StructuredTool]:
    return dict(_tool_registry)
