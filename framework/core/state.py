from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


def _merge_dicts(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Reducer that merges two dicts, with new values taking precedence."""
    return {**(old or {}), **(new or {})}


def _or_bool(a: bool, b: bool) -> bool:
    """Reducer that latches True — once a step needs clarification it stays set."""
    return bool(a) or bool(b)


class AgentState(TypedDict):
    """
    Shared state that flows through the entire agent graph.

    - messages:                Full conversation history (accumulated via add_messages reducer).
    - next_agent:              Routing signal written by RouterAgent/SupervisorAgent.
    - task_results:            Named outputs keyed by agent name (merged across agents).
    - metadata:                Arbitrary session-level key/value data (merged across agents).
    - error:                   Non-None when a node wants to signal a failure upstream.
    - clarification_needed:    Set True by a step that needs user input before proceeding.
                               SequentialAgent stops execution when this is True.
    - clarification_questions: Questions the agent needs answered before it can proceed.
                               Populated by make_agent_node() when the agent outputs the
                               CLARIFICATION_NEEDED structured signal; displayed to the user
                               as plain English by app.py.
    """
    messages: Annotated[list, add_messages]
    next_agent: str
    task_results: Annotated[dict[str, Any], _merge_dicts]
    metadata: Annotated[dict[str, Any], _merge_dicts]
    error: str | None
    clarification_needed: Annotated[bool, _or_bool]
    clarification_questions: list[str]
