import json
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from framework.core.log import logger

# ── Clarification protocol ─────────────────────────────────────────────────────
# Appended to every LLMAgent system prompt so all agents know how to signal that
# they need more information from the user.  The framework strips the marker from
# the stored AIMessage and formats the questions as plain English before display.

CLARIFICATION_INSTRUCTION = """\

────────────────────────────────────────────────────────────────────────────────
CLARIFICATION PROTOCOL
If you cannot complete your task because essential information is missing from
the conversation, write a brief explanation, then end your response with EXACTLY:

CLARIFICATION_NEEDED: {"questions": ["<question 1>", "<question 2>"]}

Rules:
• The CLARIFICATION_NEEDED line must be the very last line of your response.
• Only use it when you genuinely cannot proceed without the missing information.
• Do not ask for information that is already in the conversation.
• If you can proceed (even partially), do so instead of asking.
────────────────────────────────────────────────────────────────────────────────"""

_CLARIFICATION_PREFIX = "CLARIFICATION_NEEDED:"


def _parse_clarification(content: str) -> tuple[str, list[str]]:
    """
    Detect and extract the structured clarification signal from agent output.

    Scans the response bottom-up for a line starting with CLARIFICATION_NEEDED:
    followed by a JSON object.  Returns (clean_content, questions) where
    clean_content has the marker line stripped, and questions is the parsed list.
    Returns (original_content, []) if no valid marker is found.
    """
    lines = content.split("\n")
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line.startswith(_CLARIFICATION_PREFIX):
            json_str = line[len(_CLARIFICATION_PREFIX):].strip()
            try:
                data = json.loads(json_str)
                questions = data.get("questions", [])
                if isinstance(questions, list) and questions:
                    clean = "\n".join(lines[:i]).rstrip()
                    return clean, [str(q) for q in questions]
            except (json.JSONDecodeError, AttributeError):
                pass
            # Found prefix but couldn't parse — treat as no clarification signal
            break
    return content, []


def make_agent_node(agent: "BaseAgent"):
    """
    Factory that produces a LangGraph node function wrapping an agent.
    Used by SequentialAgent, RouterAgent, and ParallelAgent to embed
    sub-agents as nodes inside a parent graph.

    After each run, checks whether the agent signalled that it needs
    clarification by outputting a structured CLARIFICATION_NEEDED marker
    (see CLARIFICATION_INSTRUCTION).  If detected:
      • The marker is stripped from the stored AIMessage so the raw JSON is
        never shown to the user.
      • clarification_needed=True is set in state so SequentialAgent can
        stop the pipeline early.
      • clarification_questions=[...] is populated in state so app.py can
        format the questions as plain English before displaying them.

    Config forwarding
    -----------------
    The node accepts a RunnableConfig second argument which LangGraph injects
    automatically.  Forwarding it to the inner ainvoke() propagates the callback
    context (set up by astream_events on the outer graph) into each agent's
    internal graph, making tool-call events visible to the outer event stream.
    Without this, tool events from inside agent graphs would be invisible to
    Workflow.stream_with_events() and the AGENT_STEPS feature would not work.
    """
    async def node(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        msgs_before = len(state.get("messages", []))
        logger.debug("AGENT", "start", agent=agent.name, msg_count=msgs_before)

        try:
            result = await agent.compile().ainvoke(state, config=config)
        except Exception as exc:
            logger.error("AGENT", f"agent '{agent.name}' raised during ainvoke", exc=exc)
            raise

        # Find the last AI message added during this step.
        new_msgs = result.get("messages", [])[msgs_before:]
        last_ai = next((m for m in reversed(new_msgs) if isinstance(m, AIMessage)), None)

        # has_pending_tool_calls: True when the last AI message is still requesting
        # a tool call (i.e. mid-ReAct-loop).  We only check the terminal response.
        has_pending_tool_calls = bool(getattr(last_ai, "tool_calls", None)) if last_ai else False

        # Normalise content to str (Anthropic returns list-of-blocks for tool-use responses).
        content = last_ai.content if last_ai else ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )

        if last_ai is not None and not has_pending_tool_calls:
            clean_content, questions = _parse_clarification(content)
        else:
            clean_content, questions = content, []

        if questions:
            logger.info("AGENT", "clarification needed", agent=agent.name,
                        questions=questions, response_preview=clean_content[:150])

            # Replace the AIMessage with a clean version (marker stripped) so the
            # raw JSON is never stored in conversation history or shown to the user.
            msgs = result.get("messages", [])
            clean_ai = AIMessage(content=clean_content, id=getattr(last_ai, "id", None))
            msgs_cleaned = [clean_ai if m is last_ai else m for m in msgs]

            result = {
                **result,
                "messages": msgs_cleaned,
                # Also clean task_results so downstream agents don't see the marker.
                "task_results": {
                    **result.get("task_results", {}),
                    agent.name: clean_content,
                },
                "clarification_needed": True,
                "clarification_questions": questions,
            }
        else:
            logger.debug("AGENT", "complete", agent=agent.name, response_preview=content[:150])

        return result

    node.__name__ = agent.name
    return node


class BaseAgent(ABC):
    """
    Base class for all agent types.

    Subclasses implement `_build_graph()` which returns a LangGraph StateGraph.
    The graph is compiled once and cached on first use.

    Two main agent families:
      - LLMAgent    : non-deterministic, LLM decides the next action.
      - WorkflowAgent: deterministic, fixed execution structure
                       (SequentialAgent, ParallelAgent, LoopAgent, RouterAgent).
    """

    def __init__(self, name: str, description: str = "", display_name: str = ""):
        self.name = name
        self.description = description
        self.display_name = display_name or name   # falls back to internal name if not set
        self._graph = None

    @abstractmethod
    def _build_graph(self) -> StateGraph:
        ...

    def compile(self, checkpointer=None):
        if checkpointer is not None:
            # Always build fresh when a checkpointer is supplied — never return
            # a cached graph that was compiled without one.
            return self._build_graph().compile(checkpointer=checkpointer)
        if self._graph is None:
            self._graph = self._build_graph().compile()
        return self._graph

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        return await self.compile().ainvoke(state)

    async def stream_events(self, state: dict[str, Any]) -> AsyncIterator:
        async for event in self.compile().astream_events(state, version="v2"):
            yield event
