from abc import ABC, abstractmethod
from typing import Any, AsyncIterator
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

from framework.core.log import logger

# Phrases that indicate the agent is requesting more input from the user.
# Used as a fallback when the response doesn't contain a literal "?".
# LLMs sometimes phrase clarification requests as statements ("Please provide …")
# rather than questions, so checking only for "?" is too fragile.
_CLARIFICATION_PHRASES = (
    "please provide",
    "could you provide",
    "can you provide",
    "please share",
    "please give me",
    "i need a jenkins",
    "i need the jenkins",
    "i'll need",
    "i will need",
    "no jenkins url",
    "no url",
    "missing url",
)


def _needs_clarification(content: str) -> bool:
    """Return True if the content signals the agent needs more user input."""
    if "?" in content:
        return True
    lower = content.lower()
    return any(phrase in lower for phrase in _CLARIFICATION_PHRASES)


def make_agent_node(agent: "BaseAgent"):
    """
    Factory that produces a LangGraph node function wrapping an agent.
    Used by SequentialAgent, RouterAgent, and ParallelAgent to embed
    sub-agents as nodes inside a parent graph.

    After each run, detects whether the agent is asking for clarification
    and sets clarification_needed=True in state so SequentialAgent can stop early.

    Clarification detection
    -----------------------
    We look at the FINAL AI message produced during this step and check two things:
      1. It has no pending tool_calls — meaning the agent is done with tools and
         is giving its terminal response (not mid-way through a ReAct loop).
      2. Its content contains a question mark.

    The previous check also required that NO tools were called at all during the
    step.  That was too strict: an agent with tools (e.g. log_fetcher_agent) might
    call get_jenkins_builds, receive an error because no job name was given, and
    then ask "Please provide the job name?"  Since a ToolMessage exists, the old
    check silently passed clarification_needed=False, and the next agent in the
    SequentialAgent would start running — even though the first step never got the
    information it needed.

    The corrected check ignores whether tools were called during the step and
    focuses only on the nature of the terminal response: if the agent's last word
    is a question, it needs more input before the workflow can continue.

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
            # Forward the LangGraph config so callbacks (including those registered
            # by astream_events) propagate into the agent's internal graph.
            result = await agent.compile().ainvoke(state, config=config)
        except Exception as exc:
            logger.error("AGENT", f"agent '{agent.name}' raised during ainvoke", exc=exc)
            raise

        # Find the last AI message added during this step.
        new_msgs = result.get("messages", [])[msgs_before:]
        last_ai = next((m for m in reversed(new_msgs) if isinstance(m, AIMessage)), None)

        # has_pending_tool_calls: True when the last AI message is still requesting
        # a tool call (i.e. mid-ReAct-loop).  False when the agent has settled on
        # a final response (with or without having used tools earlier in the step).
        has_pending_tool_calls = bool(getattr(last_ai, "tool_calls", None)) if last_ai else False

        # content may be a list (Anthropic content blocks) — normalise to str for logging.
        content = last_ai.content if last_ai else ""
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )

        # Detect clarification: the agent's final response asks for more input.
        # We check for "?" AND common "please provide" style phrases because LLMs
        # sometimes phrase clarification requests as statements rather than questions,
        # which means a "?" check alone misses them and the pipeline continues
        # incorrectly.
        # We intentionally do NOT check whether tools were called during the step —
        # a tool-using agent that couldn't complete its task (e.g. missing URL)
        # will call tools, hit an error, and then ask the user a question.  That
        # final request still signals clarification_needed even though tools ran.
        clarification = (
            agent.check_clarification
            and last_ai is not None
            and not has_pending_tool_calls
            and _needs_clarification(content)
        )

        if clarification:
            logger.info("AGENT", "clarification needed", agent=agent.name, response_preview=content[:150])
            result = {**result, "clarification_needed": True}
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

    def __init__(self, name: str, description: str = "", display_name: str = "", check_clarification: bool = True):
        self.name = name
        self.description = description
        self.display_name = display_name or name   # falls back to internal name if not set
        self.check_clarification = check_clarification
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
