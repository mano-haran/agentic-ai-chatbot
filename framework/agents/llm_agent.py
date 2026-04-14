import re
from typing import Any
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Some LLM backends (e.g. Phi-3 variants) emit internal control tokens such as
# <|end|>, <|start|>, <|channel|> inside the response text.  If these reach the
# context history, subsequent API calls fail with formatting/validation errors.
_ARTIFACT_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def _strip_artifact_tokens(text: str) -> str:
    """Remove internal LLM artifact tokens (e.g. <|end|>, <|start|>) from text."""
    return _ARTIFACT_TOKEN_RE.sub("", text).strip()


import config
from framework.agents.base import BaseAgent, CLARIFICATION_INSTRUCTION
from framework.core.state import AgentState
from framework.providers.factory import get_llm


class _AgentInput(BaseModel):
    """Pydantic schema for sub-agent tool invocation."""
    input: str


class LLMAgent(BaseAgent):
    """
    Non-deterministic agent that implements the ReAct loop:  agent → tools → agent → … → END
    Tools and sub-agents are both presented to the LLM as callable tools.
    Sub-agents are wrapped transparently — the LLM cannot tell the difference.

    Args:
        name:           Unique agent identifier.
        role:           System-prompt / persona injected on every LLM call.
        model:          Model identifier — any string accepted by the provider
                        (e.g. "gpt-4o", "claude-sonnet-4-6", "gemini-2.0-flash").
        provider:       "openai" | "azure" | "anthropic" | "google".
                        Inferred from model name when omitted.
        tools:          List of StructuredTool objects (from @tool decorator).
        sub_agents:     Child BaseAgent instances exposed to the LLM as tools.
        max_iterations: Safety cap on the ReAct loop.
        description:    One-liner used by RouterAgent and IntentRouter for matching.
        temperature:    LLM sampling temperature.
    """

    def __init__(
        self,
        name: str,
        role: str,
        model: str = config.DEFAULT_MODEL,
        tools: list[StructuredTool] | None = None,
        sub_agents: list[BaseAgent] | None = None,
        max_iterations: int = 3,
        description: str = "",
        display_name: str = "",
        temperature: float = 0.0,
        max_tokens: int = config.MAX_TOKENS_HARD_LIMIT,
    ):
        super().__init__(name, description or role[:120], display_name)
        self.role = role
        self.model = model
        self._tools: list[StructuredTool] = tools or []
        self.sub_agents: list[BaseAgent] = sub_agents or []
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = config.clamp_tokens(max_tokens)

    # ------------------------------------------------------------------
    # Sub-agent → tool wrapping
    # ------------------------------------------------------------------

    def _wrap_sub_agent(self, agent: BaseAgent) -> StructuredTool:
        """
        Wraps a sub-agent as a StructuredTool so the LLM can invoke it by name.
        The tool accepts a free-text `input` string and returns the sub-agent's
        last AI message as a string.
        """
        agent_ref = agent  # capture for closure

        async def _invoke(input: str) -> str:  # noqa: A002
            from langchain_core.messages import HumanMessage
            result = await agent_ref.compile().ainvoke({
                "messages": [HumanMessage(content=input)],
                "next_agent": "",
                "task_results": {},
                "metadata": {},
                "error": None,
                "clarification_needed": False,
                "clarification_questions": [],
            })
            msgs = result.get("messages", [])
            return msgs[-1].content if msgs else "(no response)"

        return StructuredTool(
            name=agent.name,
            description=agent.description or f"Delegate task to the {agent.name} agent.",
            coroutine=_invoke,
            args_schema=_AgentInput,
        )

    def _all_tools(self) -> list[StructuredTool]:
        return self._tools + [self._wrap_sub_agent(sa) for sa in self.sub_agents]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        all_tools = self._all_tools()
        llm = get_llm(
            model_id=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if all_tools:
            llm = llm.bind_tools(all_tools)

        # Append the clarification protocol to the agent's role so every agent
        # knows exactly how to signal that it needs more information from the user.
        # The framework (make_agent_node) detects and strips the structured marker
        # before the response reaches the user.
        system_msg = SystemMessage(content=self.role + CLARIFICATION_INSTRUCTION)

        # Per-agent iteration counter stored in shared state metadata. Enforcing
        # this cap is what keeps the ReAct loop from burning tokens when an
        # over-eager LLM keeps emitting tool_calls.  The key is unique per agent
        # name so nested/sibling agents don't interfere.
        iter_key = f"_react_{self.name}_iter"
        max_iter = self.max_iterations

        # Async node so we stay on the event loop and avoid thread-executor
        # issues that can cause silent failures when llm.invoke() is called
        # from a sync function nested inside ainvoke().
        async def agent_node(state: AgentState) -> dict[str, Any]:
            response = await llm.ainvoke([system_msg] + state["messages"])
            # content may be a list of blocks (e.g. Anthropic tool-use response).
            # Normalise to a plain string for task_results so downstream code
            # that expects a string (e.g. logging, clarification detection) works.
            content = response.content
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                    if not (isinstance(b, dict) and b.get("type") == "tool_use")
                )
            # Strip artifact tokens from the normalised string.
            content = _strip_artifact_tokens(content)
            # Also sanitise response.content so the AIMessage stored in the
            # messages history is clean — contaminated history causes subsequent
            # LLM calls to fail with formatting errors.
            if isinstance(response.content, str):
                response.content = _strip_artifact_tokens(response.content)
            elif isinstance(response.content, list):
                for block in response.content:
                    if isinstance(block, dict) and "text" in block:
                        block["text"] = _strip_artifact_tokens(block["text"])

            # Increment the per-agent ReAct iteration counter so should_continue
            # can enforce max_iterations.
            prior = (state.get("metadata") or {}).get(iter_key, 0)
            return {
                "messages": [response],
                "next_agent": self.name,
                "task_results": {self.name: content},
                "metadata": {iter_key: prior + 1},
            }

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            has_tool_calls = bool(getattr(last, "tool_calls", None))
            iters = (state.get("metadata") or {}).get(iter_key, 0)
            # ``iters`` counts completed agent_node runs, so max_iterations=N
            # means "allow at most N rounds of tool calls".  The (N+1)-th LLM
            # call synthesises the final answer from tool results; if that
            # call still requests more tools we END anyway to cap token spend.
            if has_tool_calls and iters <= max_iter:
                return "tools"
            return END

        g = StateGraph(AgentState)
        g.add_node("agent", agent_node)

        if all_tools:
            g.add_node("tools", ToolNode(all_tools))
            g.set_entry_point("agent")
            g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
            g.add_edge("tools", "agent")
        else:
            g.set_entry_point("agent")
            g.add_edge("agent", END)

        return g
