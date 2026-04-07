from typing import Any
from pydantic import BaseModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

import config
from framework.agents.base import BaseAgent
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
        max_iterations: int = 10,
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

        system_msg = SystemMessage(content=self.role)

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
            return {
                "messages": [response],
                "next_agent": self.name,
                "task_results": {self.name: content},
            }

        def should_continue(state: AgentState) -> str:
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
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
