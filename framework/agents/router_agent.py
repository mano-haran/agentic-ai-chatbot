from typing import Any
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

import config
from framework.agents.base import BaseAgent, make_agent_node
from framework.core.state import AgentState
from framework.providers.factory import get_llm


class RouterAgent(BaseAgent):
    """
    Hybrid agent: deterministic graph structure, LLM-driven routing decision.

    Classifies the user's intent with a single LLM call and routes to exactly
    one sub-agent.  Unlike SupervisorAgent there is no feedback loop — the
    routing decision is one-shot and final.

    Graph shape:  router_node → (conditional) → chosen_sub_agent → END

    Typical use: top-level dispatcher that fans out to specialised workflows,
    or an intra-workflow branch selector.

    Args:
        model:    Any model string supported by the factory (e.g. "gpt-4o-mini").
                  A small/fast model is recommended — the task is simple classification.
        provider: Explicit provider override. Inferred from model name when omitted.
    """

    def __init__(
        self,
        name: str,
        sub_agents: list[BaseAgent],
        model: str = config.DEFAULT_ROUTING_MODEL,
        description: str = "",
        display_name: str = "",
    ):
        super().__init__(name, description, display_name)
        if not sub_agents:
            raise ValueError(f"RouterAgent '{name}' requires at least one sub-agent.")
        self.sub_agents = sub_agents
        self.model = model

    def _build_graph(self) -> StateGraph:
        agent_map = {a.name: a for a in self.sub_agents}
        default_route = self.sub_agents[0].name
        agent_list = "\n".join(
            f"  - {a.name}: {a.description}" for a in self.sub_agents
        )
        llm = get_llm(model_id=self.model, temperature=0.0, max_tokens=64)

        def router_node(state: AgentState) -> dict[str, Any]:
            last_msg = state["messages"][-1].content if state["messages"] else ""
            prompt = (
                f"Select the most appropriate agent for the following request.\n\n"
                f"Available agents:\n{agent_list}\n\n"
                f"Request: {last_msg}\n\n"
                f"Reply with ONLY the agent name — nothing else."
            )
            response = llm.invoke([HumanMessage(content=prompt)])
            chosen = response.content.strip()
            if chosen not in agent_map:
                chosen = default_route
            return {"next_agent": chosen, "metadata": {"route": chosen}}

        def get_route(state: AgentState) -> str:
            return state.get("metadata", {}).get("route", default_route)

        g = StateGraph(AgentState)
        g.add_node("router", router_node)

        for agent in self.sub_agents:
            g.add_node(agent.name, make_agent_node(agent))
            g.add_edge(agent.name, END)

        g.set_entry_point("router")
        g.add_conditional_edges(
            "router",
            get_route,
            {a.name: a.name for a in self.sub_agents},
        )
        return g
