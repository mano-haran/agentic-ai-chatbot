import asyncio
from typing import Any
from langgraph.graph import StateGraph, END

from framework.agents.base import BaseAgent, make_agent_node
from framework.core.state import AgentState


class SequentialAgent(BaseAgent):
    """
    Deterministic workflow agent that runs sub-agents sequentially.

    Runs sub-agents one after another in the declared order.
    Shared state (messages + task_results) accumulates so each agent has
    full context from all prior agents in the chain.

    Graph shape:  sub_agent_0 → sub_agent_1 → … → sub_agent_N → END
    """

    def __init__(self, name: str, sub_agents: list[BaseAgent], description: str = "", display_name: str = ""):
        super().__init__(name, description, display_name)
        if not sub_agents:
            raise ValueError(f"SequentialAgent '{name}' requires at least one sub-agent.")
        self.sub_agents = sub_agents

    def _build_graph(self) -> StateGraph:
        g = StateGraph(AgentState)

        for agent in self.sub_agents:
            g.add_node(agent.name, make_agent_node(agent))

        g.set_entry_point(self.sub_agents[0].name)

        for i in range(len(self.sub_agents) - 1):
            current_name = self.sub_agents[i].name
            next_name = self.sub_agents[i + 1].name

            # Stop early if the current step needs clarification or hit an error
            def make_condition(nxt: str):
                def condition(state: AgentState) -> str:
                    if state.get("clarification_needed") or state.get("error"):
                        return END
                    return nxt
                return condition

            g.add_conditional_edges(
                current_name,
                make_condition(next_name),
                {next_name: next_name, END: END},
            )

        g.add_edge(self.sub_agents[-1].name, END)
        return g


class ParallelAgent(BaseAgent):
    """
    Deterministic workflow agent that runs all sub-agents concurrently.

    Runs all sub-agents concurrently using asyncio.gather.
    Results are merged into task_results keyed by agent name.
    Messages from parallel branches are NOT merged (to avoid incoherent
    conversation history); use task_results to access individual outputs.

    Graph shape:  parallel_node → END
                  (fan-out/fan-in happens inside the node via asyncio)
    """

    def __init__(self, name: str, sub_agents: list[BaseAgent], description: str = "", display_name: str = ""):
        super().__init__(name, description, display_name)
        if not sub_agents:
            raise ValueError(f"ParallelAgent '{name}' requires at least one sub-agent.")
        self.sub_agents = sub_agents

    def _build_graph(self) -> StateGraph:
        agents = self.sub_agents  # capture

        async def parallel_node(state: AgentState) -> dict[str, Any]:
            results = await asyncio.gather(
                *[a.compile().ainvoke(state) for a in agents],
                return_exceptions=True,
            )
            merged: dict[str, Any] = {}
            errors: list[str] = []
            for result in results:
                if isinstance(result, Exception):
                    errors.append(str(result))
                else:
                    merged.update(result.get("task_results", {}))
            return {
                "task_results": merged,
                "error": "; ".join(errors) if errors else None,
            }

        g = StateGraph(AgentState)
        g.add_node("parallel", parallel_node)
        g.set_entry_point("parallel")
        g.add_edge("parallel", END)
        return g


class LoopAgent(BaseAgent):
    """
    Deterministic workflow agent that repeatedly runs a sub-agent until a condition is met.

    Re-runs a single sub-agent until either:
      • max_iterations is reached, or
      • the sub-agent writes metadata["done"] = True into state.

    Graph shape:  sub_agent → check → sub_agent (loop) or END
    """

    def __init__(
        self,
        name: str,
        sub_agent: BaseAgent,
        max_iterations: int = 5,
        description: str = "",
        display_name: str = "",
    ):
        super().__init__(name, description, display_name)
        self.sub_agent = sub_agent
        self.max_iterations = max_iterations

    def _build_graph(self) -> StateGraph:
        counter_key = f"_loop_{self.name}_iter"
        sub = self.sub_agent
        max_iter = self.max_iterations

        async def loop_node(state: AgentState) -> dict[str, Any]:
            result = await sub.compile().ainvoke(state)
            iters = state.get("metadata", {}).get(counter_key, 0) + 1
            result_meta = result.get("metadata", {})
            return {**result, "metadata": {**result_meta, counter_key: iters}}

        def should_continue(state: AgentState) -> str:
            iters = state.get("metadata", {}).get(counter_key, 0)
            done = state.get("metadata", {}).get("done", False)
            return END if (done or iters >= max_iter) else "loop"

        g = StateGraph(AgentState)
        g.add_node("loop", loop_node)
        g.set_entry_point("loop")
        g.add_conditional_edges("loop", should_continue, {"loop": "loop", END: END})
        return g
