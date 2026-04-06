import importlib
from pathlib import Path
from typing import Any

import yaml

from framework.agents.base import BaseAgent
from framework.agents.llm_agent import LLMAgent
from framework.agents.workflow_agents import SequentialAgent, ParallelAgent, LoopAgent
from framework.agents.router_agent import RouterAgent
from framework.loader.schema import AgentSchema, WorkflowFileSchema
from framework.workflow.workflow import Workflow


class YAMLLoader:
    """
    Loads a Workflow from a YAML file.

    The YAML schema mirrors the Python DSL 1-to-1 so switching between
    the two authoring styles requires minimal effort.

    Resolution order for agent dependencies:
      The loader performs a recursive depth-first build so sub-agents are
      always constructed before their parent (no topological sort required).
    """

    def load(self, path: str | Path) -> Workflow:
        with open(path) as f:
            raw = yaml.safe_load(f)

        schema = WorkflowFileSchema(**raw)

        # 1. Import and register tools declared in the file
        loaded_tools: dict[str, Any] = {}
        for ref in schema.tools:
            mod = importlib.import_module(ref.module)
            obj = getattr(mod, ref.function)
            loaded_tools[ref.name] = obj

        # 2. Build agent objects (recursive, depth-first)
        built: dict[str, BaseAgent] = {}
        for agent_schema in schema.agents:
            if agent_schema.name not in built:
                self._build(agent_schema, schema.agents, loaded_tools, built)

        # 3. Assemble Workflow
        entry = built[schema.entry_agent]
        return Workflow(
            name=schema.name,
            display_name=schema.display_name,
            description=schema.description,
            entry_agent=entry,
            intents=[i.model_dump() for i in schema.intents],
        )

    # ------------------------------------------------------------------

    def _build(
        self,
        schema: AgentSchema,
        all_schemas: list[AgentSchema],
        tools: dict[str, Any],
        built: dict[str, BaseAgent],
    ) -> None:
        # Ensure sub-agents are built first
        sub_agents: list[BaseAgent] = []
        for sa_name in schema.sub_agents:
            if sa_name not in built:
                sa_schema = next(s for s in all_schemas if s.name == sa_name)
                self._build(sa_schema, all_schemas, tools, built)
            sub_agents.append(built[sa_name])

        agent_tools = [tools[t] for t in schema.tools if t in tools]

        agent: BaseAgent
        match schema.type:
            case "llm":
                agent = LLMAgent(
                    name=schema.name,
                    display_name=schema.display_name,
                    role=schema.role or "",
                    model=schema.model,
                    tools=agent_tools,
                    sub_agents=sub_agents,
                    max_iterations=schema.max_iterations,
                    description=schema.description,
                    temperature=schema.temperature,
                    max_tokens=schema.max_tokens,
                )
            case "sequential":
                agent = SequentialAgent(
                    name=schema.name,
                    display_name=schema.display_name,
                    sub_agents=sub_agents,
                    description=schema.description,
                )
            case "parallel":
                agent = ParallelAgent(
                    name=schema.name,
                    display_name=schema.display_name,
                    sub_agents=sub_agents,
                    description=schema.description,
                )
            case "loop":
                if not sub_agents:
                    raise ValueError(f"LoopAgent '{schema.name}' needs exactly one sub-agent.")
                agent = LoopAgent(
                    name=schema.name,
                    display_name=schema.display_name,
                    sub_agent=sub_agents[0],
                    max_iterations=schema.max_iterations,
                    description=schema.description,
                )
            case "router":
                agent = RouterAgent(
                    name=schema.name,
                    display_name=schema.display_name,
                    sub_agents=sub_agents,
                    model=schema.model,
                    description=schema.description,
                )
            case _:
                raise ValueError(f"Unknown agent type '{schema.type}' for agent '{schema.name}'.")

        built[schema.name] = agent
