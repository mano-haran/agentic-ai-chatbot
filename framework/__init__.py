"""
Framework public API — simple, declarative interface for defining agentic workflows.

Python DSL quick-start:
    from framework import LLMAgent, SequentialAgent, tool, Workflow

    @tool(description="My custom tool")
    def my_tool(arg: str) -> str: ...

    agent = LLMAgent(name="my_agent", role="You are ...", tools=[my_tool])
    wf = Workflow(name="my_workflow", description="...", entry_agent=agent)

YAML quick-start:
    from framework import YAMLLoader
    workflow = YAMLLoader().load("workflows/my_workflow/workflow.yaml")
"""

from framework.agents.llm_agent import LLMAgent
from framework.agents.workflow_agents import SequentialAgent, ParallelAgent, LoopAgent
from framework.agents.router_agent import RouterAgent
from framework.tools.decorators import tool
from framework.workflow.workflow import Workflow
from framework.workflow.intent_router import IntentRouter
from framework.loader.yaml_loader import YAMLLoader
from framework.providers.factory import get_llm

__all__ = [
    # Agent types
    "LLMAgent",
    "SequentialAgent",
    "ParallelAgent",
    "LoopAgent",
    "RouterAgent",
    # DSL helpers
    "tool",
    # Workflow layer
    "Workflow",
    "IntentRouter",
    # YAML loader
    "YAMLLoader",
    # LLM factory (advanced use — direct model access)
    "get_llm",
]
