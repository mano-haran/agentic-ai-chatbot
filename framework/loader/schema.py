from pydantic import BaseModel, Field
from typing import Literal, Optional

import config


class ToolRef(BaseModel):
    """Points to a tool function in a Python module."""
    name: str
    module: str       # importable module path, e.g. "tools.jenkins"
    function: str     # attribute name in that module


class AgentSchema(BaseModel):
    name: str
    type: Literal["llm", "sequential", "parallel", "loop", "router"] = "llm"

    # LLMAgent fields
    role: Optional[str] = None
    model: str = config.DEFAULT_MODEL
    max_tokens: int = config.MAX_TOKENS_HARD_LIMIT
    tools: list[str] = Field(default_factory=list)      # tool names from ToolRef list
    max_iterations: int = 10
    temperature: float = 0.0

    # WorkflowAgent / RouterAgent fields
    sub_agents: list[str] = Field(default_factory=list)  # agent names (resolved by loader)

    description: str = ""


class IntentSchema(BaseModel):
    """Maps a regex pattern to a workflow name for fast intent routing."""
    pattern: str
    workflow: str


class WorkflowFileSchema(BaseModel):
    """Top-level schema for a workflow YAML file."""
    name: str
    description: str
    version: str = "1.0"
    entry_agent: str                                        # name of the root agent
    tools: list[ToolRef] = Field(default_factory=list)
    agents: list[AgentSchema] = Field(default_factory=list)
    intents: list[IntentSchema] = Field(default_factory=list)
