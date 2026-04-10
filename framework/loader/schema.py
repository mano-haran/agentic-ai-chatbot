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
    display_name: str = ""   # human-readable label shown in the task list UI;
                             # falls back to `name` if omitted

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

    # When False, the clarification-detection heuristic is skipped for this
    # agent.  Set to False for pipeline-middle agents that output structured or
    # verbatim content (e.g. RAG search results, fetched page content) that may
    # legitimately contain "?" characters without the agent asking the user for
    # more input.  Defaults to True so existing agents are unaffected.
    check_clarification: bool = True


class IntentSchema(BaseModel):
    """Maps a regex pattern to a workflow name for fast intent routing."""
    pattern: str
    workflow: str


class WorkflowFileSchema(BaseModel):
    """Top-level schema for a workflow YAML file."""
    name: str
    display_name: str = ""   # human-readable label shown in the chat window and task list;
                             # falls back to `name` if omitted
    description: str
    version: str = "1.0"
    entry_agent: str                                        # name of the root agent
    tools: list[ToolRef] = Field(default_factory=list)
    agents: list[AgentSchema] = Field(default_factory=list)
    intents: list[IntentSchema] = Field(default_factory=list)
    action_prompt: str = ""  # question asked when the user clicks this workflow's
                             # quick-start button on the welcome screen;
                             # leave empty to omit the button for this workflow
    aliases: list[str] = Field(default_factory=list)
                             # short informal names recognised by the switch command,
                             # e.g. ["confluence", "kb"] for the DevOps Knowledgebase
