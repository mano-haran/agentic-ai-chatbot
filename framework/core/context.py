from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunContext:
    """
    Passed through agent execution trees to carry session-level information.
    """
    session_id: str
    user_message: str
    workflow_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
