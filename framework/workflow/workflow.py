from typing import Any, AsyncGenerator, AsyncIterator
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from framework.agents.base import BaseAgent
from framework.agents.workflow_agents import SequentialAgent
from framework.core.checkpointer import get_checkpointer
from framework.core.state import AgentState


def _build_initial_state(
    user_message: str,
    history: list[BaseMessage] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the initial AgentState for a workflow run.

    history is prepended before the new user message so every agent in the
    workflow sees the full prior conversation, not just the current question.

    State layout:
        [*history messages*]  [HumanMessage(user_message)]  → agents add more
    """
    messages: list[BaseMessage] = list(history or []) + [HumanMessage(content=user_message)]
    return {
        "messages": messages,
        "next_agent": "",
        "task_results": {},
        "metadata": metadata or {},
        "error": None,
        "clarification_needed": False,
    }


def extract_history(result: dict[str, Any]) -> list[BaseMessage]:
    """
    Extract conversation-worthy messages from a completed workflow result.

    Keeps:  HumanMessage (user inputs)
            AIMessage without tool_calls (final agent responses)

    Discards:
            AIMessage WITH tool_calls  (intermediate "I'll call a tool" steps —
                                        not meaningful as conversation context)
            ToolMessage                (raw tool output — often very large and
                                        not useful as standalone context)

    These kept messages form the `history` passed into the next turn, giving
    every agent in subsequent runs full awareness of prior exchanges.

    Note: For very long conversations you may want to trim this list to the
    last N messages to stay within model context limits. A good starting point
    is keeping the last 20 messages (~10 user/assistant exchange pairs).
    """
    kept: list[BaseMessage] = []
    for m in result.get("messages", []):
        if isinstance(m, HumanMessage):
            kept.append(m)
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            kept.append(m)
    return kept


class Workflow:
    """
    Top-level runnable unit.  Wraps an entry_agent (any BaseAgent subclass)
    and exposes run() / stream() methods for the Chainlit app layer.

    Checkpointing
    -------------
    The workflow creates a checkpointer based on APP_ENV:
      - dev  → MemorySaver (in-process, wiped on restart)
      - prod → AsyncPostgresSaver (persistent, requires POSTGRES_URL)

    Every run() / stream() call requires a thread_id so state is isolated
    per user/session. The caller (Chainlit, API layer) owns the thread_id.

    Multi-turn usage
    ----------------
    With checkpointing enabled, state is persisted automatically between
    calls with the same thread_id — no need to pass history manually:

        # Turn 1
        await workflow.run("Why did my build fail?", thread_id="user-123")

        # Turn 2 — agents see the full prior conversation via checkpointer
        await workflow.run("What about the second error?", thread_id="user-123")

    Python DSL quick-start:
        workflow = Workflow(
            name="jenkins_log_analysis",
            description="Analyse Jenkins failures and suggest fixes",
            entry_agent=jenkins_pipeline,
            intents=[{"pattern": r"jenkins|build fail", "workflow": "jenkins_log_analysis"}],
        )
    """

    def __init__(
        self,
        name: str,
        description: str,
        entry_agent: BaseAgent,
        intents: list[dict[str, str]] | None = None,
    ):
        self.name = name
        self.description = description
        self.entry_agent = entry_agent
        self.intents: list[dict[str, str]] = intents or []
        self._checkpointer = get_checkpointer()
        self._compiled = self.entry_agent.compile(checkpointer=self._checkpointer)

    def steps(self) -> list[str]:
        """
        Return the ordered list of agent step names for progress tracking.
        For a SequentialAgent entry point, returns each sub-agent name in order.
        For any other agent type, returns the single entry agent name.
        """
        if isinstance(self.entry_agent, SequentialAgent):
            return [a.name for a in self.entry_agent.sub_agents]
        return [self.entry_agent.name]

    def _config(self, thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    async def run(
        self,
        user_message: str,
        thread_id: str,
        history: list[BaseMessage] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the workflow to completion and return the final AgentState."""
        return await self._compiled.ainvoke(
            _build_initial_state(user_message, history, metadata),
            config=self._config(thread_id),
        )

    async def stream(
        self,
        user_message: str,
        thread_id: str,
        history: list[BaseMessage] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncIterator:
        """
        Yield LangGraph astream_events (v2) for the full workflow execution.
        Filter by event["event"] == "on_chat_model_stream" to get tokens.
        """
        async for event in self._compiled.astream_events(
            _build_initial_state(user_message, history, metadata),
            config=self._config(thread_id),
            version="v2",
        ):
            yield event

    async def stream_steps(
        self,
        user_message: str,
        thread_id: str,
        history: list[BaseMessage] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[tuple[str | None, dict[str, Any]], None]:
        """
        Stream workflow execution yielding per-step progress events.

        Yields (completed_step_name, current_state) each time a step finishes.
        Yields (None, final_state) once when the workflow is fully complete.

        Why stream_mode=["updates", "values"] instead of stream_mode="values"
        -----------------------------------------------------------------------
        The old implementation used stream_mode="values" and inferred which step
        had just completed by watching for new keys appearing in task_results.
        This had two failure modes caused by the LangGraph checkpointer:

          1. Stale task_results from a previous run (same thread_id) were already
             in the checkpoint.  When the first node of a new run emitted its
             state, ALL prior steps appeared as "new" keys in task_results
             simultaneously, so stream_steps reported every step as done at once —
             in unpredictable set-iteration order — regardless of what actually ran.

          2. The _or_bool reducer for clarification_needed means a True value from
             a prior run persists in the checkpoint and can never be reset to False
             by a new run that starts with clarification_needed=False.  (Both are
             fixed together by the per-run thread_id in app.py, but the stream_mode
             fix makes step detection correct even if thread_ids are ever reused.)

        stream_mode=["updates", "values"] yields interleaved tuples:
          ("updates", {node_name: node_output})  — which node just ran, exactly
          ("values",  full_accumulated_state)    — state after that node's output

        Using "updates" for step detection means we see only the node that ran in
        THIS call, never stale keys from the checkpoint.  Using "values" gives us
        the full state (messages, clarification_needed, etc.) to pass to the caller.
        """
        initial = _build_initial_state(user_message, history, metadata)
        config = self._config(thread_id)
        expected_steps = set(self.steps())

        # Node name detected from "updates" — paired with the next "values" state.
        pending_step: str | None = None
        final_state: dict[str, Any] = {}

        async for mode, data in self._compiled.astream(
            initial, config=config, stream_mode=["updates", "values"]
        ):
            if mode == "updates":
                # data = {node_name: node_output_dict}
                # Captures exactly which agent node executed in this invocation.
                # Only care about top-level workflow steps, not internal sub-graph nodes.
                for node_name in data:
                    if node_name in expected_steps:
                        pending_step = node_name

            elif mode == "values":
                # data = full accumulated state after applying the latest updates.
                final_state = data
                if pending_step is not None:
                    # Pair the completed step with the authoritative post-update state
                    # so the caller has clarification_needed, messages, etc.
                    yield pending_step, data
                    pending_step = None

        yield None, final_state
