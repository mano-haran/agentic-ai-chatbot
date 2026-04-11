from typing import Any, AsyncGenerator, AsyncIterator, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

import config
from framework.core.log import logger
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
        "clarification_questions": [],
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

    """
    kept: list[BaseMessage] = []
    for m in result.get("messages", []):
        if isinstance(m, HumanMessage):
            kept.append(m)
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            kept.append(m)
    return kept


def compact_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Apply the configured history compaction strategy to the extracted history.

    Strategy is controlled by HISTORY_STRATEGY in .env (default: "none"):

      none    — return messages unchanged; history grows unbounded.
                Safe default: no risk of losing context, no extra LLM calls.

      window  — keep only the most recent HISTORY_WINDOW_SIZE messages.
                Zero cost. Older context is hard-dropped, so very long
                investigations may lose early details.

      summary — if the history exceeds HISTORY_WINDOW_SIZE messages, ask the
                LLM to write a concise prose summary of the older portion, then
                return [SystemMessage(summary)] + the recent tail.
                Preserves semantic context at the cost of one extra LLM call
                whenever the threshold is crossed.

    Always call this on the output of extract_history(), never on raw state
    messages (ToolMessages and intermediate AIMessages should be filtered first).
    """
    strategy = config.HISTORY_STRATEGY.lower()

    if strategy == "none" or len(messages) == 0:
        return messages

    if strategy == "window":
        return _apply_window(messages)

    if strategy == "summary":
        return _apply_summary(messages)

    # Unknown strategy — log a warning and fall back to no-op
    print(
        f"[history] Unknown HISTORY_STRATEGY='{strategy}'. "
        "Valid values: none, window, summary. Falling back to 'none'.",
        flush=True,
    )
    return messages


# ── Strategy implementations ───────────────────────────────────────────────────

def _apply_window(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Keep only the last HISTORY_WINDOW_SIZE messages.

    The window always starts on a HumanMessage so the conversation begins
    with a user turn rather than a dangling assistant response.
    """
    n = config.HISTORY_WINDOW_SIZE
    if len(messages) <= n:
        return messages

    tail = messages[-n:]

    # Advance past any leading AIMessage so the window starts with a user turn
    for i, m in enumerate(tail):
        if isinstance(m, HumanMessage):
            tail = tail[i:]
            break

    print(
        f"[history] window: kept {len(tail)} of {len(messages)} messages "
        f"(window={n})",
        flush=True,
    )
    return tail


def _apply_summary(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Summarise the older portion of the history with an LLM call when it
    exceeds HISTORY_WINDOW_SIZE messages, then return:
        [SystemMessage(<summary>)] + <recent tail>

    The recent tail is the last (HISTORY_WINDOW_SIZE // 2) messages so the
    LLM always has direct access to the most recent exchange in full.

    If the history is within the window limit, returns it unchanged — no LLM
    call is made.
    """
    n = config.HISTORY_WINDOW_SIZE
    if len(messages) <= n:
        return messages

    # Split: everything older than the recent tail gets summarised
    tail_size = max(n // 2, 2)          # at least 2 messages kept verbatim
    older = messages[:-tail_size]
    recent = messages[-tail_size:]

    # Format older messages as readable text for the summarisation prompt
    older_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
        f"{m.content[:500]}"          # cap per-message length to avoid inception
        for m in older
        if isinstance(m, (HumanMessage, AIMessage))
    )

    prompt = (
        "Summarise the following conversation excerpt concisely in 3-6 sentences. "
        "Capture the key questions asked, decisions made, and any important facts "
        "established.  Write in the third person (e.g. 'The user asked…').\n\n"
        f"{older_text}"
    )

    try:
        from framework.providers.factory import get_llm
        llm = get_llm(model_id=config.HISTORY_SUMMARY_MODEL, temperature=0.0, max_tokens=512)
        response = llm.invoke([HumanMessage(content=prompt)])
        summary_text = f"[Earlier conversation summary]\n{response.content.strip()}"
        print(
            f"[history] summary: condensed {len(older)} older messages into a summary, "
            f"kept {len(recent)} recent messages verbatim",
            flush=True,
        )
        return [SystemMessage(content=summary_text)] + list(recent)
    except Exception as exc:
        # If summarisation fails for any reason, fall back to window to avoid
        # passing an oversized history that could blow the context limit
        print(
            f"[history] summary failed ({exc}); falling back to window strategy",
            flush=True,
        )
        return _apply_window(messages)


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
        display_name: str = "",
        action_prompt: str = "",
        aliases: list[str] | None = None,
    ):
        self.name = name
        self.display_name = display_name or name   # falls back to internal name if not set
        self.description = description
        self.entry_agent = entry_agent
        self.intents: list[dict[str, str]] = intents or []
        self.action_prompt = action_prompt   # follow-up shown when the quick-start button is clicked
        self.aliases: list[str] = [a.lower() for a in (aliases or [])]
        self._checkpointer = get_checkpointer()
        # Compiled graph is built lazily on first use so startup only pays the
        # cost of parsing YAML and constructing agent config objects.
        # LangGraph's StateGraph.compile() is expensive (graph validation, runnable
        # wiring, checkpointer setup) — deferring it keeps app startup fast.
        self._compiled = None

    def steps(self) -> list[str]:
        """
        Return the ordered list of internal agent names for progress tracking.
        For a SequentialAgent entry point, returns each sub-agent name in order.
        For any other agent type, returns the single entry agent name.
        """
        if isinstance(self.entry_agent, SequentialAgent):
            return [a.name for a in self.entry_agent.sub_agents]
        return [self.entry_agent.name]

    def agent_display_names(self) -> dict[str, str]:
        """
        Map each internal agent name to its display_name for the task list UI.
        Internal names are used as dict keys throughout the workflow; display
        names are the human-readable labels shown in the Chainlit sidebar.
        """
        if isinstance(self.entry_agent, SequentialAgent):
            return {a.name: a.display_name for a in self.entry_agent.sub_agents}
        return {self.entry_agent.name: self.entry_agent.display_name}

    def _get_compiled(self):
        """
        Return the compiled LangGraph, building it on first call.

        Compilation is deferred from __init__ so the app starts immediately
        after loading YAML — the first user message that hits a workflow pays
        the one-time compilation cost instead of every process start.
        """
        if self._compiled is None:
            import time
            t0 = time.perf_counter()
            self._compiled = self.entry_agent.compile(checkpointer=self._checkpointer)
            print(
                f"[workflow] compiled '{self.name}' in {time.perf_counter() - t0:.2f}s",
                flush=True,
            )
        return self._compiled

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
        return await self._get_compiled().ainvoke(
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
        async for event in self._get_compiled().astream_events(
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

        async for mode, data in self._get_compiled().astream(
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

    async def stream_with_events(
        self,
        user_message: str,
        thread_id: str,
        history: list[BaseMessage] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AsyncGenerator[tuple, None]:
        """
        Stream workflow execution yielding typed event tuples for both
        workflow-level step tracking AND within-step tool calls.

        Used when AGENT_STEPS != "off" so the UI can show collapsible cl.Step
        blocks for each tool invocation alongside the TaskList step indicators.

        Yields
        ------
        ("step_start", step_name)              a pipeline stage began
        ("step_done",  step_name, partial)     a stage finished; partial contains
                                               clarification_needed and error flags
        ("tool_start", tool_name, tool_input)  a tool was invoked; input is a dict
        ("tool_end",   tool_name, tool_output) a tool returned; output is a string
        ("done",       None,      final_state) whole workflow complete; final_state
                                               contains the full messages list

        How tool events are captured
        ----------------------------
        LangGraph's astream_events(version="v2") propagates events through the
        callback system.  make_agent_node() forwards the outer RunnableConfig
        (which carries those callbacks) into each agent's inner ainvoke() call.
        This makes on_tool_start / on_tool_end events from inside agent graphs
        visible to this outer stream — without that forwarding, tool events would
        be silently swallowed.

        Final state
        -----------
        Per-node outputs (from on_chain_end) are partial dicts — they contain only
        the keys the node updated, and messages use the add_messages reducer so we
        cannot reconstruct the full list by merging partials.  Instead, after the
        event stream ends we read the authoritative final state directly from the
        LangGraph checkpointer, which holds the fully-reduced state.
        """
        initial = _build_initial_state(user_message, history, metadata)
        run_config = self._config(thread_id)
        expected_steps = set(self.steps())

        # Accumulate scalar sentinel fields from per-node outputs.
        # We only track these (not messages) since messages need the add_messages
        # reducer that only LangGraph applies correctly.
        partial: dict[str, Any] = {}

        async for event in self._get_compiled().astream_events(
            initial, config=run_config, version="v2"
        ):
            evt = event["event"]
            name = event.get("name", "")
            data = event.get("data", {})

            if evt == "on_chain_start" and name in expected_steps:
                yield ("step_start", name)

            elif evt == "on_chain_end" and name in expected_steps:
                output = data.get("output", {})
                if isinstance(output, dict):
                    for key in ("clarification_needed", "error", "next_agent",
                                "clarification_questions"):
                        if key in output:
                            partial[key] = output[key]
                yield ("step_done", name, dict(partial))

            elif evt == "on_tool_start":
                yield ("tool_start", name, data.get("input", {}))

            elif evt == "on_tool_end":
                yield ("tool_end", name, data.get("output", ""))

        # Read the authoritative final state from the checkpointer so we have the
        # complete messages list (including all add_messages reductions).
        try:
            cp = await self._checkpointer.aget_tuple(run_config)
            final_state: dict[str, Any] = (
                cp.checkpoint.get("channel_values", partial) if cp else partial
            )
        except Exception as exc:
            logger.warning("WORKFLOW", "could not read final state from checkpointer",
                           exc=str(exc))
            final_state = partial

        yield ("done", None, final_state)
