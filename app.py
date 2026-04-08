"""
Chainlit entry point.

Per-session state  (cl.user_session)
──────────────────────────────────────
  current_workflow  (str | None)
      The last workflow that handled a message.
      Used by route_with_context() to decide follow-up vs new topic.
      Preserved (not reset) when a workflow step asks for clarification,
      so the user's answer can be routed back to the same workflow.

  history  (list[BaseMessage])
      Clean conversation: Human messages + final AI responses (no tool noise).
      Passed to workflow.run() so every agent sees all prior turns.

  awaiting_clarification  (bool)
      Set to True in two distinct situations:
        A) Router asked a clarifying question (current_workflow is None).
           Next message → route() fresh, ignoring continuation heuristics.
        B) A workflow step asked for more info (current_workflow is set).
           Next message → skip routing entirely and go straight back to
           the same workflow, so the user's answer is not mis-routed to
           a different workflow or the fallback.

Message handling per turn
──────────────────────────
  awaiting_clarification=True, current_workflow=None  →  route() fresh   (case A)
  awaiting_clarification=True, current_workflow=set   →  reuse workflow  (case B)
  normal, first message                               →  route()
  normal, follow-up detected                          →  keep current_workflow
  normal, new topic detected                          →  route() fresh
  routing returns clarification                       →  ask question, set flag

Run:
  chainlit run app.py
"""

import re
import uuid

import chainlit as cl
from chainlit import Task, TaskList, TaskStatus
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage

from framework.agents.llm_agent import LLMAgent
from framework.core.log import logger
from framework.loader.yaml_loader import YAMLLoader
from framework.workflow.intent_router import IntentRouter, RoutingDecision
from framework.workflow.workflow import Workflow, extract_history, compact_history


def _extract_content(content) -> str:
    """
    Safely convert an AIMessage.content to a plain string.

    Some providers (notably Anthropic with tool-calling) return content as a
    list of typed blocks rather than a plain string, e.g.:
        [{"type": "text", "text": "Here is the analysis…"},
         {"type": "tool_use", "id": "…", …}]

    Passing such a list directly to cl.Message(content=…) produces an empty
    or broken response.  This function extracts only the text blocks and joins
    them so the user always sees readable output.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)

# ── Load workflows ─────────────────────────────────────────────────────────────

_loader = YAMLLoader()
_workflows: dict[str, Workflow] = {}

for _yaml in Path("workflows").rglob("workflow.yaml"):
    try:
        _wf = _loader.load(_yaml)
        _workflows[_wf.name] = _wf
        print(f"[startup] loaded workflow: {_wf.name}")
    except Exception as exc:
        print(f"[startup] WARN: failed to load {_yaml}: {exc}")

_FALLBACK_NAME = "general_assistant"


def _build_fallback_workflow(workflows: dict[str, Workflow]) -> Workflow:
    """
    General-purpose fallback workflow for unmatched intents.

    The system prompt is built dynamically so the LLM always knows what
    specialized workflows are available and can guide users toward them.
    """
    descriptions = "\n".join(
        f"  • {wf.display_name}: {wf.description}"
        for wf in workflows.values()
    )
    role = f"""You are a DevOps AI assistant with the following specialized capabilities:

{descriptions}

These are your ONLY capabilities. Do not claim or imply any others.

Guidelines:
- If the user asks what you can do, list ONLY the capabilities above — nothing else.
- If the user's question matches one of your capabilities, answer briefly and encourage
  them to ask specifically so you can use the right tools to help fully.
- If the user asks something outside your capabilities, say so clearly and briefly,
  then remind them what you can help with.
- Never fabricate capabilities, tools, or knowledge you do not have.
- Keep responses concise and friendly."""

    agent = LLMAgent(
        name=_FALLBACK_NAME,
        display_name="General Assistant",
        description="General assistant for questions not covered by specialized workflows.",
        role=role,
        tools=[],
    )
    return Workflow(
        name=_FALLBACK_NAME,
        display_name="General Assistant",
        description="General assistant",
        entry_agent=agent,
    )


_fallback_workflow = _build_fallback_workflow(_workflows)
_workflows[_FALLBACK_NAME] = _fallback_workflow

_router = IntentRouter(
    list(_workflows.values()),
    fallback_workflow_name=_FALLBACK_NAME,
)

# ── Chainlit handlers ──────────────────────────────────────────────────────────


@cl.on_chat_start
async def on_chat_start() -> None:
    session_id = cl.context.session.id
    cl.user_session.set("current_workflow", None)
    cl.user_session.set("history", [])
    cl.user_session.set("awaiting_clarification", False)

    logger.info("SESSION", "chat started", session_id=session_id,
                loaded_workflows=list(_workflows.keys()))

    # Build capability descriptions with per-workflow usage hints
    _USAGE_HINTS: dict[str, str] = {
        "jenkins_log_analysis": (
            "Share a **Job URL**, a **Build URL**, or **paste error text** directly."
        ),
        "confluence_qa": (
            "Ask any question about your DevOps processes or onboarding procedures."
        ),
    }

    lines: list[str] = []
    for wf in _workflows.values():
        if wf.name == _FALLBACK_NAME:
            continue
        hint = _USAGE_HINTS.get(wf.name, "")
        hint_str = f"\n  ↳ {hint}" if hint else ""
        lines.append(f"- **{wf.display_name}**: {wf.description}{hint_str}")
    capabilities = "\n".join(lines) if lines else "_No workflows loaded._"

    # Build quick-start action buttons for workflows that have an action_prompt
    actions = [
        cl.Action(
            name="start_workflow",
            payload={"workflow": wf.name},
            label=f"▶ {wf.display_name}",
            description=wf.description,
        )
        for wf in _workflows.values()
        if wf.name != _FALLBACK_NAME and wf.action_prompt
    ]

    await cl.Message(
        content=(
            "Hello! I'm your DevOps AI assistant.\n\n"
            f"**Available capabilities:**\n{capabilities}\n\n"
            "Type your request and I'll route it automatically — or click a button below to get started."
        ),
        actions=actions if actions else None,
    ).send()


@cl.action_callback("start_workflow")
async def on_start_workflow(action: cl.Action) -> None:
    """
    Handles clicks on the quick-start workflow buttons shown on the welcome screen.

    Sets the session's current_workflow and awaiting_clarification so the next
    user message is routed directly to the chosen workflow.  Sends the workflow's
    action_prompt as the assistant's first message to guide the user.
    """
    workflow_name: str = action.payload.get("workflow", "")
    workflow = _workflows.get(workflow_name)
    if not workflow:
        await cl.Message(content="Workflow not found.").send()
        return

    # Pre-select the workflow — next message bypasses intent routing
    cl.user_session.set("current_workflow", workflow_name)
    cl.user_session.set("awaiting_clarification", True)

    history: list = cl.user_session.get("history", [])
    prompt = workflow.action_prompt.strip()

    # Record the button click in history as a synthetic user message
    cl.user_session.set("history", history + [
        HumanMessage(content=f"[started {workflow.display_name}]"),
        AIMessage(content=prompt),
    ])

    logger.info("SESSION", "quick-start button clicked", workflow=workflow_name)
    await cl.Message(content=prompt, author="assistant").send()


def _detect_switch_command(text: str, workflows: dict[str, "Workflow"]) -> str | None:
    """
    Detect explicit workflow switch commands like "switch to jenkins workflow" or
    "use confluence workflow".  Returns the matched workflow name, or None.

    Matches patterns such as:
      - "switch to <name>"
      - "use <name> workflow"
      - "change to <name>"
      - "go to <name>"
      - "open <name>"
    where <name> matches a workflow's display_name or internal name (case-insensitive).
    """
    normalized = text.strip().lower()
    prefixes = r"(?:switch\s+to|use|change\s+to|go\s+to|open|start)\s+"
    for wf in workflows.values():
        if wf.name == _FALLBACK_NAME:
            continue
        # Build aliases: internal name and each word of the display name
        aliases = [
            re.escape(wf.name.replace("_", " ")),
            re.escape(wf.display_name.lower()),
        ]
        # Also allow partial match on just the first word of the display name
        first_word = wf.display_name.split()[0].lower() if wf.display_name else ""
        if first_word:
            aliases.append(re.escape(first_word))
        pattern = rf"^{prefixes}({'|'.join(aliases)})(\s+workflow)?$"
        if re.match(pattern, normalized):
            return wf.name
    return None


@cl.on_message
async def on_message(message: cl.Message) -> None:
    # ── 1. Read session state ────────────────────────────────────────────────────
    history = cl.user_session.get("history", [])
    current_workflow: str | None = cl.user_session.get("current_workflow")
    awaiting_clarification: bool = cl.user_session.get("awaiting_clarification", False)

    logger.debug("SESSION", "message received",
                 session_id=cl.context.session.id,
                 current_workflow=current_workflow,
                 awaiting_clarification=awaiting_clarification,
                 history_len=len(history),
                 message=message.content[:120])

    # ── 1b. Natural-language workflow switch command ──────────────────────────────
    #
    # Allow the user to explicitly switch workflows at any time by typing a
    # command like "switch to Jenkins workflow" or "use Confluence workflow".
    # This is checked before routing so it always takes effect immediately,
    # even when awaiting_clarification or mid-workflow.
    switch_target = _detect_switch_command(message.content, _workflows)
    if switch_target and switch_target in _workflows:
        target_wf = _workflows[switch_target]
        cl.user_session.set("current_workflow", switch_target)
        cl.user_session.set("awaiting_clarification", True)
        prompt = target_wf.action_prompt.strip() if target_wf.action_prompt else (
            f"Switched to **{target_wf.display_name}**. How can I help?"
        )
        cl.user_session.set("history", history + [
            HumanMessage(content=message.content),
            AIMessage(content=prompt),
        ])
        logger.info("ROUTING", "explicit workflow switch", target=switch_target)
        await cl.Message(
            content=f"*Switching to **{target_wf.display_name}***\n\n{prompt}",
            author="assistant",
        ).send()
        return

    # ── 2. Route ─────────────────────────────────────────────────────────────────
    #
    # awaiting_clarification=True can mean one of two things:
    #
    #   Case A — the ROUTER asked a clarifying question (e.g. "which workflow did
    #   you mean?").  current_workflow is None because no workflow ran yet.
    #   The user's answer must go through route() fresh so we can pick the right
    #   workflow from their reply.  We deliberately skip route_with_context() here
    #   because a short answer like "jenkins" would look like a follow-up and get
    #   mis-classified as a continuation of whatever ran before.
    #
    #   Case B — a WORKFLOW STEP asked for more information (e.g. "please provide
    #   the Jenkins URL").  current_workflow is still set to that workflow's name.
    #
    #   Previous behaviour: bypass routing entirely → always sent the reply back to
    #   the same workflow.  Problem: if the user typed something unrelated (topic
    #   switch), the same workflow kept running and produced wrong output.
    #
    #   New behaviour: use route_with_context() which checks whether the message
    #   continues the current workflow or starts a new topic:
    #     • Short messages (< 35 chars) are still treated as continuations — a
    #       Jenkins URL reply is typically longer and will be evaluated by the LLM.
    #     • Longer messages that look like a different topic are routed fresh so the
    #       user lands on the right workflow (or the general assistant).
    #     • Messages matching another workflow's intent patterns route to that
    #       workflow immediately via regex, without an LLM call.
    #
    if awaiting_clarification:
        cl.user_session.set("awaiting_clarification", False)
        if current_workflow and current_workflow != _FALLBACK_NAME:
            # Case B: let route_with_context() decide — it returns the current
            # workflow for likely replies, and routes fresh on topic switches.
            # strict=True disables the short-message heuristic so that brief
            # unrelated replies like "hello" or "never mind" are checked by the
            # LLM instead of being blindly treated as continuations.
            decision = await _router.route_with_context(
                message.content,
                current_workflow=current_workflow,
                history=history,
                strict=True,
            )
        else:
            # Case A: route the user's answer fresh so we can pick the right
            # workflow from their reply to the router's question.
            decision = await _router.route(message.content)
    else:
        decision = await _router.route_with_context(
            message.content,
            current_workflow=current_workflow,
            history=history,
        )

    logger.info("ROUTING", "decision",
                workflow=decision.workflow,
                needs_clarification=decision.needs_clarification,
                clarification_preview=decision.clarification[:80] if decision.clarification else None)

    # ── 3. Handle clarification request ──────────────────────────────────────────
    #
    # The router isn't confident enough to pick a workflow.
    # Send its question to the user, record the exchange in history, and return.
    # The user's next message will be routed fresh (awaiting_clarification=True).
    #
    if decision.needs_clarification:
        await cl.Message(
            content=decision.clarification,
            author="assistant",
        ).send()

        # Preserve the exchange so the next routing call has context.
        updated_history = history + [
            HumanMessage(content=message.content),
            AIMessage(content=decision.clarification),
        ]
        cl.user_session.set("history", updated_history)
        cl.user_session.set("awaiting_clarification", True)
        # Don't touch current_workflow — it stays as-is until a workflow runs.
        return

    # ── 4. Resolve workflow ──────────────────────────────────────────────────────
    workflow_name = decision.workflow
    workflow = _workflows.get(workflow_name)

    if not workflow:
        await cl.Message(
            content=f"No workflow found for your request (resolved: `{workflow_name}`)."
        ).send()
        return

    # ── 5. Notify on mid-conversation topic switch ───────────────────────────────
    switched = current_workflow is not None and workflow_name != current_workflow
    if switched:
        await cl.Message(
            content=f"*Switching topic → **{workflow.display_name}***",
            author="router",
        ).send()

    cl.user_session.set("current_workflow", workflow_name)
    logger.info("WORKFLOW", "selected", workflow=workflow_name, switched=switched)

    # ── 6. Run workflow with progress tracking ───────────────────────────────────
    step_names = workflow.steps()

    # Build TaskList with all steps in READY state up front.
    # A new TaskList object is created every turn so the sidebar shows a clean
    # slate for the current run (not the icon states from a prior run).
    display_names = workflow.agent_display_names()
    task_list = TaskList()
    task_list.status = workflow.display_name
    tasks: dict[str, Task] = {}
    for name in step_names:
        task = Task(title=display_names.get(name, name), status=TaskStatus.READY)
        tasks[name] = task
        await task_list.add_task(task)
    await task_list.send()

    # Mark the first step as running before execution begins
    if step_names:
        tasks[step_names[0]].status = TaskStatus.RUNNING
        await task_list.update()

    # Generate a fresh thread_id for every workflow invocation.
    #
    # Why not reuse cl.context.session.id?
    # LangGraph's checkpointer accumulates state across calls that share the
    # same thread_id.  Reusing the session id causes two problems:
    #
    #   1. Stale task_results: after a successful 3-step run, the checkpoint
    #      holds all three agents in task_results.  The next run's first state
    #      emission includes ALL THREE keys, so stream_steps reports every step
    #      as "done" before any agent has actually run in the new call.
    #
    #   2. Sticky clarification_needed: the _or_bool reducer (True OR False = True)
    #      means a True value from a prior run can never be reset to False by
    #      starting a new invocation.  The SequentialAgent then sees
    #      clarification_needed=True immediately and exits before running any step.
    #
    # Using a UUID per invocation gives each call a fresh checkpoint slot.
    # Conversation context is passed explicitly via the `history` argument, so
    # no cross-run state is lost.
    run_thread_id = f"{cl.context.session.id}-{uuid.uuid4().hex}"

    result: dict = {}
    logger.info("WORKFLOW", "stream_steps start", workflow=workflow_name,
                steps=step_names, thread_id=run_thread_id)
    try:
        async for completed_name, state in workflow.stream_steps(
            message.content,
            thread_id=run_thread_id,
            history=history,
        ):
            if completed_name is None:
                # Final yield — workflow fully complete
                result = state
                logger.info("WORKFLOW", "stream_steps complete", workflow=workflow_name)
                break

            stopped_early = bool(state.get("clarification_needed") or state.get("error"))
            logger.info("WORKFLOW", "step complete", step=completed_name,
                        stopped_early=stopped_early,
                        clarification_needed=state.get("clarification_needed"),
                        error=state.get("error"))

            # Mark the completed step FAILED if it stopped the pipeline, DONE otherwise.
            # Remaining steps that never ran stay READY ("not started") — they are NOT
            # marked FAILED because they were never attempted.
            if completed_name in tasks:
                tasks[completed_name].status = (
                    TaskStatus.FAILED if stopped_early else TaskStatus.DONE
                )
                await task_list.update()

            if stopped_early:
                # Pipeline halted — leave unstarted steps in READY state so the
                # sidebar shows them as "not started" rather than failed.
                result = state
                break

            # Only advance the progress indicator to the next step when the
            # current step completed successfully (checked above).
            idx = step_names.index(completed_name) if completed_name in step_names else -1
            if 0 <= idx < len(step_names) - 1:
                tasks[step_names[idx + 1]].status = TaskStatus.RUNNING
                await task_list.update()

    except Exception as exc:
        logger.error("WORKFLOW", f"stream_steps raised for workflow '{workflow_name}'",
                     exc=exc, workflow=workflow_name)
        # Mark all in-progress tasks as failed so the UI shows a clean terminal state
        for task in tasks.values():
            if task.status in (TaskStatus.READY, TaskStatus.RUNNING):
                task.status = TaskStatus.FAILED
        await task_list.update()
        await cl.Message(
            content=f"An error occurred while running **{workflow.display_name}**:\n```\n{exc}\n```"
        ).send()
        return

    # ── Finalise task list ───────────────────────────────────────────────────────
    # Only flip RUNNING → DONE.  This handles the single-step workflow case where
    # the step is marked RUNNING before the loop but stream_steps exits via the
    # "completed_name is None" path without ever yielding the step name.
    #
    # READY tasks are intentionally left as READY — they represent steps that
    # were never attempted (pipeline stopped early).  Stamping them DONE would
    # falsely imply they ran successfully.
    #
    # FAILED tasks are left unchanged (a failed step stays failed).
    for task in tasks.values():
        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.DONE
    await task_list.update()

    # ── 7. Surface the final response ────────────────────────────────────────────
    ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    # _extract_content() normalises content to str — some providers (e.g. Anthropic
    # with tool-calling) return a list of typed blocks instead of a plain string.
    final = _extract_content(ai_messages[-1].content) if ai_messages else "Workflow completed with no output."
    logger.debug("RESPONSE", "sending final response",
                 workflow=workflow_name, content_preview=final[:150])
    await cl.Message(content=final).send()

    # ── 8. If a step asked for clarification, pause here ─────────────────────────
    # The clarification question has already been shown as the final AI message.
    #
    # Intentionally NOT resetting current_workflow here.  Keeping it set is what
    # distinguishes "workflow step asked" (case B) from "router asked" (case A) in
    # the routing block above.  On the next turn, step 2 will see
    # awaiting_clarification=True AND current_workflow=<this workflow> and bypass
    # routing entirely, sending the user's answer straight back here.
    #
    # If we reset current_workflow to None, step 2 would fall into case A and call
    # route() fresh.  The user's answer (e.g. raw log output) rarely contains the
    # workflow's trigger keywords, so it would land on the fallback — and the
    # workflow would never receive the information it asked for.
    if result.get("clarification_needed"):
        cl.user_session.set("awaiting_clarification", True)
        cl.user_session.set("history", compact_history(extract_history(result)))
        return

    # ── 9. Persist clean history ─────────────────────────────────────────────────
    cl.user_session.set("history", compact_history(extract_history(result)))
