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

import uuid

import chainlit as cl
from chainlit import Task, TaskList, TaskStatus
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage

from framework.agents.llm_agent import LLMAgent
from framework.loader.yaml_loader import YAMLLoader
from framework.workflow.intent_router import IntentRouter, RoutingDecision
from framework.workflow.workflow import Workflow, extract_history, compact_history

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
    cl.user_session.set("current_workflow", None)
    cl.user_session.set("history", [])
    cl.user_session.set("awaiting_clarification", False)

    lines = [f"- **{w.display_name}**: {w.description}" for w in _workflows.values() if w.name != _FALLBACK_NAME]
    capabilities = "\n".join(lines) if lines else "_No workflows loaded._"
    await cl.Message(
        content=(
            "Hello! I'm your DevOps AI assistant.\n\n"
            f"**Available capabilities:**\n{capabilities}\n\n"
            "Describe what you need and I'll route your request automatically."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    # ── 1. Read session state ────────────────────────────────────────────────────
    history = cl.user_session.get("history", [])
    current_workflow: str | None = cl.user_session.get("current_workflow")
    awaiting_clarification: bool = cl.user_session.get("awaiting_clarification", False)

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
    #   the build logs").  current_workflow is still set to that workflow's name
    #   because we intentionally did NOT reset it in step 8 below.
    #   The user's answer (e.g. the log contents) must go back to the SAME workflow
    #   so the agent can process the answer with the conversation history intact.
    #   If we ran route() here the message would often fail to match the workflow's
    #   regex patterns and land on the fallback instead, causing an infinite loop
    #   where the workflow keeps asking for the same information.
    #
    if awaiting_clarification:
        cl.user_session.set("awaiting_clarification", False)
        if current_workflow and current_workflow != _FALLBACK_NAME:
            # Case B: bypass routing — send the user's answer directly to the
            # workflow that asked the question.  History already contains the
            # workflow's clarifying question, so the agent has full context.
            decision: RoutingDecision = RoutingDecision(workflow=current_workflow, clarification=None)
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
    async for completed_name, state in workflow.stream_steps(
        message.content,
        thread_id=run_thread_id,
        history=history,
    ):
        if completed_name is None:
            # Final yield — workflow fully complete
            result = state
            break

        # Mark the completed step as done
        if completed_name in tasks:
            tasks[completed_name].status = TaskStatus.DONE
            await task_list.update()

        # If stopped early, mark remaining steps as failed
        if state.get("clarification_needed") or state.get("error"):
            for name in step_names:
                if tasks[name].status in (TaskStatus.READY, TaskStatus.RUNNING):
                    tasks[name].status = TaskStatus.FAILED
            await task_list.update()
            result = state
            break

        # Mark the next step as running
        idx = step_names.index(completed_name) if completed_name in step_names else -1
        if 0 <= idx < len(step_names) - 1:
            tasks[step_names[idx + 1]].status = TaskStatus.RUNNING
            await task_list.update()

    # ── Finalise task list ───────────────────────────────────────────────────────
    # After the stream_steps loop, some tasks may still show RUNNING or READY.
    # This happens in two situations:
    #   1. Single-step workflow: the step is marked RUNNING before the loop starts
    #      (line above), but if stream_steps exits via the "completed_name is None"
    #      branch (normal completion) without ever yielding the step name, the task
    #      never transitions to DONE.
    #   2. Multi-step workflow with an early break (clarification / error): the break
    #      already marks unstarted steps FAILED, but the CURRENTLY-RUNNING step
    #      may not have been captured as a task_results key yet.
    # Marking any residual RUNNING/READY tasks as DONE here ensures the sidebar
    # always shows a clean terminal state rather than a spinner that never resolves.
    # Tasks already marked FAILED (early stop) are intentionally left unchanged.
    for task in tasks.values():
        if task.status in (TaskStatus.READY, TaskStatus.RUNNING):
            task.status = TaskStatus.DONE
    await task_list.update()

    # ── 7. Surface the final response ────────────────────────────────────────────
    ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
    final = ai_messages[-1].content if ai_messages else "Workflow completed with no output."
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
