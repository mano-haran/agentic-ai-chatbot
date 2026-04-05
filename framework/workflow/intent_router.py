"""
Intent router — maps a user message to the right Workflow.

Single-turn routing strategy
------------------------------
Stage 1 — Regex scan (free, instant):
  Collect ALL workflows whose intent patterns match the message.

  a) No matches        →  LLM classifies from scratch.
  b) One match, short  →  Trust regex (keyword-heavy query, reliable).
  c) One match, long   →  LLM confirms (keyword may be incidental in a longer sentence).
  d) Multiple matches  →  LLM arbitrates (genuinely ambiguous across workflows).

Stage 2 — LLM classification (cheap, ~50ms):
  Given workflow names + descriptions, the LLM either:
    ROUTE: <workflow_name>    — confident match found
    CLARIFY: <question>       — message is too vague to route reliably

  When called after a single regex match (case c), the matched name is
  passed as a hint so the LLM can confirm or override it rather than
  starting from scratch.

Multi-turn routing strategy
-----------------------------
After the first message, route_with_context() checks whether the new
message continues the current workflow before running Stage 1/2.

  Heuristics (free):  very short message, or starts with a follow-up phrase.
  LLM (cheap):        longer ambiguous messages the heuristics can't judge.

Clarification flow
------------------
When route() returns a RoutingDecision with needs_clarification=True,
the caller (app.py) should:
  1. Send the clarification question to the user.
  2. Save the exchange to history.
  3. Set session["awaiting_clarification"] = True.
  4. On the next message, skip the continuation check and call route()
     directly so the user's answer is routed fresh, not treated as a
     follow-up to whichever workflow was active before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, AIMessage

import config
from framework.providers.factory import get_llm


# ── Return type ───────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """
    Result of a routing call.

    Exactly one of `workflow` or `clarification` is set:
      workflow      — the name of the workflow to run.
      clarification — a question the router wants to ask the user before
                      it can route confidently.

    Usage in app.py:
        decision = await router.route_with_context(...)
        if decision.needs_clarification:
            await cl.Message(content=decision.clarification).send()
            return
        workflow = workflows[decision.workflow]
    """
    workflow: str | None
    clarification: str | None

    @property
    def needs_clarification(self) -> bool:
        return self.clarification is not None


# ── Meta / capability question detection ─────────────────────────────────────
# Matched against the full message, case-insensitive.
# These are questions about the assistant itself — never route them to a workflow.
_META_RE = re.compile(
    r"^\s*(hi|hey|hello|howdy|greetings)\b"                          # greetings
    r"|what (can|do) you (do|help|support|offer|know|handle)"        # capability questions
    r"|what are (your|the) (capabilities|features|workflows|options)"
    r"|what('?re| are) you (capable of|able to do|good at)"
    r"|how can you help|what do you (support|offer)"
    r"|tell me (about|what) you (can|do|support)"
    r"|what('?s| is) (available|supported|possible)",
    re.IGNORECASE,
)

# ── Follow-up detection regex ─────────────────────────────────────────────────

# Matched against the START of the message, case-insensitive.
_FOLLOW_UP_RE = re.compile(
    r"^\s*("
    r"what about|can you|could you|also|and |but |now |show|tell|explain|"
    r"more|again|try|retry|fix|same|that|it |this|those|the |why|how|when|"
    r"which|re-?run|go ahead|proceed|ok|okay|sure|yes|no|thanks|thank you|"
    r"what (is|are|was|were)|what'?s|and what|so what|make it|do it|"
    r"help me|give me|list|any other|another|next|previous|last"
    r")\b",
    re.IGNORECASE,
)


# ── Router ────────────────────────────────────────────────────────────────────

class IntentRouter:
    """
    Routes a user message to the most appropriate registered Workflow,
    with awareness of ongoing multi-turn conversations.

    Returns RoutingDecision — either a resolved workflow name or a
    clarifying question to ask the user.
    """

    # Single regex match on a message this short is trustworthy.
    _SHORT_MESSAGE_WORDS = 8

    def __init__(
        self,
        workflows: list,
        model: str = config.DEFAULT_ROUTING_MODEL,
        fallback_workflow_name: str | None = None,
    ):
        self._workflows: dict = {w.name: w for w in workflows}
        self._model = model
        self._fallback_workflow_name = fallback_workflow_name
        self._fast_llm = None       # 16 tokens  — continuation yes/no
        self._classify_llm = None   # 256 tokens — route or clarify

    def register(self, workflow) -> None:
        self._workflows[workflow.name] = workflow

    # ── LLM clients ───────────────────────────────────────────────────────────

    def _fast_llm_client(self):
        """Minimal-token LLM for yes/no continuation decisions."""
        if self._fast_llm is None:
            self._fast_llm = get_llm(model_id=self._model, temperature=0.0, max_tokens=16)
        return self._fast_llm

    def _classify_llm_client(self):
        """Higher-token LLM for classification and clarification questions."""
        if self._classify_llm is None:
            self._classify_llm = get_llm(model_id=self._model, temperature=0.0, max_tokens=256)
        return self._classify_llm

    # ── Stage 1: regex scan ───────────────────────────────────────────────────

    def _scan_patterns(self, message: str) -> list[str]:
        """
        Scan ALL workflow intent patterns against the message.

        Returns a deduplicated list of matching workflow names.
        Does NOT stop at the first hit — collecting all matches is what
        lets us detect ambiguity and avoid first-match bias.
        """
        matched: list[str] = []
        seen: set[str] = set()
        for workflow in self._workflows.values():
            for intent in getattr(workflow, "intents", []):
                pattern = intent.get("pattern", "")
                target = intent.get("workflow", workflow.name)
                if (
                    pattern
                    and target not in seen
                    and target in self._workflows
                    and re.search(pattern, message, re.IGNORECASE)
                ):
                    matched.append(target)
                    seen.add(target)
        return matched

    # ── Stage 2: LLM classification ───────────────────────────────────────────

    async def _llm_classify(
        self,
        message: str,
        hint: str | None = None,
    ) -> RoutingDecision:
        """
        Ask the LLM to route the message — or admit it can't and ask the user.

        The LLM is instructed to reply in one of two formats:

            ROUTE: <workflow_name>
                → confident match; we use this workflow.

            CLARIFY: <question for the user>
                → message is too vague; we surface the question to the user
                  before routing so they can provide the missing context.

        Args:
            message: The user message to classify.
            hint:    Workflow name the regex stage suggested (case c above).
                     The LLM uses it as a starting point but can override it.
        """
        descriptions = "\n".join(
            f"  - {name}: {w.description}" for name, w in self._workflows.items()
        )
        hint_line = (
            f"The keyword matcher suggests '{hint}' — confirm or override.\n\n"
            if hint else ""
        )
        prompt = (
            f"You are a routing assistant for a DevOps AI assistant.\n\n"
            f"Available workflows:\n{descriptions}\n\n"
            f"{hint_line}"
            f"User message: {message}\n\n"
            f"Instructions:\n"
            f"  • Only reply ROUTE if the user's message clearly and specifically\n"
            f"    describes a task that one of the above workflows handles.\n\n"
            f"  • Reply CLARIFY in ALL of the following cases:\n"
            f"    - The message is a greeting (hi, hello, hey …)\n"
            f"    - The message asks about capabilities, features, or what you can do\n"
            f"    - The message is too vague or ambiguous to pick a workflow confidently\n"
            f"    - The message does not clearly match any of the available workflows\n\n"
            f"  • When clarifying, list each available workflow with a one-line\n"
            f"    description so the user knows exactly what they can ask for.\n\n"
            f"  ROUTE: <workflow_name>\n"
            f"  CLARIFY: <your question or capability summary>\n\n"
            f"Reply with ONLY one of those two formats — nothing else."
        )
        response = self._classify_llm_client().invoke([HumanMessage(content=prompt)])
        return self._parse_llm_response(response.content.strip())

    def _parse_llm_response(self, text: str) -> RoutingDecision:
        """
        Parse the LLM's ROUTE:/CLARIFY: response into a RoutingDecision.

        Falls back gracefully if the LLM doesn't follow the format exactly.
        """
        if text.upper().startswith("ROUTE:"):
            name = text[len("ROUTE:"):].strip()
            if name in self._workflows:
                return RoutingDecision(workflow=name, clarification=None)
            # LLM hallucinated a name — treat as ambiguous and clarify
            return self._cannot_route()

        if text.upper().startswith("CLARIFY:"):
            question = text[len("CLARIFY:"):].strip()
            return RoutingDecision(workflow=None, clarification=question)

        # LLM ignored format instructions — try to salvage a workflow name
        for name in self._workflows:
            if name.lower() in text.lower():
                return RoutingDecision(workflow=name, clarification=None)

        return self._cannot_route()

    def _fallback(self) -> RoutingDecision:
        """
        Route to the registered fallback workflow (general assistant).
        Falls back to _cannot_route() if no fallback workflow is configured.
        """
        if self._fallback_workflow_name:
            return RoutingDecision(workflow=self._fallback_workflow_name, clarification=None)
        return self._cannot_route()

    def _cannot_route(self) -> RoutingDecision:
        """
        Used when the message is genuinely ambiguous between multiple specific
        workflows and the LLM cannot arbitrate.  Lists the options and asks the
        user to clarify which one they mean.
        """
        # Exclude the fallback workflow from the options list
        options = "\n".join(
            f"• **{name}** — {w.description}"
            for name, w in self._workflows.items()
            if name != self._fallback_workflow_name
        )
        return RoutingDecision(
            workflow=None,
            clarification=(
                f"I can see your request could relate to multiple areas. "
                f"Here's what I support:\n\n{options}\n\n"
                f"Which one are you asking about?"
            ),
        )

    # ── Public single-turn routing ────────────────────────────────────────────

    async def route(self, message: str) -> RoutingDecision:
        """
        Route a single message to the best-matching workflow.

        Decision table:
          meta/capability question → fallback (free, no LLM call)
          matches = 0              → fallback (free, no LLM call)
          matches = 1, short       → trust regex
          matches = 1, long        → LLM confirms; CLARIFY → fallback
          matches ≥ 2              → LLM arbitrates; CLARIFY → ask user to choose
        """
        # Short-circuit: meta/capability questions go straight to fallback
        if _META_RE.search(message):
            return self._fallback()

        matches = self._scan_patterns(message)
        word_count = len(message.split())

        # No pattern match at all → fallback handles it (saves an LLM call)
        if len(matches) == 0:
            return self._fallback()

        if len(matches) == 1:
            if word_count <= self._SHORT_MESSAGE_WORDS:
                return RoutingDecision(workflow=matches[0], clarification=None)
            # LLM confirms the single match for longer messages
            decision = await self._llm_classify(message, hint=matches[0])
            # LLM wasn't confident → let fallback handle it
            if decision.needs_clarification:
                return self._fallback()
            return decision

        # Multiple matches → LLM arbitrates; if still ambiguous, ask the user
        # (don't use fallback here — the user IS asking about a real workflow)
        return await self._llm_classify(message)

    # ── Public multi-turn routing ─────────────────────────────────────────────

    async def route_with_context(
        self,
        message: str,
        current_workflow: str | None = None,
        history: list | None = None,
    ) -> RoutingDecision:
        """
        Route with awareness of the ongoing conversation.

        If there is an active workflow and the message looks like a follow-up,
        the same workflow is returned without any LLM call.
        Otherwise delegates to route().

        Args:
            message:          New user message.
            current_workflow: Workflow active in this session, or None.
            history:          Prior conversation messages, or empty list.
        """
        # Always re-route fresh from the fallback — it's a catch-all, not a domain,
        # so there's no meaningful "continuation" to detect.
        if (not current_workflow
                or current_workflow not in self._workflows
                or current_workflow == self._fallback_workflow_name):
            return await self.route(message)

        if await self._is_continuation(message, current_workflow, history or []):
            return RoutingDecision(workflow=current_workflow, clarification=None)

        return await self.route(message)

    # ── Continuation detection ────────────────────────────────────────────────

    async def _is_continuation(
        self,
        message: str,
        current_workflow: str,
        history: list,
    ) -> bool:
        """
        Returns True if the message continues the current workflow.

        Heuristics first (free); LLM only for genuinely ambiguous cases.
        """
        stripped = message.strip()

        if len(stripped) < 35:
            return True

        if _FOLLOW_UP_RE.match(stripped):
            return True

        if not history:
            return False

        return await self._llm_is_continuation(message, current_workflow, history)

    async def _llm_is_continuation(
        self,
        message: str,
        current_workflow: str,
        history: list,
    ) -> bool:
        """LLM yes/no: does this message continue the active workflow?"""
        readable = [
            m for m in history[-6:]
            if isinstance(m, (HumanMessage, AIMessage))
            and not getattr(m, "tool_calls", None)
        ]
        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: "
            f"{m.content[:200].strip()}"
            for m in readable
        )
        wf_desc = self._workflows[current_workflow].description
        prompt = (
            f"Active workflow: {current_workflow} — {wf_desc}\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            f"New message: {message}\n\n"
            f"Is this a follow-up to the active workflow, or a completely new topic?\n"
            f"Reply with ONLY: follow-up  OR  new-topic"
        )
        response = self._fast_llm_client().invoke([HumanMessage(content=prompt)])
        return "follow-up" in response.content.lower()

    # ── Utilities ─────────────────────────────────────────────────────────────

    def workflow_names(self) -> list[str]:
        return list(self._workflows.keys())
