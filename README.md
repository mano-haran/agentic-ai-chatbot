# Agentic Framework — DevOps AI Assistant

A **production-ready framework** for building DevOps chatbots powered by agentic AI workflows.
Built on top of [LangGraph](https://github.com/langchain-ai/langgraph) with a developer-friendly
layer that lets you define workflows in plain YAML **or** Python — your choice.

---

## Table of Contents

1. [What is this?](#1-what-is-this)
2. [Key Concepts](#2-key-concepts)
3. [Project Structure](#3-project-structure)
4. [Low-Level Architecture](#4-low-level-architecture)
5. [Agent Types](#5-agent-types)
6. [How State Flows Between Agents](#6-how-state-flows-between-agents)
7. [Defining a Workflow — YAML vs Python DSL](#7-defining-a-workflow--yaml-vs-python-dsl)
8. [Intent Routing — Single-Turn, Multi-Turn, and Clarification](#8-intent-routing--single-turn-multi-turn-and-clarification)
9. [The Jenkins Log Analysis Use Case — End-to-End Walk-Through](#9-the-jenkins-log-analysis-use-case--end-to-end-walk-through)
10. [Design Decisions](#10-design-decisions)
11. [Adding a New Workflow (Step-by-Step)](#11-adding-a-new-workflow-step-by-step)
12. [LLM Providers](#12-llm-providers)
13. [Getting Started](#13-getting-started)

---

## 1. What is this?

### The problem it solves

Modern DevOps teams deal with many repetitive tasks: investigating failed builds, scaling
Kubernetes pods, checking monitoring dashboards, etc.  Building a chatbot that can automate
these tasks usually requires deep knowledge of LangGraph or similar frameworks — lots of
boilerplate, graph wiring, and state management code.

This framework hides all of that complexity.  You describe **what** your agents should do
(in YAML or Python), and the framework figures out **how** to wire everything into a
LangGraph execution graph.

### What you get

- A **Chainlit chat UI** where users type natural language requests.
- An **Intent Router** that automatically picks the right workflow — users never have to
  select an assistant manually.
- A **workflow engine** that runs multi-step AI pipelines with tool calling, parallel
  execution, loops, and routing.
- A **simple developer API** — if you can write a YAML file you can build a workflow.

---

## 2. Key Concepts

Before diving into the code, here are the five building blocks you need to understand:

| Concept | Plain English | Analogy |
|---|---|---|
| **Tool** | A Python function that an agent can call (e.g. "fetch Jenkins log") | A button the AI can press |
| **Agent** | An AI "worker" with a role, optional tools, and optional sub-agents | An employee with a job description |
| **Workflow** | A named, runnable unit that wraps a root agent and carries intent patterns | A department that handles specific requests |
| **Intent Router** | Reads the user's message and picks the right Workflow | A receptionist |
| **AgentState** | A shared data bag that flows through the entire execution | A shared whiteboard everyone can read and write |

---

## 3. Project Structure

```
agentic_framework/
│
├── app.py                          ← Chainlit entry point (the chat UI)
├── config.py                       ← Environment variables (API keys, URLs)
├── requirements.txt
├── .env.example                    ← Copy to .env and fill in your keys
│
├── framework/                      ← The reusable wrapper library (don't edit often)
│   ├── __init__.py                 ← Public API: import everything from here
│   │
│   ├── core/
│   │   ├── state.py                ← AgentState — the shared data bag
│   │   └── context.py              ← RunContext — session metadata carrier
│   │
│   ├── agents/
│   │   ├── base.py                 ← BaseAgent ABC + make_agent_node() helper
│   │   ├── llm_agent.py            ← LLMAgent  (non-deterministic, ReAct loop)
│   │   ├── workflow_agents.py      ← SequentialAgent, ParallelAgent, LoopAgent
│   │   └── router_agent.py         ← RouterAgent (LLM-driven branching)
│   │
│   ├── tools/
│   │   └── decorators.py           ← @tool decorator + global tool registry
│   │
│   ├── workflow/
│   │   ├── workflow.py             ← Workflow class (run / stream)
│   │   └── intent_router.py        ← Routes user message → correct Workflow
│   │
│   └── loader/
│       ├── schema.py               ← Pydantic models that validate YAML files
│       └── yaml_loader.py          ← Parses YAML → Python agent/workflow objects
│
├── tools/                          ← Your tool implementations (one file per system)
│   └── jenkins.py                  ← Jenkins API tools
│
└── workflows/                      ← Your workflow definitions (one folder per use case)
    └── jenkins_log_analysis/
        ├── workflow.yaml           ← YAML definition (preferred for non-developers)
        └── workflow.py             ← Python DSL equivalent (preferred for developers)
```

### Rule of thumb

- **`framework/`** — Framework internals.  You rarely touch this unless you're adding a
  new agent pattern.
- **`tools/`** — Add a new file here when you want to connect a new external system
  (e.g. `tools/datadog.py`, `tools/kubectl.py`).
- **`workflows/`** — Add a new folder here for each new use case
  (e.g. `workflows/kubernetes/`, `workflows/monitoring/`).

---

## 4. Low-Level Architecture

### Layer diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Chainlit Frontend  (app.py)                        │
│   Handles chat UI, session, message streaming, step visibility               │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │  user message (plain text)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Intent Router                                        │
│   framework/workflow/intent_router.py                                         │
│                                                                               │
│   Stage 1 — Regex scan: collect ALL matching workflows  (free, instant)      │
│   Stage 2 — Decision: 0/1-short/1-long/2+ → trust / confirm / arbitrate     │
│   Stage 3 — LLM: ROUTE: <name>  or  CLARIFY: <question>  (~50ms)            │
│                                                                               │
│   Returns: RoutingDecision(workflow=...) or RoutingDecision(clarification=…) │
└──────┬────────────────┬───────────────────────┬─────────────────────────────┘
       │                │                       │
       ▼                ▼                       ▼
  jenkins_log_    kubernetes_wf          monitoring_wf        ...more workflows
  analysis_wf     (not yet built)        (not yet built)
       │
       │  Each workflow wraps one root agent
       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Framework Core  (framework/)                              │
│                                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  LLMAgent   │  │ Sequential  │  │  Parallel    │  │   RouterAgent    │  │
│  │  (ReAct     │  │ Agent       │  │  Agent       │  │   (LLM classify  │  │
│  │   loop)     │  │ (chain)     │  │  (gather)    │  │    + branch)     │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘  └────────┬─────────┘  │
│         └────────────────┴─────────────────┴───────────────────┘            │
│                                    │                                          │
│                          Each agent compiles to a                             │
│                          LangGraph StateGraph on first use                    │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LangGraph Engine                                       │
│   StateGraph / CompiledGraph — handles node execution, edge routing,         │
│   state merging, async streaming                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       External Systems                                        │
│   Jenkins API  │  Kubernetes  │  Datadog  │  PagerDuty  │  …                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How a user message travels through the system

```
User types: "Why did my jenkins build fail?"
               │
               ▼
         app.py: on_message()
          reads session: current_workflow, history, awaiting_clarification
               │
               ▼  (1) context-aware routing
         IntentRouter.route_with_context(message, current_workflow, history)
            → first message? → route() fresh
            → follow-up?     → keep current_workflow (no LLM cost)
            → new topic?     → route() fresh
               │
               │  route() decision table:
               │    regex → 0 matches     → LLM classifies from scratch
               │    regex → 1 match, short → trust regex (free)
               │    regex → 1 match, long  → LLM confirms with hint
               │    regex → 2+ matches     → LLM arbitrates
               │
               │  LLM may return:
               │    ROUTE: jenkins_log_analysis  → RoutingDecision(workflow=...)
               │    CLARIFY: "Jenkins or K8s?"   → RoutingDecision(clarification=...)
               │
               ├─ needs_clarification? ──► send question to user
               │                          save exchange to history
               │                          set awaiting_clarification = True
               │                          return  ← wait for user's next message
               │
               ▼  (2) run workflow
         workflow.run(user_message, history=session_history)
            → prepends full prior conversation to state
            → calls jenkins_analysis_pipeline.compile().ainvoke(state)
               │
               ▼  (3) SequentialAgent: log_fetcher_agent
            → calls get_jenkins_builds, fetch_build_log, get_build_info
            → writes log summary to messages
               │
               ▼  (4) SequentialAgent: log_analyzer_agent
            → reads ALL messages including fetched log
            → writes structured failure analysis
               │
               ▼  (5) SequentialAgent: fix_suggester_agent
            → reads everything: question + log + analysis
            → writes actionable fixes in Markdown
               │
               ▼  (6) surface result + persist history
         app.py: extracts last AIMessage → sends to Chainlit
                 extract_history(result) → saves to session
                   (keeps Human + final AI messages; drops tool noise)
               │
               ▼
         User sees formatted fix recommendations
```

---

## 5. Agent Types

The framework provides four agent types.  Two are **non-deterministic** (the LLM decides
what happens next) and three are **deterministic** (the execution path is fixed in code).

### LLMAgent — non-deterministic

The LLM is in control.  It decides which tools to call, in what order, and when to stop.

```
         ┌─────────────┐
 start ──►   agent     │  LLM thinks: "I need to call fetch_build_log"
         │  (LLM call) │
         └──────┬──────┘
                │ has tool calls?
          yes ──┤
                ▼
         ┌─────────────┐
         │    tools    │  tools execute, results added to messages
         │  (ToolNode) │
         └──────┬──────┘
                │
                └─── back to agent ────►  repeat until LLM says "I'm done"
                │
          no ───┘
                ▼
              END
```

**When to use:** Any step where the LLM needs to call tools, reason about results, or
delegate to sub-agents.

**Sub-agents as tools:** If you give an LLMAgent a list of `sub_agents`, each sub-agent
is automatically wrapped as a callable tool.  The LLM can invoke them by name, just like
a regular tool.  This enables tree-shaped agent hierarchies.

---

### SequentialAgent — deterministic

Runs sub-agents one after another in a fixed order.  Each agent sees everything the
previous agents wrote.

```
 start ──► agent_A ──► agent_B ──► agent_C ──► END
```

**When to use:** Multi-step pipelines where output from step N feeds into step N+1
(e.g. fetch → analyse → report).

---

### ParallelAgent — deterministic

Runs all sub-agents at the same time using `asyncio.gather`.  Results are collected in
`task_results` keyed by agent name.

```
              ┌──► agent_A ──┐
 start ───────┤──► agent_B ──├──► aggregate ──► END
              └──► agent_C ──┘
```

**When to use:** Independent checks that don't depend on each other
(e.g. check Jenkins + check Datadog + check Kubernetes simultaneously).

---

### LoopAgent — deterministic

Runs a single sub-agent repeatedly until a condition is met.

```
 start ──► sub_agent ──► check ──► sub_agent  (loop)
                              └──► END         (done or max_iterations reached)
```

**When to use:** Polling loops, retry logic, iterative refinement.

---

### RouterAgent — hybrid

Makes one LLM call to classify the user's intent, then routes to exactly one sub-agent.
No loop — classification is one-shot.

```
 start ──► router (LLM) ──► agent_A ──► END
                        └──► agent_B ──► END
                        └──► agent_C ──► END
```

**When to use:** Intra-workflow branching where you need to choose between specialised
sub-pipelines (e.g. "is this a deployment issue or a build issue?").

---

## 6. How State Flows Between Agents

Every agent reads from and writes to a single shared object called `AgentState`:

```python
# framework/core/state.py

class AgentState(TypedDict):
    messages:     list           # Full conversation — accumulated (never overwritten)
    next_agent:   str            # Routing signal used by RouterAgent
    task_results: dict[str, Any] # Each agent writes its output here, keyed by name
    metadata:     dict[str, Any] # Session-level data (flags, config, counters)
    error:        str | None     # Set by a node to signal failure upstream
```

### Why `messages` is the backbone

The `messages` list uses LangGraph's `add_messages` reducer — new messages are **appended**,
never replaced.  This means:

- `log_fetcher_agent` adds its tool calls + results to `messages`.
- `log_analyzer_agent` starts with that full history already in `messages`.
- `fix_suggester_agent` sees everything: the user's question, the raw log, and the analysis.

No explicit "pass output from A to B" wiring is needed.  Context flows naturally.

### `task_results` — explicit named outputs

While `messages` provides implicit context, `task_results` provides explicit, named access
to each agent's final output:

```python
result = await workflow.run("why did my build fail?")

# Access any agent's output directly
log_analysis = result["task_results"]["log_analyzer_agent"]
fix_report   = result["task_results"]["fix_suggester_agent"]
```

---

## 7. Defining a Workflow — YAML vs Python DSL

Both styles produce identical objects.  Choose based on who is defining the workflow.

### YAML (recommended for teams, non-developers, and config-driven deployments)

```yaml
# workflows/my_workflow/workflow.yaml

name: my_workflow
description: "What this workflow does — used by IntentRouter for LLM matching"
version: "1.0"

tools:
  - name: my_tool          # how agents reference this tool
    module: tools.myfile   # Python import path
    function: my_function  # attribute name in that module

agents:
  - name: my_agent
    type: llm              # llm | sequential | parallel | loop | router
    role: |
      You are an expert at ...
      Your task: 1. Do X  2. Do Y
    tools:
      - my_tool
    model: claude-sonnet-4-6

  - name: my_pipeline
    type: sequential
    sub_agents:
      - my_agent           # agent names, built bottom-up by the loader

entry_agent: my_pipeline

intents:
  - pattern: "keyword1|keyword2"    # regex, matched case-insensitively
    workflow: my_workflow
```

### Python DSL (recommended for developers who want IDE support and type checking)

```python
# workflows/my_workflow/workflow.py

from framework import LLMAgent, SequentialAgent, Workflow
from tools.myfile import my_function   # already a StructuredTool via @tool decorator

my_agent = LLMAgent(
    name="my_agent",
    role="You are an expert at ...",
    tools=[my_function],
    model="claude-sonnet-4-6",
)

my_pipeline = SequentialAgent(
    name="my_pipeline",
    sub_agents=[my_agent],
)

my_workflow = Workflow(
    name="my_workflow",
    description="What this workflow does",
    entry_agent=my_pipeline,
    intents=[{"pattern": r"keyword1|keyword2", "workflow": "my_workflow"}],
)
```

### Writing a tool

```python
# tools/myfile.py

from framework import tool

@tool(description="Fetch something from an external system")
def my_function(param1: str, param2: int = 5) -> str:
    """Detailed docstring — shown to the LLM so it knows when to use this tool."""
    # Call your API here
    return "result"
```

The `@tool` decorator does two things:
1. Wraps the function as a LangChain `StructuredTool` (the LLM can call it).
2. Registers it in a global registry so YAML files can reference it by name.

---

## 8. Intent Routing — Single-Turn, Multi-Turn, and Clarification

The `IntentRouter` sits between Chainlit and the workflows. Every user message passes
through it before any workflow runs.  It returns a `RoutingDecision` — not just a name.

### What is `RoutingDecision`?

```python
@dataclass
class RoutingDecision:
    workflow:      str | None   # name of the workflow to run
    clarification: str | None   # question to ask the user (when routing is uncertain)

    @property
    def needs_clarification(self) -> bool:
        return self.clarification is not None
```

Every routing call returns one of these two outcomes.  `app.py` always checks
`needs_clarification` before trying to run a workflow.

---

### Single-turn routing

When a message arrives the router runs two stages:

**Stage 1 — Regex scan (free, instant)**

Every workflow declares `intents` — a list of regex patterns. The router scans *all* of
them and collects *every* workflow whose pattern matches. It never stops at the first hit,
because collecting all matches is what makes ambiguity detectable.

**Stage 2 — Decision table**

| Regex result | Message length | Action |
|---|---|---|
| 0 matches | any | LLM classifies from scratch |
| 1 match | ≤ 8 words | Trust regex — short keyword queries are reliable |
| 1 match | > 8 words | LLM confirms with regex result as a hint |
| 2+ matches | any | LLM arbitrates — message spans multiple topics |

Why the word-count threshold?  Short messages (`"jenkins build failing"`, 3 words) are
keyword-heavy — the match is almost certainly correct.  Longer sentences often contain
keywords incidentally: *"I know jenkins is popular, but my kubernetes pods are crashing"*
matches the Jenkins pattern but should route to Kubernetes.

**Stage 3 — What the LLM is asked to produce**

The LLM is not asked "which workflow?".  It is asked to choose between two formats:

```
ROUTE: jenkins_log_analysis
```
or
```
CLARIFY: I can help with Jenkins CI/CD issues or Kubernetes troubleshooting.
         Are you dealing with a failing build, or a deployment/pod problem?
```

The `ROUTE:` prefix means "I'm confident — use this workflow."
The `CLARIFY:` prefix means "I don't have enough information — ask the user first."

When the regex provided a hint, the prompt tells the LLM: *"The keyword matcher suggests
`jenkins_log_analysis` — confirm or override."*  This keeps the LLM grounded while still
letting it catch false positives.

**Full single-turn flow**

```
Message: "I know jenkins is great but my k8s pods are crashing"
          │
          ▼  _scan_patterns() → ["jenkins_log_analysis"]  (1 match)
          │  word_count = 11 > 8  →  LLM confirms with hint
          │
          ▼  _llm_classify(message, hint="jenkins_log_analysis")
          │  LLM sees hint + both workflow descriptions + message
          │  LLM replies: "ROUTE: kubernetes"  (overrides the hint)
          │
          ▼  RoutingDecision(workflow="kubernetes", clarification=None)

Message: "help"
          │
          ▼  _scan_patterns() → []  (0 matches)
          │
          ▼  _llm_classify("help")
          │  LLM: not enough info to pick a workflow
          │  LLM replies: "CLARIFY: I can help with Jenkins build failures or
          │                Kubernetes issues. Which are you dealing with?"
          │
          ▼  RoutingDecision(workflow=None, clarification="I can help with...")
          │
          ▼  app.py: sends clarification question, sets awaiting_clarification=True
```

---

### Clarification flow

When `needs_clarification` is True, `app.py`:
1. Sends the question to the user.
2. Appends `[HumanMessage(original), AIMessage(question)]` to the session history.
3. Sets `awaiting_clarification = True` in the session.
4. Returns — does not run any workflow.

On the **next** user message, `app.py` sees `awaiting_clarification = True` and calls
`route()` directly instead of `route_with_context()`.  This skips the continuation check.
Without this flag, a short answer like `"jenkins"` (4 chars) would be misidentified as a
follow-up to the previous workflow by the `len < 35` heuristic.

```
Turn 1:  User: "help me with something"
         Router: CLARIFY → "Jenkins builds or Kubernetes pods?"
         Session: awaiting_clarification = True

Turn 2:  User: "jenkins"
         app.py sees awaiting_clarification=True → calls route("jenkins") directly
         _scan_patterns → 1 match, 1 word ≤ 8 → trust regex
         RoutingDecision(workflow="jenkins_log_analysis") ✓
         Session: awaiting_clarification = False
```

---

### Multi-turn routing

After the first message, subsequent messages go through `route_with_context()` which
first decides whether the message continues the current workflow before doing any regex
or LLM work.

**Step 1 — Heuristics (free)**

| Condition | Decision |
|---|---|
| Message < 35 characters | Follow-up — stay in current workflow |
| Starts with a follow-up phrase (`"what about"`, `"can you"`, `"also"` …) | Follow-up |

**Step 2 — LLM disambiguation (cheap, ~50ms)**

For longer messages that don't match the heuristics, the router sends the last few
conversation exchanges plus the new message to the LLM and asks:

> *"Is this a follow-up to the active workflow, or a completely new topic?
> Reply with ONLY: follow-up OR new-topic"*

This uses the fast LLM client (16-token output limit) — the task is a binary yes/no.

**Two LLM clients**

| Client | Token limit | Used for |
|---|---|---|
| `_fast_llm_client` | 16 tokens | Continuation yes/no — binary answer |
| `_classify_llm_client` | 256 tokens | ROUTE:/CLARIFY: with potential long question |

Keeping them separate avoids paying for extra tokens on the cheap continuation check.

**Full multi-turn example**

```
Turn 1:  "my jenkins build is failing"
         → first message, no context → route() → regex match → jenkins ✓

Turn 2:  "what about the second error?"
         → len < 35 → follow-up heuristic fires → stay on jenkins (no LLM call) ✓

Turn 3:  "I'd also like to check why my production pods are restarting"
         → len > 35, no follow-up prefix → LLM disambiguation
         → LLM: "new-topic" → route() fresh → kubernetes ✓
         → app.py shows "Switching topic → kubernetes"

Turn 4:  "ok"
         → len < 35 → follow-up → stay on kubernetes ✓
```

---

### Declaring intents in YAML and Python DSL

**YAML:**
```yaml
intents:
  - pattern: "jenkins|build.fail|pipeline.fail"
    workflow: jenkins_log_analysis
  - pattern: "why.*(build|pipeline|job)"
    workflow: jenkins_log_analysis
```

**Python DSL:**
```python
intents=[
    {"pattern": r"jenkins|build.fail", "workflow": "jenkins_log_analysis"},
    {"pattern": r"why.*(build|pipeline)", "workflow": "jenkins_log_analysis"},
]
```

Patterns are Python `re.search` expressions matched case-insensitively.
The `.` in `build.fail` matches any character (space, hyphen, underscore) so it covers
`"build fail"`, `"build-fail"`, `"build_fail"` etc.

Write patterns that are **specific**, not broad.  A pattern like `"jenkins"` alone will
match any message mentioning Jenkins, including ones about other topics.  Prefer
`"jenkins.*(fail|error|log|build)"` — it only fires when Jenkins + a problem word appear
together.  The LLM stage handles vague messages; regex should only fire when it's certain.

---

## 9. The Jenkins Log Analysis Use Case — End-to-End Walk-Through

### What it does

When a user says anything like "why did my Jenkins build fail?", the framework:
1. Fetches the most recent failed build's logs and metadata from Jenkins.
2. Analyses the logs to classify the failure type and identify the root cause.
3. Suggests specific, actionable fixes with code snippets.

### Agent hierarchy

```
jenkins_analysis_pipeline  (SequentialAgent)
├── log_fetcher_agent      (LLMAgent  + 3 tools)
│       tools: get_jenkins_builds, fetch_build_log, get_build_info
├── log_analyzer_agent     (LLMAgent, no tools)
└── fix_suggester_agent    (LLMAgent, no tools)
```

### Step-by-step execution

**Step 1 — `log_fetcher_agent`**

The LLM receives the user's question plus the role:
> *"You are a Jenkins automation expert. Fetch logs for the failing build..."*

It reasons:
1. "I need to know recent builds" → calls `get_jenkins_builds("my-service")`
2. "Build 142 is FAILURE" → calls `fetch_build_log("my-service", 142)`
3. "I also need metadata" → calls `get_build_info("my-service", 142)`
4. "I have everything" → writes a summary, stops looping

After this step, `messages` contains: user question + 3 tool calls + 3 tool results +
the fetcher's summary.

**Step 2 — `log_analyzer_agent`**

Receives the full `messages` history.  Its role says:
> *"Analyse the log. Produce: Failure Classification, Failed Stage, Error Messages,
> Root Cause, Affected Components."*

It produces a structured analysis like:
```
## Failure Classification
Type: compilation error

## Failed Pipeline Stage
Build (Maven compile phase)

## Error Messages
[ERROR] UserService.java:[87,32] error: cannot find symbol
        symbol: method getUserById(Long)
...

## Root Cause
UserRepository interface was refactored — the getUserById(Long) method was
renamed or removed, but UserService.java was not updated.
```

This is appended to `messages`.  No tool calls needed.

**Step 3 — `fix_suggester_agent`**

Receives everything: user question + raw logs + analysis.  Produces:
```markdown
## Immediate Fix

1. Open `UserService.java` line 87 and update the method call:
   ```java
   // Before (broken)
   User user = userRepository.getUserById(userId);

   // After (fixed)
   User user = userRepository.findById(userId).orElseThrow(...);
   ```

2. Repeat for OrderService.java lines 134 and 156...

## Why This Happened
The repository interface was refactored without updating all callers.

## Prevention
Add a `@Deprecated` annotation with a migration note before removing methods.
Consider using ArchUnit tests to enforce interface contracts.

## Effort Estimate
quick-fix (< 1 hour)
```

### Sample conversation

```
User:   "My jenkins build for my-service has been failing since this morning"

Bot:    [routing to: jenkins_log_analysis]
        [log_fetcher_agent: fetched build #142 logs and metadata]
        [log_analyzer_agent: identified 3 compilation errors in UserService and OrderService]

        ## Immediate Fix
        The build is failing due to a compilation error introduced in commit
        "Refactor UserService to use new repository interface"...

        [full fix report in Markdown]
```

---

## 10. Design Decisions

### Why LangGraph instead of something simpler?

LangGraph gives us:
- **Proper async streaming** — tokens stream to the UI as the LLM generates them.
- **Built-in state management** — no manual passing of results between steps.
- **Production-grade execution** — retries, checkpointing, and observability hooks available.
- **Composable graphs** — a compiled sub-graph is just a callable; it can be used as a
  node in any parent graph.

The downside is that LangGraph has a steep learning curve.  This framework hides that.

### Why is the Python DSL structured the way it is?

The mental model — agents, tools, sub-agents — maps cleanly to DevOps use cases.
This framework compiles to LangGraph, giving better streaming and Python ecosystem
integration while keeping the authoring surface simple and readable.

### Why two authoring formats (YAML + Python)?

| Who | Preferred format |
|---|---|
| Developer building a new workflow | Python — IDE autocomplete, type hints, refactoring |
| Ops team defining a simple workflow | YAML — no Python needed, readable config file |
| Platform team deploying across environments | YAML — workflow definitions become config, not code |

Both formats produce the same `Workflow` object.  You can start with Python during
development and convert to YAML for production deployment.

### Why does SequentialAgent pass the full message history to each sub-agent?

Because LLMs use conversation history as context.  When `log_analyzer_agent` sees all
the prior messages including the raw Jenkins log, it can reference specific error lines
without needing explicit data passing.  This keeps the wiring simple and makes agents
more capable.

The `task_results` dict exists for cases where you need **programmatic access** to a
specific agent's output (e.g. to pass a value into a tool call, or to display in the UI).

### Why does the IntentRouter use regex first, then LLM?

Regex is instant and free.  Most DevOps requests are repetitive and well-suited to
simple keyword patterns.  The LLM fallback handles ambiguous or verbose requests where
keywords aren't reliable.  Using the cheapest model for routing keeps latency under
~100ms and cost near zero.

### Why collect all regex matches instead of stopping at the first?

The old "first match wins" approach had two silent failure modes:

1. **False positive** — `"jenkins"` fires on *"I know jenkins is popular but my pods are
   crashing"* and the LLM never gets a chance to correct it.
2. **First-match bias** — when two patterns match, the winner was determined by
   registration order, not relevance.

Collecting all matches before deciding lets the router detect *ambiguity* (≥ 2 matches)
and *incidental keywords* (1 match in a long sentence) — both cases where the LLM should
have the final say.

### Why does the router return `RoutingDecision` instead of just a string?

A string return type forces callers to invent conventions for "I don't know" — empty
string, `None`, a special sentinel.  `RoutingDecision` makes the two outcomes explicit
and type-safe.  Every call site is forced to check `needs_clarification` before
attempting to run a workflow, so it's impossible to accidentally run the wrong one on
an unresolved intent.

### Why is there an `awaiting_clarification` session flag?

After the router asks a clarifying question, the user's short answer (`"jenkins"`,
`"the build"`) would trigger the multi-turn continuation heuristic `len < 35` and be
classified as a follow-up to whichever workflow was active *before* the clarification.
The flag bypasses that heuristic and forces fresh routing, so the answer is evaluated on
its own merits.

### Why two separate LLM clients in IntentRouter?

| Client | Tokens | Task |
|---|---|---|
| `_fast_llm_client` | 16 | Continuation yes/no — one word answer |
| `_classify_llm_client` | 256 | Route or write a clarification question |

Forcing every call through a 256-token client wastes cost on the continuation check,
which only ever needs `"follow-up"` or `"new-topic"` as output.  Keeping them separate
means you pay for capacity only when you actually need it.

### Why does `_llm_classify` pass the regex hint but not enforce it?

When exactly one workflow matched a long message, the regex result is *probably* right
but not guaranteed.  Passing it as a hint keeps the LLM grounded (it doesn't start from
scratch) while still allowing an override.  If the hint were enforced, the whole point of
calling the LLM would be lost.

### Why are sub-agents wrapped as tools in LLMAgent?

This is the core of the sub-agent pattern.  When an LLMAgent has sub-agents, the LLM
can choose to delegate work to them just by calling them by name — the same way it calls
any other tool.  This gives the LLM full control over orchestration while keeping the
sub-agents self-contained and testable.

---

## 11. Adding a New Workflow (Step-by-Step)

Let's say you want to add a **Kubernetes troubleshooting** workflow.

**Step 1 — Create the tools**

```python
# tools/kubectl.py

from framework import tool

@tool(description="Get the status of pods in a Kubernetes namespace")
def get_pods(namespace: str = "default") -> str:
    """Returns pod names, statuses, and restart counts."""
    # kubectl get pods -n {namespace} -o json
    ...

@tool(description="Get logs from a Kubernetes pod")
def get_pod_logs(pod_name: str, namespace: str = "default", tail: int = 100) -> str:
    """Returns the last N lines of logs from a pod."""
    ...
```

**Step 2 — Create the workflow folder**

```
workflows/
└── kubernetes/
    ├── workflow.yaml    ← define here
    └── __init__.py
```

**Step 3 — Write the YAML**

```yaml
name: kubernetes
description: "Troubleshoot Kubernetes pods: check status, fetch logs, diagnose issues"
version: "1.0"

tools:
  - name: get_pods
    module: tools.kubectl
    function: get_pods
  - name: get_pod_logs
    module: tools.kubectl
    function: get_pod_logs

agents:
  - name: k8s_agent
    type: llm
    role: |
      You are a Kubernetes expert.
      Help the user troubleshoot pod issues by fetching status and logs.
    tools:
      - get_pods
      - get_pod_logs
    model: gpt-4o            # or claude-sonnet-4-6, gemini-2.0-flash, etc.
    # provider: openai       # optional — inferred from model name

entry_agent: k8s_agent

intents:
  - pattern: "kubernetes|kubectl|pod.*crash|pod.*fail|k8s|deployment.*fail"
    workflow: kubernetes
  - pattern: "why.*pod|pod.*restart|crashloopbackoff|oomkilled"
    workflow: kubernetes
```

**Step 4 — That's it.**

`app.py` auto-discovers all `workflow.yaml` files in the `workflows/` directory on startup.
Your new workflow is live.

---

## 12. LLM Providers

The framework separates *what model you use* from *how the code talks to it*.
All agents go through a single factory (`framework/providers/factory.py`) so
switching providers is a config change, not a code change.

### Supported providers

| Provider | Value | Covers |
|---|---|---|
| `openai` | `DEFAULT_PROVIDER=openai` | OpenAI + any OpenAI-compatible API |
| `azure` | `DEFAULT_PROVIDER=azure` | Azure OpenAI Service |
| `anthropic` | `DEFAULT_PROVIDER=anthropic` | Anthropic Claude |
| `google` | `DEFAULT_PROVIDER=google` | Google Gemini |

### Provider auto-detection from model name

You rarely need to set `provider` explicitly.  The factory infers it:

| Model prefix | Resolved provider |
|---|---|
| `gpt-*`, `o1*`, `o3*`, `o4*` | `openai` |
| `claude-*` | `anthropic` |
| `gemini-*` | `google` |
| anything else | `DEFAULT_PROVIDER` from `.env` |

### OpenAI-compatible endpoints (Groq, Ollama, vLLM, Together AI …)

Set `OPENAI_BASE_URL` in `.env` — no code changes needed:

```bash
# Groq
OPENAI_API_KEY=gsk_...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
DEFAULT_MODEL=llama3-70b-8192

# Ollama (local)
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama          # Ollama ignores this but the client requires it
DEFAULT_MODEL=llama3.2
```

### Per-agent provider override

Mix providers within one workflow — set `provider` on individual agents:

```yaml
# YAML
agents:
  - name: fast_classifier
    type: llm
    model: gpt-4o-mini          # cheap, fast
    provider: openai

  - name: deep_analyzer
    type: llm
    model: claude-sonnet-4-6    # more capable for complex analysis
    provider: anthropic
```

```python
# Python DSL
from framework import LLMAgent

fast_agent = LLMAgent(name="fast", role="...", model="gpt-4o-mini")
deep_agent = LLMAgent(name="deep", role="...", model="claude-sonnet-4-6", provider="anthropic")
```

### Installing provider packages

`requirements.txt` includes only `langchain-openai` by default.
Uncomment what you need:

```bash
pip install langchain-openai       # OpenAI / OpenAI-compatible (already included)
pip install langchain-anthropic    # Anthropic Claude
pip install langchain-google-genai # Google Gemini
```

---

## 13. Getting Started

### Prerequisites

- Python 3.11+
- An API key for your chosen provider (OpenAI by default)

### Install

```bash
git clone <repo-url>
cd agentic_framework

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY (or your provider's key)
```

### Run

```bash
chainlit run app.py
```

Open [http://localhost:8000](http://localhost:8000) and ask:

> *"My Jenkins build for my-service has been failing since this morning, can you check?"*

### Project dependencies at a glance

| Package | Purpose |
|---|---|
| `chainlit` | Chat UI, message streaming, step visibility |
| `langgraph` | Graph execution engine, state management |
| `langchain-openai` | OpenAI + OpenAI-compatible LLM integration (default) |
| `langchain-anthropic` | Anthropic Claude *(optional)* |
| `langchain-google-genai` | Google Gemini *(optional)* |
| `langchain-core` | StructuredTool, messages, ToolNode |
| `pydantic` | YAML schema validation |
| `pyyaml` | YAML file parsing |
| `python-dotenv` | Load `.env` file |
