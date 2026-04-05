# Architecture Considerations: Monolith vs Multi-Tier

## The State Problem First

Before the tier breakdown — this is the central risk. Session state currently lives in `cl.user_session`:

```
current_workflow, history, awaiting_clarification
```

`history` grows as a list of LangChain `BaseMessage` objects. If you split tiers, **this state must leave the process**. LangGraph has native support for this via checkpointers (`MemorySaver` → `AsyncRedisSaver` / Postgres), so it's solvable — but it's the first thing to design around, not an afterthought.

---

## Option A: Full Monolith (Current)

**Pros**
- Zero network hops — tool calls are in-process function calls
- `cl.user_session` just works — no external store needed
- Single deployment, single log stream, easy local dev
- Debuggable end-to-end in one process

**Cons**
- Can't scale tiers independently — a CPU-heavy tool call starves the UI
- One crash takes everything down
- Tools are tightly coupled to this app — can't reuse across a Slack bot, CLI, or API
- No isolation boundary between user-facing code and privileged tool operations (e.g. Jenkins API calls)

---

## Option B: UI Layer + Agent Core (2-Tier)

Split Chainlit from the orchestration layer. Agent Core exposes a REST / WebSocket / SSE API.

**Pros**
- Multiple UI surfaces can share one agent core (web, Slack bot, CLI, REST API)
- Scale Chainlit instances independently from agent workers
- Swap Chainlit for a different frontend without touching orchestration logic
- Agent Core can be deployed on a private network; UI can be public-facing

**Cons**
- `history` must be serialized over the wire on every request — `BaseMessage` objects become JSON payloads
- `cl.user_session` state (especially `awaiting_clarification`) must move to Redis or a DB session store
- Streaming tokens across a network boundary requires SSE or WebSocket — adds protocol complexity
- Harder to debug — distributed tracing needed

**State mitigation:** LangGraph's `AsyncRedisSaver` checkpointer handles agent state. Thin routing state (`current_workflow`, `awaiting_clarification`) moves to Redis with session IDs. History becomes part of the checkpoint — you stop passing it explicitly.

---

## Option C: 3-Tier (UI → Agent Core → MCP Gateway → MCP Servers)

MCP is the right abstraction for tools, but it changes the execution model.

**Pros**
- Tools become truly reusable services — Jenkins tools can serve multiple agents/apps
- Language-agnostic — a Jenkins MCP server could be written in Go while the analysis agent stays Python
- Process isolation — a bug in a tool server doesn't crash the agent core
- Tools can be scaled, versioned, and deployed independently
- Security boundary — tools that call internal APIs live on the private network; MCP gateway controls access
- Standard protocol — LangChain/LangGraph already has MCP client support

**Cons**
- Every tool call is now a network round-trip — currently `get_jenkins_builds()` is a local function call; over MCP it's an HTTP/SSE request. For agents with 5–10 tool calls per turn, latency adds up
- MCP gateway is a new single point of failure and ops burden
- Streaming large tool results (e.g. log files) is more complex across MCP than returning a local string
- Debugging a ReAct loop that crosses 3 network boundaries requires distributed tracing from day one
- MCP server management means separate repos, CI/CD pipelines, and health checks per tool server

> **On the MCP gateway specifically:** it solves tool discovery and auth centralization well, but adds a hop. Whether it's worth it depends on how many MCP servers you plan to run. For 2–3 tool groups, direct MCP client connections from agent core are simpler than adding a gateway.

---

## Checkpointer Options by Weight

Valkey (like Redis) is a separate server process — it's written in C and must be installed and run independently. It cannot be embedded inside a Python process.

| Option | How it runs | Good for |
|---|---|---|
| `MemorySaver` (LangGraph built-in) | In-process, no install | Dev / single instance |
| `SqliteSaver` (LangGraph built-in) | In-process, file-based | Dev / single instance, persistent |
| Valkey / Redis | Separate server process | Production, multi-instance |
| Postgres (`AsyncPostgresSaver`) | Separate server process | Production, multi-instance |

**Practical advice:**
- **Dev:** use `MemorySaver` — zero setup, lives in RAM, lost on restart
- **Dev with persistence:** `SqliteSaver` writes to a `.db` file — still single-process, no install needed
- **Prod:** move to Valkey/Redis or Postgres only when you need multiple app instances or session state to survive restarts

`SqliteSaver` is often underrated — it gives you persistent checkpointing with no infrastructure, which is likely good enough until you're running multiple instances in production.

---

## Why You Need a Checkpointer

LangGraph's checkpointer serves two distinct purposes:

### 1. State Persistence Across Invocations

Without a checkpointer, every `.invoke()` call starts from scratch. With one, graph state is saved after each node execution, keyed by a `thread_id`.

```python
# Without checkpointer — stateless, each call is independent
graph.invoke({"messages": [...]})

# With checkpointer — stateful across calls
graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "123"}})
graph.invoke({"messages": [...]}, config={"configurable": {"thread_id": "123"}})  # continues from prior state
```

### 2. Human-in-the-Loop (Interrupts)

Checkpointing is **required** for `interrupt()` / `interrupt_before` / `interrupt_after` to work. When the graph pauses waiting for human input, it needs somewhere to save the in-flight state so it can resume later.

```python
# Graph pauses here, saves state to checkpointer
value = interrupt("Please approve this action")

# Later, resume with human's response
graph.invoke(Command(resume="approved"), config=...)
```

### What Each Checkpoint Gives You

| Capability | How checkpointer enables it |
|---|---|
| Multi-turn conversations | State persisted between user messages |
| Fault recovery | Re-run from last checkpoint if a node crashes |
| Time travel / replay | `graph.get_state_history(config)` lets you rewind |
| Parallel threads | Different `thread_id`s are fully isolated |

### Common Backends

```python
from langgraph.checkpoint.memory import MemorySaver       # in-process, dev only
from langgraph.checkpoint.sqlite import SqliteSaver        # local persistence
from langgraph.checkpoint.postgres import PostgresSaver    # production
```

> **TL;DR:** You need a checkpointer whenever your graph needs to be stateful across multiple calls, support human-in-the-loop interrupts, or recover from failures. Without it, the graph is purely stateless and each invocation is isolated.

---

## Practical Recommendation: Phased Approach

```
Phase 1 (now):          Chainlit + Agent Core + Tools  →  single process
                        Add LangGraph checkpointer now so state is already external

Phase 2 (when needed):  Split UI from Agent Core
                        Trigger: you want a Slack bot or REST API alongside the web UI

Phase 3 (when needed):  Move tools to MCP servers (no gateway yet)
                        Trigger: tools need to be shared across multiple agent apps,
                        or you want process isolation for privileged operations

Phase 4 (if justified): Add MCP gateway
                        Trigger: 5+ MCP servers, need centralized auth/rate-limiting/discovery
```

**The one thing to do now:** add a LangGraph checkpointer (`MemorySaver` in dev, `AsyncPostgresSaver` in prod) even while monolithic. This makes the agent core stateless at the application level — state lives in the store, not the process. That single change makes every future split much easier because you never have to solve "where does the state go" later under pressure.

The layering (UI → Orchestration → Tools) is the right end state. The question is just sequencing — don't pay the distributed systems tax until the benefit is concrete.
