# Agentic Framework — Architecture

---

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Core Components](#core-components)
3. [DevOps Knowledgebase Workflow (v2)](#devops-knowledgebase-workflow-v2)
4. [Embedding and Vector Store](#embedding-and-vector-store)
5. [Conversation History](#conversation-history)
6. [LLM Configuration](#llm-configuration)
7. [Checkpointing](#checkpointing)
8. [Adding a New Workflow](#adding-a-new-workflow)
9. [Deployment Architecture Considerations: Monolith vs Multi-Tier](#deployment-architecture-considerations-monolith-vs-multi-tier)

---

## Framework Overview

A multi-workflow agentic assistant built on **Chainlit** (UI) and **LangGraph** (workflow engine). Workflows are declared entirely in YAML — no Python required to add new capabilities. The framework handles routing, streaming, tool calls, and conversation history automatically.

```
User (Chainlit UI)
      │
      ▼
 Intent Router          ← regex fast-path → LLM fallback routing
      │
      ▼
 Workflow (YAML)        ← sequential / parallel / loop / router agents
      │
      ▼
 LangGraph Graph        ← compiled StateGraph with checkpointing
      │
      ▼
 LLM Provider           ← OpenAI / Anthropic / Azure / Google (llm_config.yaml)
```

---

## Core Components

### `app.py` — Chainlit Entry Point

Manages per-session state: `current_workflow`, `history`, `awaiting_clarification`.

**Message routing logic (per turn):**
1. Detect explicit switch commands (`switch to jenkins`) → switch immediately
2. Detect list commands (`list agents`) → show workflow list
3. `awaiting_clarification=True + current_workflow set` → `route_with_context()` (sticky)
4. `awaiting_clarification=True + no workflow` → `route()` fresh
5. Normal → `route_with_context()`

**Sticky workflow rule**: Once a user switches to a workflow, they stay in it until they explicitly name a different workflow. Queries that don't match any workflow's intent patterns are handled by the current workflow, not routed to the general assistant. Only a clear, unambiguous match to a *different* workflow triggers a switch.

**Streaming modes** (controlled by `AGENT_STEPS` env var):
- `off` — TaskList progress only (default; cleanest for end users)
- `tools` — TaskList + collapsible `cl.Step` per tool call (name only)
- `verbose` — same, with truncated tool input/output shown

**Clarification early-stop**: When a pipeline step asks for clarification (`clarification_needed=True`), `stream_with_events` does not break out of the event loop early — it continues to the `done` event which reads the full state (including messages) from the checkpointer. This ensures the agent's "please provide X" message is always surfaced to the user.

### `framework/workflow/intent_router.py` — Routing

Two routing methods:
- `route(message)` — stateless; uses regex patterns first (fast), then LLM if ambiguous
- `route_with_context(message, current_workflow, history)` — adds sticky-workflow logic on top of `route()`

**Continuation detection**: Uses `_FOLLOW_UP_RE` regex patterns and message length heuristics to decide whether a message continues the current workflow before calling the LLM.

### `framework/workflow/workflow.py` — Workflow Runtime

- **Lazy compilation**: `StateGraph.compile()` is deferred to first use so startup is fast regardless of how many workflows are loaded
- **Streaming**: `stream_steps()` for TaskList-only mode; `stream_with_events()` for tool-visibility modes
- **History**: `extract_history()` keeps only HumanMessage + final AIMessage (no tool noise); `compact_history()` applies the configured compaction strategy

### `framework/agents/` — Agent Types

| Type | Description |
|---|---|
| `LLMAgent` | Non-deterministic; LLM decides actions via ReAct tool loop |
| `SequentialAgent` | Runs sub-agents one after another; stops early on `clarification_needed` or `error` |
| `ParallelAgent` | Runs all sub-agents concurrently with `asyncio.gather` |
| `LoopAgent` | Re-runs a single sub-agent until `metadata["done"]=True` or `max_iterations` |

**Clarification detection** (`make_agent_node` in `base.py`): After each agent step, checks the final AI message for question marks or "please provide"-style phrases. If found, sets `clarification_needed=True` in state — the `SequentialAgent` conditional edges route to `END`, and `app.py` pauses the pipeline and sets `awaiting_clarification=True`.

**Config forwarding**: Each agent node accepts a `RunnableConfig` second argument injected by LangGraph and forwards it to its internal `ainvoke()`. This propagates the callback context from `astream_events` into sub-graphs, making tool call events visible to the outer `stream_with_events` stream (required for `AGENT_STEPS=tools/verbose` to work).

### `framework/loader/` — YAML Loader

`YAMLLoader.load(path)` reads a `workflow.yaml`, resolves tool imports, constructs agent objects bottom-up (sub-agents before their parents), and returns a `Workflow` instance. Lazy compilation means this is fast at startup.

---

## DevOps Knowledgebase Workflow (v2)

**File**: `workflows/devops_kb_search/workflow.yaml`

### Design Goals

1. **Complete context**: Send full Confluence pages to the LLM, not fragments. Chunk-based RAG often misses critical information that appears in different sections of the same page.
2. **Content-level search**: RAG chunks capture section-level content, not just titles. A query about "reset Jenkins credentials" finds the specific section in a long guide even if the page title is generic.
3. **Efficient context use**: Full pages can be very large. Apply intelligent compression so context windows are not wasted on irrelevant sections.
4. **Follow-up Q&A without re-fetching**: Once a page is in the conversation context, follow-up questions are answered directly without additional Confluence API calls.

### Pipeline (4 agents)

```
query_analyzer_agent
        │  SEARCH_QUERY: <phrase>  OR  [FOLLOW-UP]
        ▼
page_locator_agent          tools: find_confluence_page_ids, fetch_confluence_page
        │  PAGE_RESULTS: page_id=X | title=... | score=... | matched_sections=[...]
        ▼
page_fetcher_agent          tool: fetch_page_by_id
        │  Full (possibly compressed) page content
        ▼
answer_agent
        │  Grounded answer with citations
        ▼
     User
```

### Stage 1 — Query Analysis

- Detects follow-up questions (pages already in history → outputs `[FOLLOW-UP]`)
- Extracts a 3–6 word search phrase optimised for vector similarity search
- No tools; temperature 0.0

**Design decision — separate query analysis from retrieval**: Keeping query analysis as its own pipeline stage gives the retrieval agent a clean, focused search phrase rather than a full natural-language question. This consistently improves vector search recall for conversational queries.

### Stage 2 — Page Location

**Primary path — Vector search** (`find_confluence_page_ids`):
- Searches Chroma with `top_k * 4` candidates to maximise recall before filtering
- Groups chunks by `page_id` (read from metadata, or parsed from `?pageId=` in URL for legacy ingestions that pre-date explicit `page_id` metadata storage)
- Deduplicates: multiple chunks from the same page → one entry with the best score and a list of matched section names
- Returns ranked `PAGE_RESULTS` block: page IDs, titles, scores, matched section names

**Fallback path — CQL search** (`fetch_confluence_page`):
- Used only when vector search returns `[NO PAGES FOUND]`
- Performs full-text CQL search against the live Confluence API
- Extracts page IDs from result URLs for the next stage

**Design decision — RAG as index, not retrieval**:
Chunks are stored in Chroma purely as a search index. Their text content is discarded after the page ID is extracted. This avoids the "fragment answer" problem where the LLM sees only a portion of the relevant page and must guess at what it doesn't have.

**Design decision — chunking still required**:
A single embedding per page averages the whole document's meaning, making it impossible to locate information in a specific section of a long page. Chunks give each section its own embedding — a query about "credential reset" finds the right section even in a page broadly titled "Jenkins Guide". The key difference from v1 is *what we do with chunk hits after retrieval*: we extract page IDs instead of returning chunk text to the LLM.

**Design decision — backward compatibility for existing Chroma stores**:
Old ingestions stored `page_id` only inside the URL string. The tool handles both: if `metadata["page_id"]` exists, use it directly; otherwise parse `pageId=` from the URL. New ingestions include `page_id` as an explicit metadata field.

### Stage 3 — Page Fetching

Calls `fetch_page_by_id(page_id, query)` for each page ID (up to 3, ranked by relevance score).

#### Content Compression Strategy (Strategy 1 — Query-Guided Section Extraction)

| Page size | Action |
|---|---|
| < 4 000 chars | Return full page — no compression |
| ≥ 4 000 chars + query present | Query-guided section extraction |
| ≥ 4 000 chars + no query | Return full page (no basis for scoring sections) |

**How section extraction works:**

1. Parse the full page HTML into sections using heading tags (h1–h6). Each section is `{heading, level, content}`.
2. Score each section using keyword overlap between query terms (stopwords excluded) and section text. Heading matches are weighted 2× since headings are concise topic labels.
3. Classify each section relative to the page's highest-scoring section:
   - **HIGH** (≥ 60% of top score): full section content included
   - **MEDIUM** (≥ 20% of top score): first 3 sentences + `...`
   - **LOW** (< 20% of top score): heading only (structure preserved)
4. Always prepend a Table of Contents listing every heading with indentation. This ensures the answer agent (and the user) can see the full page structure even when content is compressed.
5. Stop adding content at 8 000 characters; append `[N more sections omitted]`.

**Design decision — keyword scoring over re-embedding**:
Re-embedding each section at query time adds an extra embedding API call per page per query. Keyword overlap is a sufficient proxy for section relevance *within a page whose overall relevance has already been confirmed by vector search*. Zero added latency, zero cost.

**Design decision — TOC always included**:
Compressed pages always show every section heading. This is critical for follow-up Q&A: the answer agent can tell the user "the Rollback Procedure section was trimmed — ask me about it specifically" rather than leaving invisible gaps in the answer.

**Design decision — 3-page cap**:
3 pages × 8 000 chars ≈ 6 000 tokens of knowledge context, leaving ample room for the answer. Vector search ranks by relevance, so the top 3 cover the most likely sources. Users can ask follow-up questions if more pages are needed.

### Stage 4 — Answer Synthesis

- **Initial mode**: Synthesises a grounded answer from fetched page content. Cites exact URLs. Notes if a page was compressed (more sections available for follow-up).
- **Follow-up mode**: Full page content is in conversation history. Answers directly with zero tool calls.
- Temperature 0.1 for consistent, factual answers.

### Follow-up Q&A Mechanism

After the first answer, the fetched page content lives in conversation history as prior AI messages. On the next turn:

1. `query_analyzer_agent` outputs `[FOLLOW-UP]` (pages visible in history, question about the same topic)
2. `page_locator_agent` sees `[FOLLOW-UP]` → outputs `[FOLLOW-UP — SKIP RETRIEVAL]`, calls no tools
3. `page_fetcher_agent` sees skip signal → outputs `[FOLLOW-UP — PAGES ALREADY IN CONTEXT]`, calls no tools
4. `answer_agent` answers from in-context history

Follow-up Q&A is instant (zero Confluence API calls, zero vector search calls) and accurate (working from the exact same pages as the initial answer).

**Edge case — follow-up about a compressed section**: If the user asks about a section that was shown as heading-only, the answer agent flags this and invites the user to ask specifically. A future enhancement could add a `re_fetch_section` tool that fetches the full page again with a targeted section query.

---

## Embedding and Vector Store

**Provider**: controlled by `EMBEDDING_PROVIDER` env var (`local` or `openai`).

**Critical**: the same embedding model and endpoint must be used for both ingestion (`scripts/ingest_confluence.py`) and querying (`tools/confluence.py`). A mismatch produces silently wrong similarity scores with no error.

**Configuration**:
| Variable | Purpose |
|---|---|
| `EMBEDDING_PROVIDER` | `local` or `openai` |
| `EMBEDDING_MODEL` | Model name; empty = provider default (`text-embedding-3-small` for openai, `all-MiniLM-L6-v2` for local) |
| `EMBEDDING_BASE_URL` | Custom API endpoint (e.g. a company gateway serving `bge-m3`) |
| `EMBEDDING_API_KEY` | API key if the embedding endpoint differs from the LLM endpoint |

**Design decision — separate `EMBEDDING_BASE_URL` from `OPENAI_BASE_URL`**:
`OpenAIEmbeddings` reads `OPENAI_API_BASE` from the environment while the LLM layer uses `OPENAI_BASE_URL`. These can point to different services (e.g. an LLM gateway vs. a dedicated embedding endpoint). Passing `base_url` and `api_key` as explicit constructor kwargs to `OpenAIEmbeddings` prevents silent routing to the wrong endpoint regardless of what other `OPENAI_*` variables are set.

---

## Conversation History

**Extraction** (`extract_history`): keeps HumanMessage + final AIMessage per turn. Discards intermediate AIMessages with tool calls and ToolMessages (raw tool output) — these are large and add noise without improving future routing or answer quality.

**Compaction strategies** (controlled by `HISTORY_STRATEGY`):

| Strategy | Behaviour | Trade-off |
|---|---|---|
| `none` | Unbounded growth | Safe default; context eventually overflows on very long sessions |
| `window` | Keep last N messages | Simple, zero cost; older context hard-dropped |
| `summary` | LLM summary of older turns + recent tail verbatim | Best context preservation; one LLM call when threshold crossed |

---

## LLM Configuration

All providers and models are registered in `llm_config.yaml`. Adding a new provider or model requires only a YAML edit — no code changes.

**Routing model**: a separate (typically cheaper) model for intent routing. Set via `DEFAULT_ROUTING_MODEL`. Routing calls are short and don't require the most capable model.

---

## Checkpointing

A fresh `thread_id` is generated per message turn (not per session). This ensures the LangGraph checkpointer always starts from a clean state — stale `clarification_needed` flags or `task_results` from a previous turn never bleed into the current run. Conversation context is carried explicitly via the `history` parameter.

| Environment | Backend | Notes |
|---|---|---|
| `dev` | `MemorySaver` | In-process; lost on restart |
| `prod` | `AsyncPostgresSaver` | Persistent; requires `POSTGRES_URL` |

---

## Adding a New Workflow

1. Create `workflows/<name>/workflow.yaml` following the schema in `framework/loader/schema.py`
2. Declare tools, agents, and the entry agent
3. Add intent patterns so the router can direct traffic to it
4. Restart the app — the YAML is loaded automatically at startup

No Python code required unless your workflow needs a new tool (add it to `tools/`).

---

# Deployment Architecture Considerations: Monolith vs Multi-Tier

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
