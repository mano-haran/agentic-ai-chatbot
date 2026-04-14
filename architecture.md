# Agentic Framework — Architecture

---

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Core Components](#core-components)
3. [DevOps Knowledgebase Workflow](#devops-knowledgebase-workflow)
4. [RAG Improvement Stack](#rag-improvement-stack)
5. [Embedding and Vector Store](#embedding-and-vector-store)
6. [Conversation History](#conversation-history)
7. [LLM Configuration](#llm-configuration)
8. [Checkpointing](#checkpointing)
9. [Adding a New Workflow](#adding-a-new-workflow)
10. [Deployment Architecture Considerations: Monolith vs Multi-Tier](#deployment-architecture-considerations-monolith-vs-multi-tier)

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

## DevOps Knowledgebase Workflow

**File**: `workflows/devops_kb_search/workflow.yaml`

### Design Goals

1. **Complete context**: Send the full Confluence page to the LLM, not fragments. Chunk-based RAG often misses critical information that appears in different sections of the same page.
2. **Content-level search**: RAG chunks capture section-level content, not just titles. A query about "reset Jenkins credentials" finds the specific section in a long guide even if the page title is generic.
3. **Any question about the page**: Because the LLM receives the entire page, users can ask any follow-up about any section — including material the original search phrase did not mention.
4. **Minimal round trips**: Retrieval and page fetch are combined into a single atomic tool call so the happy path is exactly two LLM calls.

### Pipeline (2 agents)

```
retriever_agent             tool: search_and_fetch_pages(query, top_k=1)
        │   1. Vector search on Chroma (+ optional BM25 + reranker) → page_id
        │   2. Fetches the FULL page by that page_id from Confluence
        │   PAGES_FETCHED: 1  +  full page text (title, URL, body)
        ▼
answer_agent
        │   Grounded answer with inline citations — uses the complete page verbatim
        ▼
     User
```

### Stage 1 — Retrieve (inline query rewrite + atomic fetch)

The retriever distils the user's latest message (resolving follow-up references against the prior conversation) into a focused 3–8 word search phrase and calls `search_and_fetch_pages(query=<phrase>, top_k=1)` exactly once. That single tool call performs:

1. **Vector search** on Chroma to rank candidate chunks, grouped by `page_id` (read from metadata, or parsed from `?pageId=` in URL for legacy ingestions).
2. **Optional BM25 + reranker** layered on top of vector results via Reciprocal Rank Fusion and a cross-encoder.
3. **Full-page fetch** from Confluence for the best-matching page ID.
4. Returns the complete page (title, URL, body as plain text) wrapped in a `PAGES_FETCHED: N` envelope.

**Fallback path — CQL search** (`fetch_confluence_page`):
- Triggered inside `search_and_fetch_pages` only when vector search returns `[NO PAGES FOUND]` (e.g. because the Chroma store is empty or all scores fall below `SIMILARITY_THRESHOLD`).
- Performs full-text CQL search against the live Confluence API, picks the top result and fetches it.

**Design decision — RAG as index, not retrieval**:
Chunks are stored in Chroma purely as a search index. Their text content is discarded after the page ID is extracted. This avoids the "fragment answer" problem where the LLM sees only a portion of the relevant page and must guess at what it doesn't have.

**Design decision — chunking still required**:
A single embedding per page averages the whole document's meaning, making it impossible to locate information in a specific section of a long page. Chunks give each section its own embedding — a query about "credential reset" finds the right section even in a page broadly titled "Jenkins Guide". The key difference from chunk-answer RAG is *what we do with chunk hits after retrieval*: we extract the page ID and fetch the complete page, instead of returning chunk text to the LLM.

**Design decision — atomic `search_and_fetch_pages`**:
Combining search and fetch into one tool removes an entire LLM ↔ tool round trip. The retriever agent makes one tool call and returns the output verbatim; the answer agent reads the full page from the shared message state. Happy path = two LLM calls total.

**Design decision — backward compatibility for existing Chroma stores**:
Old ingestions stored `page_id` only inside the URL string. The tool handles both: if `metadata["page_id"]` exists, use it directly; otherwise parse `pageId=` from the URL. New ingestions include `page_id` as an explicit metadata field.

### Stage 2 — Answer Synthesis

- Uses the FULL page content produced by the retriever — every section is available verbatim.
- Answers the user's question using only the fetched page (no prior knowledge, no invented details).
- Can answer questions about any section, including follow-ups on topics the search phrase did not mention.
- If the page genuinely does not contain an answer, says so clearly rather than guessing.
- Emits inline citations `[1]`, `[2]` keyed to a Sources block with URLs copied verbatim from the fetched content.
- No tools; temperature 0.1 for consistent, factual answers.

### Follow-up Q&A Mechanism

Follow-ups are handled naturally by the retriever: it sees the prior conversation and either (a) resolves a pronoun/reference and re-searches for the same page, reusing the Confluence page cache on the tool side, or (b) fetches a different page if the user has pivoted to a new topic. Because the complete page is always delivered to the answer agent, follow-up questions about any section of the current page are answered correctly without needing a separate "skip retrieval" branch.

---

## RAG Improvement Stack

The RAG pipeline can be progressively enhanced by enabling three optional layers on top of the base vector search. Each layer is independently toggled via environment variables and is fully air-gapped safe.

### Full Stack Diagram

```
INGESTION  (scripts/ingest_confluence.py)
─────────────────────────────────────────────────────────────────────────────
  Confluence API / Local files
          │
          ▼
  ┌─────────────────────────────────────────────────────────┐
  │  CHUNKING STRATEGY  (CHUNKING_STRATEGY=html|docling)     │
  │                                                          │
  │  html (default)          docling                         │
  │  ─────────────           ─────────────────────────────   │
  │  BeautifulSoup           docling HybridChunker           │
  │  heading-based           • Tables → atomic unit          │
  │  sections                • Code   → atomic unit          │
  │                          • Heading breadcrumbs in text   │
  │                          • Token-aware chunk sizing      │
  └────────────────────────────┬────────────────────────────┘
                               │ chunks (text + metadata)
              ┌────────────────┴────────────────┐
              ▼                                 ▼
     Chroma vector store              BM25 index (data/bm25.pkl)
     (embed + persist)                (ENABLE_BM25=true)
     always built                     rank-bm25, pure Python


RETRIEVAL  (tools/confluence.py → find_confluence_page_ids)
─────────────────────────────────────────────────────────────────────────────
  User query
      │
      ├─────────────────────────────────────────────┐
      │                                             │
      ▼                                             ▼
  Vector search                            BM25 keyword search
  (Chroma cosine similarity)               (ENABLE_BM25=true)
  always active                            rank_bm25.BM25Okapi
      │                                             │
      │  ranked page_id list                        │  ranked page_id list
      └──────────────────────┬──────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   RRF MERGE     │  (ENABLE_BM25=true)
                    │  Reciprocal     │  score(page) = Σ 1/(k + rank)
                    │  Rank Fusion    │  k = RRF_K (default 60)
                    └────────┬────────┘
                             │ merged page_id list (up to RERANKER_CANDIDATE_K)
                             ▼
                    ┌─────────────────────────────────────────┐
                    │          RERANKER                        │
                    │  RERANKER_PROVIDER=                      │
                    │                                          │
                    │  none (default)                          │
                    │    pass-through, keep RRF order          │
                    │                                          │
                    │  openai_compatible                       │
                    │    POST /v1/rerank (httpx, no SDK)       │
                    │    model: mxbai-rerank-large-v1          │
                    │                                          │
                    │  local                                   │
                    │    sentence-transformers CrossEncoder     │
                    │    air-gapped: model from models/ dir    │
                    └────────┬────────────────────────────────┘
                             │ top RERANKER_TOP_N page IDs
                             ▼
                    retriever_agent
                    (search_and_fetch_pages → full page → answer)
```

### Layer-by-Layer Explanation

#### Layer 1 — Context-Aware Chunking (Docling)

**Problem with default HTML chunking**: BeautifulSoup splits pages at heading boundaries. A table row spanning multiple `<tr>` elements may be split into separate chunks, breaking its meaning. Same for multi-line code blocks.

**Docling solution**: `HybridChunker` treats tables and code as atomic units that are never split. Every chunk carries heading breadcrumbs (e.g. "Installation > Prerequisites > Python Setup") so the embedding captures the full hierarchical context, not just the local paragraph text. Chunk sizes are token-aware, avoiding the very-small and very-large extremes that hurt retrieval quality.

**Air-gapped configuration**: Docling's HTML backend requires no ML models for document conversion. `HybridChunker` uses a tokenizer (for token counting). Set `DOCLING_TOKENIZER_MODEL` to a local path to avoid any network access.

**Design decision — docling off by default**: Docling adds a dependency and slightly longer ingestion time. The default `html` strategy is sufficient for most pages. Enable `docling` when you have pages with complex tables or code-heavy documentation.

#### Layer 2 — BM25 Hybrid Retrieval + RRF

**Problem with vector-only search**: Dense embeddings capture semantic similarity but can miss exact keyword matches. A query for "JENKINS_HOME variable" may rank a page about "setting environment variables" lower than a page about "Jenkins overview" because the embedding of the longer phrase is more diffuse.

**BM25 solution**: BM25 (Best Match 25) is a classical keyword ranking function that scores documents by term frequency and inverse document frequency. It excels at exact keyword recall.

**Reciprocal Rank Fusion**: Rather than normalising or weighting raw scores from two different distributions (cosine similarity vs. BM25 scores), RRF works purely on *rank positions*. Score = Σ 1/(k + rank), where k=60 dampens the impact of top-1 vs top-2 rank differences. This makes the merge robust without tuning.

**BM25 operates at page level**: Both vector and BM25 searches group their chunk hits by `page_id` to produce ranked page lists. RRF merges those page-level lists. This is consistent with the RAG-as-index design: chunks are search indices, not answer sources.

**Air-gapped**: `rank_bm25` is pure Python — no network dependency, no ML models.

**Design decision — BM25 off by default**: BM25 requires building and persisting a separate index file (`data/bm25.pkl`). Existing deployments are unaffected. Enable it by setting `ENABLE_BM25=true` and re-running the ingest script.

#### Layer 3 — Cross-Encoder Reranking

**Problem with bi-encoder retrieval**: Vector embeddings and BM25 both score documents independently of each other. A cross-encoder sees the (query, document) pair together and can capture fine-grained semantic alignment that bi-encoders miss.

**Two provider options**:

| Provider | How it runs | Good for |
|---|---|---|
| `openai_compatible` | POST to internal `/v1/rerank` endpoint via httpx | Company-hosted mxbai-rerank-large-v1, Infinity, vLLM |
| `local` | sentence-transformers CrossEncoder in-process | Fully air-gapped; model in `models/` directory |

**Two-stage retrieval**: The reranker is expensive (O(n × query_length) cross-attention). The pipeline first collects `RERANKER_CANDIDATE_K` candidate pages cheaply via vector+BM25, then passes their best-matching chunk texts to the reranker. Only `RERANKER_TOP_N` pages are returned after reranking.

**Design decision — reranker text = best chunk, not full page**: The reranker scores (query, document) pairs where each document is the top-scoring chunk for that page. This is significantly faster than reranking against full pages, and the best chunk is the most relevant snippet the page has to offer anyway. The full page is still what gets fetched and sent to the answer agent — the reranker only picks *which* page.

**Design decision — failures are non-fatal**: If the reranker endpoint is unreachable, the pipeline logs a warning and falls back to vector/RRF ordering. The answer agent still gets useful pages — reranking is a quality improvement, not a requirement.

**Air-gapped**:
- `openai_compatible`: only needs your internal gateway reachable (not the internet)
- `local`: set `TRANSFORMERS_OFFLINE=1` and `RERANKER_LOCAL_MODEL=models/<model-dir>`. Uses `sentence-transformers` CrossEncoder with a locally downloaded model.

### Full Stack Recommendation

For the highest accuracy retrieval in an air-gapped environment:

```
CHUNKING_STRATEGY=docling
ENABLE_BM25=true
RERANKER_PROVIDER=openai_compatible   # or: local
RERANKER_BASE_URL=https://your-gateway.internal
RERANKER_MODEL=mxbai-rerank-large-v1
RERANKER_CANDIDATE_K=30
RERANKER_TOP_N=5
```

Incremental adoption — each layer is independently useful:
- **BM25 alone** improves recall with no latency cost at query time
- **Reranker alone** on vector results already gives a meaningful precision boost
- **All three** gives the highest quality at the cost of ingestion time (docling) and query latency (reranker API call)

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
