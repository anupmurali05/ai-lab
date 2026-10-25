# AI Engineer Upskilling Plan — Days 1–20 (with 21–40 preview)

## Part 0 — How to use this

**Time budget:** 60–90 minutes per day. On a bad day, do the "20-minute version" noted on each day — read the concept and skip the build. Don't skip more than two builds in a row; reading about async retries and writing async retries are different skills.

**One repo for everything.** Create `ai-lab/` on Day 1 and keep adding to it. By Day 20 you'll have a working RAG + agent system with evals that you built yourself, which is worth more than notes.

**One corpus for everything.** Pick a public document set on Day 9 and reuse it for the rest of the plan (arXiv papers, a company's public docs, Wikipedia dumps, SEC filings). Do not put internal/PayPal documents into external APIs or hosted tools while learning — use the public corpus for the lab, and apply the patterns to internal systems inside PayPal's approved stack.

**Keep a log.** One file, `LOG.md`, three lines a day: what I learned, what surprised me, what I still don't understand. The third line is the valuable one — it becomes your next month's plan.

**Library versions move fast.** Everything named below is stable and widely used, but check current docs before pinning versions.

### Quick self-diagnostic — skip days you already own
Answer honestly. If you can answer a question cleanly in 60 seconds, skim that day instead of doing it.

| # | Question | Day |
|---|---|---|
| 1 | When would you use `Protocol` instead of an abstract base class? | 1 |
| 2 | What's the difference between `model_validate` and `model_validate_json`, and why does it matter? | 2 |
| 3 | What does `asyncio.Semaphore(8)` do to 200 concurrent LLM calls, and what breaks without it? | 3 |
| 4 | How do you write a deterministic unit test for a function that calls an LLM? | 4 |
| 5 | Why does answer quality often drop when you stuff more context in? | 5 |
| 6 | When do few-shot examples make output worse? | 6 |
| 7 | Why does asking for 15 fields in one extraction call degrade quality, and what's the fix? | 7 |
| 8 | What must you log on every LLM call to debug an incident three weeks later? | 8 |
| 9 | Why does query/document asymmetry matter in embedding search? | 9 |
| 10 | How do you index a source document that gets edited every day? | 10 |
| 11 | Name two query types where BM25 beats dense retrieval. | 11 |
| 12 | Why retrieve 50 and rerank to 5, rather than retrieve 5? | 12 |
| 13 | Answer quality dropped. How do you tell if it's retrieval or generation? | 13 |
| 14 | What's the difference between a chain and an agent, in one sentence? | 14 |
| 15 | What does a checkpoint buy you in a chat product? | 15 |
| 16 | What must never be summarized away when compacting a conversation? | 16 |
| 17 | Your multi-agent version is worse than the single agent. Three likeliest reasons? | 17 |
| 18 | Why is a 1–5 quality score from an LLM judge usually useless? | 18 |
| 19 | How do you catch a prompt change that raises average score but breaks one segment? | 19 |

---

## Part 1 — The topic map (everything an AI engineer needs)

This is the full territory. Days 1–20 cover Blocks 1–5. Days 21–40 cover 6–10.

### 1. Python engineering for AI
Typing and protocols · Pydantic v2 as the schema layer · asyncio (tasks, gather, TaskGroup, cancellation, timeouts) · concurrency vs parallelism, GIL, threads vs processes · rate limiting and backoff · resumable batch jobs · project tooling (uv, ruff, pytest) · configuration and secrets · structured logging · pytest, fixtures, mocking nondeterministic calls

### 2. LLM fundamentals
Tokenization and context windows · sampling parameters · cost and latency modeling (TTFT vs throughput) · prompt caching · prompt engineering as a versioned artifact · structured output, JSON schema, constrained decoding · function/tool calling · streaming · resilience patterns around the model call

### 3. Retrieval (RAG)
Embedding models and their failure modes · chunking strategies (recursive, semantic, heading-bounded, late, contextual) · parsing PDFs/PPTXs/tables · incremental ingestion, dedup, deletes, versioning · vector indexes (HNSW/IVF parameters and the recall/latency curve) · lexical search and hybrid fusion (RRF) · metadata filtering · reranking (cross-encoder, listwise LLM) · query rewriting, multi-query, HyDE, decomposition · retrieval evaluation

### 4. Agents
The tool use loop · tool schema design · stopping criteria and error feedback · ReAct vs plan-and-execute · orchestration frameworks and graph state · persistence, interrupts, human-in-the-loop · context engineering (budgeting, compaction, scratchpads, subagents) · memory · multi-agent patterns and when they hurt · routing and model cascades · MCP

### 5. Evaluation and observability
Error analysis and failure taxonomies · golden datasets · binary rubrics vs scores · LLM-as-judge design and validation · retrieval metrics (recall@k, MRR, nDCG) · generation metrics (faithfulness, groundedness, citation accuracy) · tracing and spans · cost/latency dashboards · online eval and user feedback · regression suites in CI · canary and A/B

### 6. Production LLMOps
Exact and semantic caching · model routing and cascades · latency engineering · load testing and capacity planning · fallbacks, circuit breakers, graceful degradation · prompt/version pinning and rollout · cost attribution and budgets · SLOs for probabilistic systems

### 7. Security and governance
Direct and indirect prompt injection · tool sandboxing and least privilege · permission-aware retrieval and tenant isolation · PII detection and redaction · output filtering · audit logging · red teaming · model governance, model cards, regulatory context (EU AI Act, internal review processes)

### 8. Model adaptation
Prompt vs RAG vs fine-tune decision framework · dataset curation · SFT with LoRA/QLoRA · preference tuning (DPO) at a conceptual level · embedding model fine-tuning and reranker training (highest ROI for enterprise RAG) · distillation from a large model to a small one · evaluating a fine-tune honestly

### 9. Inference and internals
Transformer architecture, attention, positional encoding · KV-cache and why it dominates memory · quantization · serving stacks (vLLM, TGI) · continuous batching · throughput vs latency tradeoffs · GPU memory math · self-host vs API decision

### 10. Data engineering and classic ML literacy
SQL and warehouse patterns (BigQuery) · batch vs streaming ingestion · orchestration (Airflow/Dagster) · data contracts and schema evolution · N2L/SQL done properly (schema linking, validation, guardrails) · classification metrics, calibration, drift · multimodal (vision models, document AI, OCR)

---

## Part 2 — Days 1–20

### Block A — Python that AI engineers actually need (Days 1–4)

#### Day 1 — Modern Python project hygiene and typing
**Why it matters:** Most AI code sucks because it's a pile of scripts. The difference between a demo and a system is boring: a package layout, typed interfaces, config that isn't hardcoded, and logs you can grep.

**Learn (~30 min):**
* Type hints beyond the basics: `Protocol`, `TypeVar`, `Literal`, `Generic`, `X | None`, `Annotated`
* `uv` for environments and dependency management; `ruff` for lint/format; `pyright` or `mypy` for type checking
* `__init__.py`: why it prevents import shadowing
* `pydantic-settings` for config from env vars
* Structured logging: JSON logs with a correlation/request id

**Build (~45 min):** Scaffold `ai-lab/`:
```text
ai-lab/
├── pyproject.toml         # uv, ruff, pyright config
└── src/ai_lab/
    ├── __init__.py
    ├── config.py          # Settings(BaseSettings) — API keys, model names, limits
    ├── logging.py         # JSON logger, request_id context var
    └── llm/
        ├── base.py        # class LLMClient(Protocol): async def complete(...) -> str
        └── text.py
```
Define `LLMClient` as a `Protocol` with `complete()` and `complete_structured()`. Write one real implementation and one fake. You'll use both all month.

**20-minute version:** Just the `Protocol` + `pydantic-settings` config file.

**Check yourself:** Why is a `Protocol` better than an ABC for an LLM client abstraction you want to swap in tests?

---

#### Day 2 — Pydantic v2 as your schema layer
**Why it matters:** Pydantic is the contract between the LLM's imagination and your system. Every structured extraction, tool definition, and API boundary goes through it.

**Learn (~35 min):**
* `BaseModel`, `Field` (description, constraints, defaults) — descriptions matter because they end up in the JSON schema the model reads
* `field_validator` vs `model_validator` (mode before vs after)
* Discriminated unions with `Field(discriminator=...)` — the clean way to model "one of these tool calls"
* `Literal` and `Enum` for constrained choices
* `model_dump_json()` — inspect what the LLM actually sees
* `TypeAdapter` for validating lists and non-model types
* `ConfigDict(extra='forbid')`
* Strict vs lax coercion; aliases for external field names

**Build (~45 min):** Write a document summary schema: `title: str`, `entities: list[Entity]`, `risk_level: Literal["low", "medium", "high"]`, `confidence: float = Field(ge=0, le=1)`, `citations: list[str]`. Then write a validate-and-repair loop: call the LLM, parse, and on validation error re-prompt once with the serialized error messages appended. Log the repair rate.

**20-minute version:** Build the schema, print `model.json_schema()`, and study what descriptions and constraints made it through.

**Check yourself:** What's the difference between `model_validate` and `model_validate_json`, and why is the second faster?

---

#### Day 3 — asyncio for LLM workloads
**Why it matters:** LLM calls are almost pure I/O wait. Sequential loops are the single most common reason AI pipelines are 20× slower than they need to be.

**Learn (~35 min):**
* Coroutine vs Task: nothing runs until awaited or scheduled
* `asyncio.gather` vs `as_completed` vs `TaskGroup` (and how exceptions propagate in each)
* `asyncio.timeout` / `wait_for`, and cancellation semantics
* `asyncio.Semaphore` for concurrency limits — your defense against 429s
* `https.AsyncClient` with connection pooling and per-request timeouts
* `asyncio.to_thread` for the sync library you can't avoid
* The trap: one blocking call (a sync SDK, `time.sleep`, a big CPU loop) freezes the entire loop

**Build (~45 min):** Run 100 summarization calls over your corpus:
* `Semaphore(8)` to cap concurrency
* Per-call timeout
* Exponential backoff with jitter on 429/too_many_requests
* Collect results with `as_completed` so you can show progress
* Aggregate total wall clock, p50/p95 latency, total tokens, total cost. Then run the same job sequentially and record the speedup (ex: 16s vs ...).

**20-minute version:** Take an existing sequential loop from your work and rewrite it with `gather` + `Semaphore`.

**Check yourself:** What actually goes wrong if you await `client.complete()` inside a `for` loop instead of gathering?

---

#### Day 4 — Parallelism, batch jobs, and testing nondeterministic code
**Why it matters:** Ingestion pipelines fail at hour three. Whether that costs you five minutes or a whole day is decided by design choices you make on day one.

**Learn (~35 min):**
* Decision table: `asyncio` for network I/O, `threads` for blocking I/O in sync libraries, `processes` for CPU work (PDF parsing, tokenization, embedding math). The GIL in Python 3.12 vs 3.13.
* `concurrent.futures.ProcessPoolExecutor`, and why you can't pass unpicklable objects to it
* Resumability: `JSONL` output keyed by stable content hash; skip if done on restart; atomic writes
* Idempotency and at-least-once semantics
* `pytest` fixtures, parametrize, `pytest-asyncio`, `monkeypatch`
* Testing LLM code: three tiers: (1) fake client returning canned responses, (2) recorded cassettes replayed offline, (3) a small "does it still work" eval suite that runs nightly, not on every commit

**Build (~45 min):** Turn Day 3's script into a resumable batch job: reads a manifest, hashes each input, writes `results.jsonl`, skips completed ids on restart, handles SIGINT cleanly. Kill it mid-run and restart to prove it works. Add `tests/` with a fake `LLMClient` and assertions on the retry, timeout, and repair loop logic.

**20-minute version:** Add hash-based skip if done to any existing pipeline you own.

**Check yourself:** How do you write a test that fails when your prompt breaks, but doesn't fail because the model phrased something differently?

---

### Block B — LLM core (Days 5–8)

#### Day 5 — Tokens, context, cost, latency
**Why it matters:** These four numbers determine whether your design is viable. Engineers who can estimate them get their architectures approved.

**Learn (~35 min):**
* Tokenization: subwords; why code and non-English text cost more; why token counts differ per model family
* Context window mechanics: it holds system + history + retrieved context + tools + output; all compete
* "Lost in the middle" — retrieval position affects whether the model uses a fact
* Pricing: input vs output token asymmetry; prompt caching and how it changes the math for long stable system prompts
* Latency: TTFT vs token/sec; output length dominates total latency; streaming changes perceived latency, not real latency
* Determinism: even `temperature=0` isn't reproducible across infra

**Build (~40 min):** Write `costs.py`: given model, input tokens, output tokens, and requests/day → monthly cost, with and without prompt caching. Then measure same prompt at 1k / 8k / 32k / 128k input tokens; measure latency, TTFT, and total latency. Plot it (this graph will settle arguments for you later).

**20-minute version:** Compute the monthly cost of one system you already run at PayPal, broken down by input/output/cached tokens.

**Check yourself:** Two designs — one 30k token prompt per request, or three 5k token requests. Which is cheaper, which is faster, and what changes the answer?

---

#### Day 6 — Prompt engineering as engineering
**Why it matters:** Prompts are the highest-leverage and least version-controlled part of most AI systems. Treat them like code: versioned, tested, diffed.

**Learn (~30 min):**
* System vs user roles and the instruction hierarchy
* Structure: delimiters (XML tags work well), explicit output contracts, ordering (instructions before long context, restate key instruction after)
* Few-shot: how to choose examples, why 3 good ones beat 20 mediocre ones, when examples cause over-copying
* Reasoning: when to ask for step-by-step, when it wastes tokens, self-consistency
* Decomposition: one prompt per job beats one prompt for five jobs
* Negative instructions ("do not invent") are weaker than positive ones ("only use facts present in <context>")
* Prompt templates, versioning, and keeping prompts out of f-strings scattered through the codebase

**Build (~50 min):** Take one real task you already do — the feature/entity extraction, or summarization. Build a 20-example test set with expected outputs. Write prompt v1 (what you have today) through v4, changing one thing at a time. Score each. Record which change gave the biggest lift.

**20-minute version:** Move every prompt in one of your projects into a `prompts/` module with version constants.

**Check yourself:** Name a situation where adding few-shot examples actually makes output worse.

---

#### Day 7 — Structured output, JSON schema, constrained decoding
* Long documents: chunked extraction + per-chunk results + merge/dedup/reconcile
* Repair loops, and when to give up and route to a human
* Tool calling mechanics: tool definitions, the `tool_use`/`tool_result` cycle, parallel tool calls
* Multi-step: the model decides which tool; you keep control of what it can do

**Build (~50 min):** Extraction pipeline over a 50-page PDF:
1. Parse and chunk
2. Extract structured entities per chunk (Pydantic-validated)
3. Merge and dedup across chunks
4. Report: schema failure rate, repair success rate, duplicate rate, per-page cost. Then split one big 15-field schema into three focused calls and compare quality and cost.

**20-minute version:** Take your largest extraction schema and split it into two calls. Measure the difference.

**Check yourself:** Why does a 15-field schema in one call degrade quality, and what are two fixes?

---

#### Day 8 — The LLM client layer (resilience)
**Why it matters:** Everything downstream sits on this. Build it once, well, and you stop rewriting retry logic in every project.

**Learn (~30 min):**
* Timeouts at three levels: connect, read, total
* Retry policy: what's retryable (429, 5xx, timeouts) and what isn't (400, content filter); backoff with jitter; retry budgets
* Fallback chains: primary model → secondary model → cached/degraded response
* Circuit breakers: stop hammering a dead dependency
* Streaming: SSE handling, partial JSON parsing, cancellation mid-stream
* Guardrails: max token budget per request and per user, hard-cost ceilings
* Telemetry per call: model, prompt version, prompt hash, input/output tokens, latency, TTFT, retry count, finish reason, cost, request id, client id

**Build (~45 min):** Finalize `ai-lab/src/ai_lab/llm/client.py` — the class every later day imports:
* `async`, with sync convenience wrapper
* `retry` + backoff + `jitter`, configurable
* `fallback` model chain
* `streaming` support
* `complete_structured(schema)` with the repair loop from Day 2
* emits a telemetry record for every call
* a pluggable cache interface (exact-match now, semantic later)

**20-minute version:** Add telemetry record to whatever LLM wrapper you use at work today.

**Check yourself:** Production incident, three weeks ago, one user got a garbage answer. What fields in your logs let you reconstruct exactly what happened?

---

### Block C — Retrieval (Days 9–13)

#### Day 9 — Embeddings
**Why it matters:** You've built ingestion pipelines. This is where you go from "it works" to knowing *why* recall is 70% instead of 90%.

**Learn (~35 min):**
* What embeddings encode, and what they don't (negation, numbers, precise identifiers — all weak)
* Dimensionality and Matryoshka embeddings (truncatable dimensions for cheap storage)
* Domain mismatch: general purpose models on jargon-heavy enterprise text
* The operational trap: changing embedding models means re-embedding everything; plan for it (model id in metadata, dual-write migration)
* Cheap wins before fine-tuning: better chunk text, adding titles/headings into the embedded text

**Build (~50 min):** Pick your corpus (~2,000 docs). Write 30 real queries by hand with the doc you'd expect back. Embed the corpus with two different models. Compare recall@5 for each. Then re-embed with the heading path prepended to each chunk and compare again — this one change often beats a model upgrade.

**20-minute version:** Write the 30 query-golden doc pairs. You'll use it for the next four days.

**Check yourself:** Why does query/document asymmetry matter, and what does a model's recommended prefix actually do?

---

#### Day 10 — Chunking and ingestion design
**Why it matters:** You're already built this once. Today is about the strategies you didn't pick, and the operational problems that show up in month six.

**Learn (~35 min):**
* Strategies: fixed-size, recursive, sentence/semantic, **heading-bounded** (you use this one), parent-document (embed small, return large), late chunking, **contextual retrieval** (prepend an LLM-written one-liner to context to each chunk before embedding — one of the biggest reported recall gains)
* Overlap: what it fixes and what it wastes
* Tables: never split a table; consider serializing rows as sentences or storing a table summary alongside
* Code and URLs: preserve them intact
* Parsing pitfalls: multi-column PDFs, scanned pages, PPTX text boxes out of reading order, footnotes bleeding into body
* Operations: content-hash dedup, incremental upsert, **tombstoning deletes** (soft-delete + tombstone), document versioning, freshness metadata, backfill strategy
* Metadata design: what you filter on later must be indexed now

**Build (~50 min):** Rebuild a mini ingestion pipeline with three interchangeable chunkers (recursive / heading-bounded / contextual). Same corpus, same embedded queries from Day 9. Report recall@5 for each. Add content-hash dedup and an upsert path: then re-ingest a modified document and verify no duplicates.

**20-minute version:** Implement contextual retrieval on 200 chunks and measure recall against your Day 9 baseline.

**Check yourself:** A source document is edited daily and its chunk boundaries shift. What breaks, and how do you design around it?

---

#### Day 11 — Vector, lexical, and hybrid search
**Why it matters:** Pure vector search fails on exactly the queries enterprise users ask most — IDs, error codes, product names, exact phrases.

**Learn (~35 min):**
* ANN indexes: HNSW (`M`, `efConstruction`, `efSearch`) and IVF (`nlist`, `nprobe`): the recall/latency/memory triangle; flat index as the ground-truth baseline
* BM25 term frequency, inverse document frequency, length normalization — why it nails rare/exact terms
* Hybrid fusion: Reciprocal Rank Fusion (simple, robust, no score normalization needed) vs weighted score blending (needs normalization, needs tuning)
* Metadata filtering: pre-filter vs post-filter, and how aggressive filters wreck ANN recall
* Milvus specifics worth knowing: cold partitions vs collections, index types, consistency levels, and why deep offset pagination is a trap (use keyset iterator pattern)

**Build (~50 min):** Build a hybrid retriever: dense (Milvus) + BM25 (rank_bm25, locally or OpenSearch), fused with RRF. Evaluate against Day 9's golden set. Then sweep `efSearch` across four values and plot recall vs p95 latency. Find the knee.

**20-minute version:** Add BM25 + RRF to an existing dense-only retriever and measure the delta.

**Check yourself:** Give two concrete query types where BM25 beats embeddings, and explain why hybrid usually beats both.

---

#### Day 12 — Reranking and query transformation
**Why it matters:** Reranking is typically the single largest quality jump per line of code in a RAG system. Query transformation is what makes multi-turn chat retrieval work at all.

**Learn (~30 min):**
* Bi-encoder vs cross-encoder: why the cross-encoder is far more accurate and far too slow to run over the whole corpus — hence retrieve-wide then rerank-narrow
* Listwise LLM reranking as an alternative; cost/latency tradeoff
* Query rewriting: resolving pronouns and ellipsis in follow-up turns ("what about the other one?")
* Multi-query expansion, HyDE (embed a hypothetical answer, not the question)
* Query decomposition for multi-hop questions
* Routing: deciding when not to retrieve at all
* Recency and authority boosting in the final ranking

**Build (~50 min):** Add a reranking stage: retrieve top-50 hybrid → cross-encoder rerank → top-5. Measure NDCG@5 and recall@5 before and after; plus added p95 latency. Then add a query rewrite step that takes conversation history + latest turn → standalone query. Test it on 10 multi-turn conversations.

**20-minute version:** Drop a cross-encoder reranker into an existing pipeline and measure. This is the highest ROI 20 minutes in the plan.

**Check yourself:** Why retrieve 50 and rerank to 5, instead of just retrieving 5? What does that cost you?

---

#### Day 13 — RAG evaluation
**Why it matters:** Without this, every change you make after today is a guess. This is what separates people who ship RAG from people who tune RAG forever.

**Learn (~30 min):**
* Golden dataset construction: question, ideal answer, **gold_chunk_ids**. Generating candidate questions with an LLM from your corpus, then human-filtering, is a legitimate shortcut
* Retrieval metrics: recall@k (usually the one that matters), precision@k, MRR, NDCG
* Generation metrics: faithfulness/groundedness (is every claim supported by retrieved context?), answer relevance, citation accuracy, completeness
* Frameworks: RAGAS and similar — useful, but understand the metric definitions before trusting the numbers
* Refusal when the answer wasn't present
* Bad question / unanswerable from corpus

**Build (~55 min):** Build a 50-question eval set for your corpus with gold chunk ids. Write `evaluate.py` that outputs a scorecard: recall@5, nDCG@5, faithfulness, and a failure-type breakdown. Run it against every pipeline variant from Days 10–12 and produce one comparison table.

**20-minute version:** Build 20 questions with gold chunk ids and measure recall@5 on your current production retriever. The number will probably surprise you.

**Check yourself:** Quality dropped after a change. What's your diagnostic sequence to determine whether it's retrieval or generation?

---

### Block D — Agents (Days 14–17)

#### Day 14 — The agent loop, from scratch, no framework
**Why it matters:** You've built multi-agent systems on a framework. Writing the loop yourself removes the mystery and makes framework debugging trivial afterward.

**Learn (~25 min):**
* The loop: prompt → model returns tool_use → you execute → return tool_result → repeat until text response or limit
* Tool schema design: names matter, descriptions matter, and per-argument descriptions matter. Ambiguous tools cause most agent failures
* Stopping criteria: max iterations, max tokens, max cost, max wall clock, no-progress detection
* Error handling: return errors to the model as tool results so it can recover, but cap the recovery attempts
* Parallel tool calls
* ReAct vs plan-and-execute vs reflection loops
* Why agents fail: context bloat, overstepping tool responsibilities, no ground-truth to check against, silent partial failures

**Build (~60 min):** Write the loop yourself in ~150 lines. Three tools: search_corpus (your Day 12 retriever), run_sql (SQLite with a small table), calculate. Log the full trajectory — every message, tool call, and result — to a JSONL file. Run 10 questions and read all 10 trajectories end-to-end. Note where it wasted steps.

**20-minute version:** Write the loop with one tool and read one full trajectory.

**Check yourself:** Chain vs agent, in one sentence. And what's the cheapest reliable way to stop a runaway loop?

---

#### Day 15 — Orchestration frameworks (LangGraph)
**Why it matters:** You've used framework orchestration. Today is about the parts most people never touch: persistence, interrupts, and streaming — which are exactly what production chat products need.

**Learn (~35 min):**
* Graph state: typed state, reducers, why state as a graph beats callback spaghetti
* Nodes, edges, conditional edges, cycles
* Checkpoints: persist state per thread → resume conversation, survive restarts, time-travel debugging
* Interrupts: pause before a tool executes; wait for human approval; resume
* Streaming: node-level events vs token streaming, and what you surface to the UI
* Subgraphs and composition
* When not to use a framework: if your flow is a straight line with two branches, plain Python is more debuggable

**Build (~50 min):** Re-implement Day 14's agent in LangGraph:
* State with messages and a custom reducer
* Interrupt on the `run_sql` node if the query contains `DROP`, `DELETE`, or `UPDATE`
* Persist to SQLite checkpointer
* Resume an interrupted run with approval or modification
* Stream tokens from the generating node to stdout

**Check yourself:** What does a checkpointer buy you in a chat product, and why is it otherwise missing?

---

#### Day 16 — Context engineering and memory
**Why it matters:** This is the current frontier skill. Most agent quality problems in long conversations are context problems, not model problems.

**Learn (~35 min):**
* Context budgeting: allocate the window explicitly across system prompt, tools, memory, retrieved docs, history, output
* Degradation with long contexts — more context is not free, and past a point it's negative
* Compaction: summarize old turns, but pin the things that must never be lost (user's stated goal, decisions made, IDs, constraints, corrections)
* Tool result truncation: return a summary + handle to fetch detail on demand
* Scratchpads / external state: write intermediate results to files or state, not into the conversation
* Subagents for context isolation: a subagent does a noisy 30-step search and returns three clean sentences to the parent
* Memory: short-term (this thread) vs long-term (across threads); what's worth persisting; retrieval vs memory; staleness and contradiction handling

**Build (~50 min):** Add to your agent: (1) a token budget enforcer that reports allocation per turn, (2) compaction that triggers at 60% window usage and preserves a pinned facts block, (3) a simple long-term memory store keyed by user with explicit write and retrieve steps. Run a 30-turn conversation and plot tokens per turn before and after compaction. Verify the agent can still recall a constraint stated in turn 2 at turn 28.

**20-minute version:** Implement tool result truncation with a "fetch full result" tool. Measure the token savings.

**Check yourself:** What must never be summarized away, and how do you enforce that mechanically rather than hoping?

---

#### Day 17 — Multi-agent patterns and routing
**Why it matters:** You've built multi-agent systems. The skill now is knowing when a single agent with good tools beats a committee — and being able to prove it with numbers.

**Learn (~30 min):**
* Supervisor/worker, sequential handoff, parallel fan-out/fan-in, debate/critic loops
* Shared state vs message passing, and the information loss at every handoff
* When multi-agent hurts: coordination overhead, error compounding across steps, cost multiplication, harder debugging, conflicting sub-decisions. Read hearsay: parallel research + multi-agent wins. Tightly coupled sequential reasoning → usually loses.
* Routing layers: cheap classifier first, deterministic rules where possible, LLM router only when needed
* Model cascades: small model attempts; escalate on low confidence or validation failure
* Deterministic fallbacks and HITL approval gates for high-stakes actions

**Build (~50 min):** Build a supervisor routing to three specialists (retrieval, SQL, summarizer), with a deterministic keyword classifier fast path in front of it. Run Day 13's eval set against both the multi-agent version and the Day 15 single agent version. Compare accuracy, latency, cost, and step count. Write down which one wins and why — honestly, even if it's the simple one.

**20-minute version:** Add a cheap model router in front of an existing agent and measure the cost reduction.

**Check yourself:** Your multi-agent version scores worse than the single agent. What are the three likeliest causes, in order?

---

### Block E — Evaluation and observability (Days 18–20)

#### Day 18 — Evals that actually correlate with quality
**Why it matters:** This is the highest-leverage skill in AI engineering and the one most engineers skip. People who can build trustworthy evals become the ones who decide what ships.

**Learn (~30 min):**
* Error analysis comes first. Before writing a single metric, read ~50 real traces and log what went wrong in your own taxonomy. Chart the top three: your top three clusters define your metrics. Metrics invented before error analysis measure the wrong things.
* Binary rubrics beat 1–5 scales: "Does the answer cite a source for every factual claim? yes/no" is reproducible; "rate quality 1–5" is not
* LLM-as-judge design: one criterion per judge, give the rubric and examples, ask for reasoning before verdict, control position bias in pairwise comparisons
* Validating the judge: label 50 examples yourself, measure judge human agreement. Below ~80%, fix the judge before trusting any of its numbers.
* Pairwise comparison when absolute scoring is unstable
* Dataset hygiene: hold out a test set, don't tune prompts on your test set, refresh holdout regularly
* Assertions and unit-test style checks (format valid, no PII, required fields present) — cheap, deterministic, catch a surprising share of failures

**Build (~55 min):** Export 50 traces from anything you've built. Read every one. Tag failures in a spreadsheet, cluster into a taxonomy, and count. Then write two LLM judges for the top two failure modes, hand-label 30 examples, and compute agreement. Iterate the judge prompt until agreement is acceptable.

**20-minute version:** Read 20 traces from a real system at work and write the failure taxonomy. Do not skip this one — it changes how you see your own systems.

**Check yourself:** Why is an LLM-produced 1–5 quality score usually useless, and what do you replace it with?

---

#### Day 19 — Observability and CI for LLM systems
**Why it matters:** Offline evals catch regressions before release. Tracing catches everything else. You need both, plumbed together.

**Learn (~30 min):**
* Tracing model: trace → spans → attributes; a span for LLM call, retrieval, and tool execution
* OpenTelemetry as the substrate; Langfuse / LangSmith / Phoenix as the LLM-aware layer on top
* What to capture: inputs, outputs, prompt version, model, latency, cost, tokens, retrieved hits and scores, tool results, used id, session id
* PII redact at capture time, not at query time; retention policy
* Online eval: cheap automated judges on a sampled slice of production traffic
* Feedback loops: explicit (thumbs up/down) and implicit (retry, copy, abandon, follow-up rephrasing — often more honest than thumbs)
* Regression CI: eval suite on PRs touching prompts or retrieval; fail on regression beyond a threshold; keep runtime and cost bounded by sampling
* Rollout: prompt versions pinned in config, canary on a traffic slice, A/B with a pre-declared metric
* Segment your metrics. Average scores hide the segments you just broke.

**Build (~55 min):** Instrument the agent with tracing end-to-end. Build a small dashboard or query showing: cost per request, p50/p95 latency, and failure rate by type. Then add a CI job that runs a 30-question eval subset on PRs and fails on >5% regression. Deliberately break a prompt and watch it catch it.

**20-minute version:** Add basic timing and token logging to one pipeline at work.

**Check yourself:** A prompt change improves the average score but breaks legal domains completely. How do you prevent this?

---

#### Day 20 — Consolidation, architecture, and defending your choices
**Why it matters:** Being able to explain and defend a design is what turns twenty days of learning into a change in how you're seen at work.

**Do (~90 min, worth blocking properly):**
1. **Diagram the system you built.** Ingestion → index → hybrid retrieval + rerank + agent + eval + observability. Every arrow labeled with what flows and what it costs.
2. **Write a one-page design doc** with chosen approach, two rejected alternatives and why, cost model (at 100 current traffic), p95 latency budget by component, top five failure modes and mitigations, and what you'd measure to know it's degrading.
3. **Audit your eval suite.** List the three weakest parts of the POC assistant / ingestion pipeline / agent stack you own at PayPal. For each: the specific gap, the fix, and the expected measurable improvement. Be concrete — "no retrieval eval set, so we can't tell if chunking changes help" is a finding; "retrieval could be better" is not.
4. **Re-give a 20-minute brown bag** for your team on one thing from these 20 days. Teaching it is the test of whether you learned it.
5. **Re-take the self-diagnostic** at the top. Note what's still shaky — that's your Day 21+ priority list.

---

## Part 3 — Days 21–40 (preview)

Detailed day-by-day comes after you finish Day 20 shaped by what you log as your weak on. The shape:

### Days 21–24 — Production LLMOps
Exact and semantic caching; load routing and cascades; latency engineering; load testing and capacity planning; fallbacks, circuit breakers, graceful degradation; prompt/version pinning and rollout; cost attribution and budgets; SLOs for probabilistic systems

### Days 27–27 — Security and governance
Direct and indirect prompt injection; tool sandboxing and least privilege; permission-aware retrieval and tenant isolation; PII detection and redaction; output filtering; audit logging; red teaming; model governance, model cards, regulatory context (EU AI Act, internal review processes)

### Days 28–31 — Model adaptation
Prompt vs RAG vs fine-tune decision framework; dataset curation; SFT with LoRA/QLoRA; preference tuning (DPO); embedding model fine-tuning and reranker training (highest ROI for enterprise RAG); distillation from a large model to a small one; evaluating a fine-tune honestly against a strong prompted baseline

### Days 32–34 — Inference and internals
Transformer architecture, attention, positional encoding; KV-cache and why it dominates memory; quantization serving stacks (vLLM, TGI); continuous batching; throughput vs latency tradeoffs; GPU memory math (INT4/INT8/Q4/FP16) and quality tradeoffs; self-host vs API decision with real numbers

### Days 35–37 — Data engineering for AI
Warehouse patterns and query optimization (BigQuery); batch vs streaming ingestion; CDC for keeping an index fresh; orchestration with Airflow or Dagster; retries, backfills, SLAs; idempotent tasks; data contracts and schema evolution for pipelines; LLMs + SQL: deep dive (generation, validation, guardrails); table selection for large schemas; few-shot SQL; generated query validation; read-only guardrails; dataset drift, multimodal (vision models, document AI, OCR)

### Days 38–40 — Breadth and capstone
Multimodal vision models for document understanding, when VLM parsing beats OCR, layout-aware extraction; classic ML literacy an AI engineer needs (precision/recall tradeoffs, calibration, class imbalance, drift detection, when a fine-tuned classifier is simply better and cheaper than an LLM call); Capstone: one end-to-end system built solo — ingestion, hybrid retrieval with reranking, an agent with tools and HITL, full eval suite, tracing, CI, cost model, design doc. Present it.

---

## Reference shelf
*Worth having open throughout, rather than reading front to back:*

* **Anthropic docs** — tool use, structured output, prompt caching, context engineering, agent design (`docs.anthropic.com`)
* **Pydantic docs** — the validators and JSON schema sections specifically
* **Python `asyncio` docs** — the task and synchronization primitives pages
* **LangGraph docs** — persistence, interrupts, streaming
* **Milvus docs** — index types, consistency levels, partitioning
* **Chip Huyen, *AI Engineering*** — best single book on the discipline as a whole
* **Hamel Husain's writing on evals** — the most practical material available on error analysis and LLM-as-judge
* **Eugene Yan's writing** — RAG patterns, evals, system design
* **Jay Alammar, "The Illustrated Transformer"** — for day 32
* **Sebastian Raschka's writing** — for the fine-tuning and internals blocks
* **RAGAS docs** — read the metric definitions, not just the API

---

## One rule for the whole plan
Every day ends with something in the repo that didn't exist that morning. Reading about retrieval evaluation for an hour and reading about it for ten minutes then building a 20-question eval set are not the same activity, and only one of them changes what you can work on next month.
