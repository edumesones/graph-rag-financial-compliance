# Fintech Agentic RAG

A retrieval-augmented generation service over SEC EDGAR filings that routes each
question between a **knowledge graph** and a **vector index**, and can use both.
FastAPI, Neo4j, ChromaDB, PostgreSQL, Redis and Prometheus, deployed with Docker
Compose.

> **On numbers:** this README contains no accuracy, quality or latency figures,
> because no evaluation has been run against this system. The only measurements
> quoted are test results that were executed and are reproducible with the
> commands given below. See [Evaluation](#evaluation-not-performed).

---

## Why a graph alongside the vector index

Vector search retrieves passages that are *semantically near* a question. That is
the right tool for "what does this filing say about supply chain risk". It is the
wrong tool for a question whose answer is a **relationship**, because the entities
involved are rarely near each other in the text.

Take a concrete compliance question:

> *Which regulations has this company been cited under, and when?*

In a 10-K, the regulation is usually named in one section, the enforcement action
described in another, and the date sits in a third. No single chunk contains the
triple. Vector search returns the chunks that most resemble the question — often
the boilerplate "we are subject to extensive regulation" paragraph, which is the
nearest neighbour of the question and contains none of the answer. Raising `k`
adds more of the same. The failure is structural, not a tuning problem: similarity
to the *question* is not the retrieval criterion when the answer is a join across
the document.

Ingestion therefore extracts entities into Neo4j:

```
(:Company)-[:SUBJECT_TO]   ->(:Regulation)
(:Company)-[:HAD_VIOLATION]->(:Violation {description, date})
(:Company)-[:HAS_RISK]     ->(:Risk)
```

The question above becomes a traversal that returns the company, the violation and
its date together, regardless of how far apart they sat in the filing. Multi-hop
questions — *which other companies were cited under the same regulation* — are one
more hop, and have no vector formulation at all.

The graph is not a replacement. It only holds what the extraction step captured,
so anything not modelled as an entity is invisible to it, and narrative questions
are answered far better by the vector index. Hence the routing layer below: the
system decides per query, and for questions that need both it runs both and fuses
the results.

---

## Architecture

Five layers, orchestrated by `src/agentic_rag/orchestrator.py`:

| Layer | Module | Responsibility |
|---|---|---|
| 1. Indexing | `layer1_indexing.py` | Semantic chunking, BGE embeddings into ChromaDB, entities into Neo4j |
| 2. Routing | `layer2_routing.py` | Classify the query as `VECTOR`, `GRAPH` or `HYBRID` |
| 3. Query building | `layer3_query_builder.py` | Decompose into sub-queries; translate to Cypher where the route calls for it |
| 4. Retrieval | `layer4_retrieval.py` | Run vector and/or graph search, fuse with Reciprocal Rank Fusion |
| 5. Generation | `layer5_generation.py` | Synthesise an answer with citations back to source chunks |

**Routing** asks the LLM to classify the query and falls back to keyword rules when
no LLM is configured, so the service degrades rather than failing. Decisions are
cached in Redis, since the same question routes the same way every time.

**Retrieval** merges vector and graph hits with Reciprocal Rank Fusion, which needs
only the rank of each result and so can combine two systems whose scores are not
comparable — cosine similarity and graph traversal order have no shared scale.

Every layer's prompts, latencies and errors are written to PostgreSQL
(`prompt_logs`, `error_logs`, `query_metrics`) and counted in Prometheus, so a bad
answer can be traced to the layer that produced it.

### Stack

- **API**: FastAPI + Gunicorn/Uvicorn — `/analyze_compliance`, `/get_system_stats`,
  `/get_prompt_logs`, `/health_check`, `/metrics`
- **Vector store**: ChromaDB, `BAAI/bge-large-en-v1.5` embeddings (run locally)
- **Graph**: Neo4j 5.24
- **LLM**: HuggingFace Inference API or OpenAI, selected by configuration
- **Logging / cache**: PostgreSQL 16, Redis 7
- **Observability**: Prometheus + Grafana
- **Corpus**: SEC EDGAR 10-K filings via `sec-edgar-downloader`

---

## Running it

```bash
cp .env.example .env     # then fill in the values
docker compose up -d
curl localhost:8000/health_check
```

`.env` needs an LLM credential (`HUGGINGFACE_TOKEN` or `OPENAI_API_KEY`), passwords
for Neo4j/PostgreSQL/Grafana, and `SEC_USER_AGENT`. SEC EDGAR's fair-access policy
requires a User-Agent identifying the requester, and ingestion fails loudly without
one rather than being silently throttled:

```bash
SEC_USER_AGENT="Your Name your.email@example.com"
```

Ingest a company, then query it:

```bash
python scripts/ingest.py --company TSLA --limit 5

curl -X POST localhost:8000/analyze_compliance \
  -H 'Content-Type: application/json' \
  -d '{"company": "Tesla Inc", "query": "Which regulations is this company subject to?"}'
```

`data/` ships empty — the corpus is downloaded from EDGAR at ingestion time and is
not redistributed here.

---

## Tests

56 test functions across 6 files. **44 are verified passing** against live
PostgreSQL 16, Redis 7 and Neo4j 5.24.2; the remaining 12 were not run and are
marked as such.

| File | Tests | What it exercises | Status |
|---|---:|---|---|
| `test_neo4j.py` | 13 | Graph connectivity, node/relationship writes, Cypher queries, constraints | **13 pass** |
| `test_redis_caching.py` | 15 | Set/get, TTL expiry, hit-miss accounting, namespace isolation, concurrency, reconnect | **15 pass** |
| `test_postgres_logging.py` | 13 | Prompt/error/metric logging, JSONB round-trip, session tracking, concurrent writes, reconnect | **12 pass + 1 benchmark** |
| `test_utils.py` | 3 | Config loading, PostgreSQL and Redis managers | **3 pass** |
| `test_integration.py` | 11 | FastAPI endpoints end to end | **not run** — needs the LLM dependencies |
| `load_test.py` | 1 | Throughput under concurrent load | **not run** |

Reproduce the 43-test default run with the three databases up:

```bash
docker compose up -d neo4j redis postgresql
pytest tests/test_neo4j.py tests/test_redis_caching.py \
       tests/test_postgres_logging.py tests/test_utils.py
# 43 passed, 1 deselected
```

**What these tests are.** They are integration tests, not unit tests. There is
almost no mocking: they talk to real databases and assert that the wiring holds —
that a prompt written is a prompt read back, that a TTL expires, that the pool
survives a disconnect. Exactly one test (`test_config_loading`) runs without
infrastructure.

**What they do not cover.** Nothing here measures retrieval quality. No test
asserts that the router picks the right strategy, that the graph extraction is
correct, or that an answer is faithful to its citations. The five RAG layers have
no direct tests; they are covered only indirectly, through the endpoint tests that
are themselves unverified. A green suite means the plumbing works, not that the
system answers well.

The one excluded test is a throughput benchmark. It asserts that 50 writes finish
within 5 seconds, which measures the host rather than the code — it passes in
isolation (~3s) and failed at 9.6s when run after the rest of its module. It is
marked `benchmark` and deselected by default; run it with `pytest -m benchmark`.

---

## Evaluation (not performed)

**No evaluation has been run on this system, and this repository contains no
ground-truth dataset.** There are no accuracy, precision, recall or latency
figures, because there is nothing here that could produce an honest one.

An earlier version of this repository shipped evaluation scaffolding and quoted
figures derived from it. Both have been removed. The scaffolding graded answers by
bag-of-words overlap against a prose rubric such as *"Should identify specific SEC
violations with dates and regulations"*. Measured against its own bundled test
plan, that scorer:

- **passed** an answer that merely echoed the rubric back, containing no facts;
- **passed** the same words shuffled into nonsense;
- **failed** a specific, correct, checkable answer about a real enforcement action.

A metric anti-correlated with correctness is worse than no metric, because it
yields a percentage that looks like evidence. It was removed rather than reported.

Building a real one was rejected for a specific reason, not for effort: the corpus
is downloaded from EDGAR at ingestion time and no documents ship here, so any
question set would have been written at the same moment as its own answer key.
That is not an evaluation, it is a restatement of what the author already believed,
and presenting it as a measurement would repeat the original error in a new form.

A genuine evaluation would need, at minimum:

1. A frozen corpus — specific filings, pinned by accession number, so results are reproducible.
2. 15–20 questions written against those filings **by reading them**, spanning vector-shaped, graph-shaped and multi-hop questions, so routing is actually tested.
3. For each, the expected answer and the **passage that supports it**, so retrieval and generation can be scored separately — a right answer from the wrong passage is a lucky guess.
4. Metrics that mean something: retrieval recall@k against the supporting passages, and faithfulness of the answer to its citations, graded by a human or an LLM judge on a rubric that has itself been spot-checked.
5. A vector-only baseline run on the same set — otherwise "hybrid helps" is an assertion, not a finding.

Until that exists, the honest claim for this repository is that it is a working
implementation of a hybrid retrieval architecture, not a validated one.

---

## Known limitations

- **No evaluation.** As above.
- **Re-ranking is an interface, not an implementation.** `RetrievalLayer` accepts a
  reranker and reports `reranking_applied`, but the orchestrator constructs it with
  `reranker=None`. Results are fused, never re-ranked.
- **Entity extraction is unvalidated.** The graph is populated by asking an LLM for
  JSON. Nothing checks precision or recall of that extraction, and a missed entity
  is silently invisible to every graph query thereafter.
- **`src/advanced/` is prototype code.** A/B testing, bandits, shadow deployment,
  HITL and root-cause analysis are implemented and importable but not wired into
  the request path.
- **Graph schema is deliberately small.** Four node types and three relationships.
  It covers the compliance questions above and nothing wider.
- **Single-tenant.** No authentication, authorisation or rate limiting on the API.

---

## Repository layout

```
src/agentic_rag/     the five retrieval layers and their orchestrator
src/db/              PostgreSQL logging and Redis cache managers
src/monitoring/      Prometheus metrics, tracing, error tracking, dashboards
src/advanced/        prototype experimentation / feedback modules (not wired in)
src/main.py          FastAPI application
scripts/ingest.py    SEC EDGAR download, chunking, embedding and graph loading
scripts/runners/     operational entrypoints
tests/               integration tests
db/init.sql          PostgreSQL schema
```

## License

MIT — see [LICENSE](LICENSE).
