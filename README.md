# Memory-Augmented Chatbot with Knowledge Graph & Hybrid RAG

An intelligent movie-domain chatbot that combines **Retrieval-Augmented Generation (RAG)**,
a **Knowledge Graph**, **long-term user memory**, and **LangGraph-orchestrated real-time
tools**, with an **evaluation framework** to measure response quality.

> **Status: complete (Parts 0–10).** A **LangGraph orchestration** drives the
> chat: per-user **memory** recall → Groq **router** → either **hybrid GraphRAG**
> retrieve *or* **live tools** (OMDB + Wikipedia search, dynamically selected) →
> personalized **Gemini** answer (Groq fallback) → memory write. A **vanilla-CSS
> chat UI** is served at `/`; answer quality is measured by a **RAGAS-style LLM
> judge**. Visualize the graph with `python scripts/show_graph.py`.

---

## Domain & data sources

Scoped to **movies** (bounded but relationship-rich, ideal for a knowledge graph):

| Source | Used for |
| --- | --- |
| Wikipedia 'List of highest-grossing films' | the film list (scraped with BeautifulSoup) |
| Wikipedia articles | long-form text for RAG |
| OMDB API (omdbapi.com) | structured metadata: genre, cast, director, plot, imdbRating |
| Wikidata (property P345) | IMDb ids — the universal key linking everything |

> TMDB was the original plan but is **geo-blocked** in some regions (incl. India),
> so we use **OMDB** for structured data and key everything by **`imdb_id`**.
> Sentiment-rich text comes from Wikipedia "Reception" sections.

## Tech stack (all free tier — no paid APIs)

| Concern | Choice |
| --- | --- |
| Generation / reasoning | **Gemini 2.5 Flash** (Google AI Studio) |
| Embeddings (corpus **and** query) | **text-embedding-004** — single model, no mixing |
| Fast routing | **Groq · Llama 3.3 70B** |
| Vector DB | **Chroma** — native metadata filtering on imdb_id/section (Part 2) |
| Knowledge graph | **Neo4j Aura** free tier (Part 3) |
| Long-term memory | **SQLite** (Part 5) |
| Orchestration | **LangGraph** (Part 6) |
| API | **FastAPI** |

---

## How it works

The system has two halves: an **offline build** that turns raw movie data into
searchable artifacts, and an **online serve** path where every chat turn flows
through a LangGraph state machine.

### 1. Build time (run once, produces artifacts)

```
Wikipedia 'highest-grossing films'  ─┐
   → IMDb id via Wikidata (P345)     ├─►  films.jsonl + chunks.jsonl   (Part 1)
   → OMDB details + Wikipedia article ┘         │
                                                 ├─►  Chroma vectors   (Part 2, gemini-embedding-001)
                                                 └─►  Neo4j graph      (Part 3, Gemini cast extraction)
```

Everything is keyed by **`imdb_id`**, so the vector store and the knowledge
graph describe the same entities and can be fused later.

### 2. Serve time — one chat turn through the LangGraph

```
            ┌──────────────────────────────────────────────────────────┐
START ─► recall ─► router ─►│ retrieve  (hybrid GraphRAG)               │─► generate ─► persist ─► END
         (memory) (Groq)    │ tools     (live OMDB / Wikipedia search)  │   (Gemini→     (memory
                            │ generate  (chit-chat / preferences)       │    Groq)        write)
            └──────────────────────────────────────────────────────────┘
```

- **recall** — loads the user's stored preferences + recent history from SQLite
  and injects them into the prompt, so every answer is personalized.
- **router** — a fast Groq (Llama 3.3 70B) classifier labels the message
  `KNOWLEDGE` → *retrieve*, `REALTIME` → *tools*, or `CHITCHAT` → *generate*, and
  the graph branches on that label.
- **retrieve (hybrid GraphRAG)** — links the query to *known* graph entities
  (people / films / genres) with **no extra LLM call**, runs structured Cypher
  lookups (filmography, cast, genre intersections for "both X and Y" questions),
  and runs a vector search **focused on the linked films' `imdb_id`s**. Graph
  facts + passages are fused into one grounded context.
- **tools** — for live or out-of-corpus questions, Groq picks which tools to call
  and a clean search term; **OMDB** (live rating / cast / plot) and **Wikipedia**
  (article summary) run and their results join the context.
- **generate** — Gemini 2.5 Flash writes the answer from memory + graph facts +
  passages + tool results, citing passages as `[1]`, `[2]`. If Gemini's daily
  quota is spent it transparently **falls back to Groq** (production stays Gemini).
- **persist** — a Groq extractor pulls durable preferences from the turn and
  writes them (deduped) back to SQLite for next time.

### 3. Measuring quality (Part 9)

Two complementary evals: a deterministic **retrieval** eval (Hit@1 / Hit@5 / MRR
+ graph entity recall, no LLM) and a **RAGAS-style** eval where an LLM judge
(Groq) scores generated answers for faithfulness, answer & context relevance, and
answer correctness against gold references.

### Design principles

- **Grounded where it counts.** Factual claims (plots, cast, box office, who
  directed what) are answered from the corpus + graph with citations;
  recommendations/opinions blend that context with the model's general film
  knowledge. The UI only lists a passage as a *Source* when the answer cited it.
- **Free-tier resilience.** No paid fallback exists, so embeddings are cached to
  SQLite, every API call retries with backoff, and the generator degrades from
  Gemini → Groq on quota rather than failing.
- **Dependency-injected graph.** Nodes close over injected services, so the whole
  orchestration is testable with fakes and renderable (`scripts/show_graph.py`)
  without any live service.

---

## Quickstart

```bash
# 1. Create an environment (Python 3.11+; developed on 3.13)
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure keys
cp .env.example .env        # GOOGLE_API_KEY, GROQ_API_KEY, and OMDB_API_KEY

# 4. Verify the providers work (skips any without a key)
python scripts/smoke_test.py

# 5. Build the movie corpus (Part 1) -> data/processed/{films,chunks}.jsonl
python scripts/build_corpus.py --n 50   # needs OMDB_API_KEY

# 6. Build the vector index (Part 2) -> data/chroma/
python scripts/build_index.py           # needs GOOGLE_API_KEY (text-embedding-004)

# 7. (optional) Measure retrieval quality on the gold set
python scripts/run_eval.py              # Hit@1 / Hit@5 / MRR

# 7b. (optional) RAGAS-style answer quality, judged by an LLM (Groq)
python scripts/run_ragas_eval.py --generator groq   # faithfulness / relevance / correctness

# 8. Build the knowledge graph (Part 3) -> Neo4j Aura
python scripts/build_graph.py           # needs NEO4J_* (+ GOOGLE_API_KEY for the cast supplement)

# 9. (optional) Visualize the LangGraph orchestration (Part 6)
python scripts/show_graph.py

# 10. Run the app — chat UI at / ; POST /chat runs the full orchestration
uvicorn app.api.main:app --reload
#   UI   http://127.0.0.1:8000/        ← open this in a browser
#   GET  http://127.0.0.1:8000/health
#   GET  http://127.0.0.1:8000/info
#   POST http://127.0.0.1:8000/chat   {"message": "How did critics react to Joker?"}
#   docs http://127.0.0.1:8000/docs
```

Run the offline tests (no keys/network needed):

```bash
pytest -q
```

---

## Project structure

```
app/
  config.py            # env/.env settings (pydantic-settings)
  logging_config.py
  llm/                 # provider-agnostic LLM + embedding layer  ← Part 0
    base.py            #   LLMClient interface + ChatMessage
    gemini.py          #   Gemini 2.5 Flash client
    groq_llm.py        #   Groq Llama 3.3 70B client (routing)
    embeddings.py      #   text-embedding-004 (doc + query), batched
    embedding_cache.py #   SQLite cache so we never re-spend quota
    retry.py           #   shared rate-limit backoff
    factory.py         #   get_llm() / get_router_llm() / get_embedder()
  api/                 # FastAPI app (skeleton in Part 0, full in Part 8)
  data/                # Part 1 — scraping / cleaning / chunking
  rag/                 # Part 2 — vector store & retrieval
  kg/                  # Part 3 — Neo4j knowledge graph (graph_store, extraction)
  memory/              # Part 5 — long-term memory (SQLite)
  orchestration/       # Part 6 — LangGraph router + nodes
  tools/               # Part 7 — live OMDB + Wikipedia search tools + runner
  evaluation/          # Part 9 — relevance / faithfulness / correctness
  api/main.py          # also serves the chat UI at / + static assets  ← Part 10
data/                  # generated artifacts (gitignored)
web/                   # Part 10 — vanilla HTML/CSS/JS chat UI (index.html, style.css, app.js)
scripts/smoke_test.py  # live provider check
tests/                 # offline unit tests
```

---

## Roadmap (build order)

| Part | Module | What it adds |
| --- | --- | --- |
| **0** ✅ | `app/llm`, `app/api`, `config` | Foundation: LLM/embedding clients, config, API skeleton |
| **1** ✅ | `app/data` | Scrape → clean → chunk the movie corpus |
| **2** ✅ | `app/rag` | Embed + Chroma vector store + retriever + streaming cited RAG + lite eval |
| **3** ✅ | `app/kg` | Entity/relationship extraction (Gemini) → Neo4j Aura graph + querying |
| **4** ✅ | `app/rag` | Hybrid retrieval — entity-link → graph + vector fusion (GraphRAG) |
| **5** ✅ | `app/memory` | Per-user long-term memory — preferences + history (SQLite) |
| **6** ✅ | `app/orchestration` | LangGraph: memory/router/retrieve/tools/generate nodes + conditional routing |
| **7** ✅ | `app/tools` | Live tools — OMDB + Wikipedia search, LLM-selected (TMDB geo-blocked → these instead) |
| **8** ✅ | `app/api` | Serving layer — streaming `/chat` over the orchestration |
| **9** ✅ | `app/evaluation` | RAGAS-style LLM-judge — faithfulness / answer & context relevance / correctness |
| **10** ✅ | `web/` | Vanilla-CSS streaming chat UI served by FastAPI at `/` |

---

## Design notes

- **Provider-agnostic LLM layer.** Everything depends on the `LLMClient`
  interface, so providers can be swapped without touching callers.
- **Free-tier discipline.** Embeddings are cached to SQLite and all API calls
  retry with exponential backoff on rate-limit/transient errors. Because there's
  no paid fallback, caching the corpus embeddings is what keeps us within quota.
- **One embedding model end-to-end.** `text-embedding-004` for both corpus and
  query (only the `task_type` hint differs: `RETRIEVAL_DOCUMENT` vs
  `RETRIEVAL_QUERY`) — no mixed-model retrieval drift.
- **Offline vs online split.** Scraping, embedding and graph construction are
  build-time jobs that produce artifacts; the chatbot loads those and serves.
