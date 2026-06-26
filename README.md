# Memory-Augmented Chatbot with Knowledge Graph & Hybrid RAG

An intelligent movie-domain chatbot that combines **Retrieval-Augmented Generation (RAG)**,
a **Knowledge Graph**, **long-term user memory**, and **LangGraph-orchestrated real-time
tools**, with an **evaluation framework** to measure response quality.

> **Status: Part 1 complete.** Foundation (LLM/embedding layer, config, FastAPI
> skeleton) + the data pipeline (Wikipedia + OMDB → cleaned, chunked corpus) are
> in place. RAG, the knowledge graph, memory, orchestration, tools and evaluation
> arrive in later parts (see roadmap).

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
| Generation / reasoning | **Gemini 2.0 Flash** (Google AI Studio) |
| Embeddings (corpus **and** query) | **text-embedding-004** — single model, no mixing |
| Fast routing | **Groq · Llama 3.3 70B** |
| Vector DB | **Chroma** — native metadata filtering on imdb_id/section (Part 2) |
| Knowledge graph | **Neo4j Aura** free tier (Part 3) |
| Long-term memory | **SQLite** (Part 5) |
| Orchestration | **LangGraph** (Part 6) |
| API | **FastAPI** |

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

# 6. Run the API skeleton
uvicorn app.api.main:app --reload
#   GET  http://127.0.0.1:8000/health
#   GET  http://127.0.0.1:8000/info
#   POST http://127.0.0.1:8000/chat   {"message": "Who directed Inception?"}
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
    gemini.py          #   Gemini 2.0 Flash client
    groq_llm.py        #   Groq Llama 3.3 70B client (routing)
    embeddings.py      #   text-embedding-004 (doc + query), batched
    embedding_cache.py #   SQLite cache so we never re-spend quota
    retry.py           #   shared rate-limit backoff
    factory.py         #   get_llm() / get_router_llm() / get_embedder()
  api/                 # FastAPI app (skeleton in Part 0, full in Part 8)
  data/                # Part 1 — scraping / cleaning / chunking
  rag/                 # Part 2 — vector store & retrieval
  graph/               # Part 3 — knowledge graph (Neo4j Aura)
  memory/              # Part 5 — long-term memory (SQLite)
  orchestration/       # Part 6 — LangGraph router + nodes
  tools/               # Part 7 — TMDB / live data tools
  evaluation/          # Part 9 — relevance / faithfulness / correctness
data/                  # generated artifacts (gitignored)
scripts/smoke_test.py  # live provider check
tests/                 # offline unit tests
```

---

## Roadmap (build order)

| Part | Module | What it adds |
| --- | --- | --- |
| **0** ✅ | `app/llm`, `app/api`, `config` | Foundation: LLM/embedding clients, config, API skeleton |
| 1 | `app/data` | Scrape → clean → chunk the movie corpus |
| 2 | `app/rag` | Embed + vector store + retriever (thin RAG end-to-end) |
| 3 | `app/graph` | Entity/relationship extraction → Neo4j Aura → NL→Cypher |
| 4 | `app/rag` | Hybrid retrieval — fuse vector + graph (GraphRAG) |
| 5 | `app/memory` | Per-user long-term memory (SQLite) |
| 6 | `app/orchestration` | LangGraph router + model/memory/RAG/tool nodes |
| 7 | `app/tools` | Real-time TMDB data tools |
| 8 | `app/api` | Full serving layer (sessions, history, streaming) |
| 9 | `app/evaluation` | Context relevance / faithfulness / answer correctness |
| 10 | — | Demo UI + logging/monitoring polish |

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
