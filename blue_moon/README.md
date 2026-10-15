# NovaCortex

Bilingual (Arabic/English) Telegram assistant for **Yarmouk University** — answers
admissions, faculties, fees, study-plan and campus questions using a hybrid of
verified facts, a RAG vector index over the scraped university knowledge base,
web search and an LLM fallback, with guardrails, semantic caching and per-user
conversation memory.

> **Local hackathon build:** the repository also contains a separate, generic
> Ollama-based document assistant. It has no Yarmouk, Telegram, Groq or web-search
> dependency. Start with `python hackathon.py preview`; the Arabic preparation
> guide is in `docs/hackathon-prep-ar.md`.

For a clean setup on another Windows or Linux laptop, follow
`docs/second-laptop-setup-ar.md`. The default local models are
`qwen2.5:7b` for answers and `embeddinggemma` for retrieval embeddings.

The hackathon assistant uses a deterministic LangGraph workflow around its
hybrid retrieval and grounded answer components. An optional MCP server exposes
the same engine to compatible clients through `python hackathon_mcp.py`.

The implementation used to live in a single ~4,200-line `NovaCortexbot.py`.
It is now a normal Python package under `src/novacortex/`; the original file
remains only as a thin compatibility shim.

### Government-service RAG safeguards

The local hackathon assistant includes a small set of high-impact strategies
for Arabic government-service documents:

- recognises service sections such as eligibility, required documents, fees,
  processing time, steps, exceptions and legal basis, and boosts the matching
  section for each question;
- can restrict a question to selected indexed files from the UI or with
  `ask "..." --source guide.pdf`, preventing unrelated services from being
  mixed into the context;
- falls back to exact, cited source passages when the generated answer or its
  quotation cannot be validated;
- warns when different sources contain conflicting numeric values with the
  same unit, or when a retrieved source explicitly says it is obsolete;
- reports likely scanned PDFs during preview so OCR can be enabled only when
  the competition documents actually require it.

Rebuild an existing hackathon index after pulling these changes so the new
service-section metadata is stored with every chunk.

### FastAPI web interface

The deployable interface is served by FastAPI while the original Streamlit UI
remains available for quick experiments. Install `hackathon_requirements.txt`,
build the index, then run:

```bash
python hackathon_api.py
```

Open `http://127.0.0.1:8000`. The same server provides `/api/health`,
`/api/sources`, `/api/ask`, and interactive API documentation at `/api/docs`.
The RAG engine stays warm between requests and a local semaphore prevents
multiple Ollama generations from exhausting laptop RAM or VRAM.

For a network deployment, set `HACKATHON_API_HOST=0.0.0.0` and a strong
`HACKATHON_API_TOKEN`. The launcher refuses a non-local bind without that
token. Put TLS in front of the app and keep Ollama, Chroma, the index and the
uploaded documents on the same private server.

## Features

- **Verified facts first** — deterministic answers for curated questions
  (founding date, deans, admissions, careers, …) with 100% confidence.
- **Hybrid RAG** — Chroma vector store + BM25 with reciprocal-rank fusion,
  cross-encoder-style reranking, priority boosting and per-document chunk caps.
- **Web-search fallback** — Serper (Google) search when the index is not enough.
- **Guardrails** — out-of-scope detection, prompt-injection blocking,
  false-premise correction and comparison-request refusal.
- **Semantic cache** — SQLite-backed cosine cache with a tunable similarity
  threshold (0.88). Time-sensitive, person-specific and numeric questions are
  excluded so a similar old question cannot reuse a stale answer.
- **Memory** — per-user conversation history with rolling summaries.
- **Bilingual NLP** — Arabic normalisation/tokenisation/stemming (NLTK
  Snowball) and language detection with an Arabic-dominance threshold.
- **Auto-reindex** — MD5 of `data/yarmouk_data.json` is cached; the vector
  store rebuilds itself when the knowledge base changes without dropping the
  middle sections of long source documents.

## Requirements

- Python **3.10+** (the pinned dependency stack requires it; the source itself
  is written to stay import-compatible with older interpreters).
- A working virtual environment with the dependencies from `requirements.txt`
  (or `pip install -e .`).

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
pip install -e .              # optional: installs the `novacortex` command
```

## Configuration

Secrets are read from the environment, optionally via a git-ignored `.env`
file in the project root:

```bash
cp .env.example .env          # then fill in your keys
```

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | Groq LLM (OpenAI-compatible endpoint) |
| `TELEGRAM_TOKEN` | yes | Telegram bot token (@BotFather) |
| `SERPER_API_KEY` | yes | Serper.dev web search (RAG fallback) |
| `GEMINI_API_KEY` | no | used only by legacy/auxiliary tooling |
| `NOVACORTEX_DATA_DIR` | no | knowledge-base directory (default `data/`) |
| `NOVACORTEX_DB_PATH` | no | SQLite database (default `novacortex_v12.db`) |
| `NOVACORTEX_LOG_FILE` | no | log file (default `logs/novacortex.log`) |

The bot refuses to start when a required secret is missing (`require_secrets`).

> **Security note:** all keys that ever lived in this repository were moved
> out of source control. Rotate any key that was previously committed, and
> keep `.env` out of git (`.gitignore` already excludes it).

## Running

Run from the project root (paths such as `data/`, `cache/`, `chroma_ultra/`
and the SQLite database are CWD-relative):

```bash
python -m novacortex          # preferred
python NovaCortexbot.py       # legacy shim (same entry point)
novacortex                    # if installed with `pip install -e .`
```

Bot commands: `/start` (registration conversation), `/clear` (wipe memory),
`/privacy` (show stored-data information), `/delete_my_data CONFIRM` (delete
the profile, question log and memory), `/cancel` (abort conversation), then
ask anything in the free-text handler.

## Project layout

```
NovaCortes/
├── NovaCortexbot.py        # thin compatibility shim (imports from the package)
├── pyproject.toml          # package metadata, entry point, pytest config
├── requirements.txt
├── .env / .env.example     # secrets (git-ignored) / template
├── src/novacortex/         # the actual implementation
│   ├── config.py           # paths, constants, secrets, require_secrets()
│   ├── llm.py              # lazy Groq/OpenAI-compatible client
│   ├── logging_setup.py    # configure_logging() -> logs/novacortex.log
│   ├── bot/handlers.py     # Telegram handlers + main()
│   ├── agent.py            # YarmoukAgent.ask() orchestrator
│   ├── pipeline.py         # EnhancedHybridPipeline (routing/answering)
│   ├── intents.py          # semantic parser, facts router, colloquial terms
│   ├── safety.py           # out-of-scope / injection / false-premise guards
│   ├── cache.py            # semantic answer cache (SQLite, threshold 0.88)
│   ├── memory.py           # per-user history + summaries
│   ├── db.py               # users/questions/stats schema
│   ├── text.py             # ArabicProcessor (normalise/tokenise/detect)
│   ├── formatting.py       # response formatting + no-result messages
│   ├── knowledge/          # static data: facts, faculty, careers, keywords
│   └── rag/                # indexing, chroma retrieval, web search,
│                           # reranking, answer generation, PDF extraction
├── data/                   # knowledge base JSON (yarmouk_data.json, …)
├── tests/                  # pytest suite (data integrity + unit + gated E2E)
├── scripts/                # scrapers, data cleaners, dashboard, generators
├── docs/architecture.md    # module map and request-flow documentation
├── legacy/                 # orphaned pre-refactor files (not imported)
└── logs/                   # novacortex.log + archive/ of historical logs
```

Runtime artefacts that stay **out of git** (see `.gitignore`): `cache/`,
`chroma_ultra/`, `*.pkl` (BM25 corpora), `*.db`, `*.log`, `dashboard_output/`,
`synthetic_questions.json`.

## Tests

```bash
python -m pytest                    # from the project root
python tests/test_data_integrity.py # standalone, no pytest needed
```

- `tests/test_data_integrity.py` — knowledge-base integrity (JSON validity,
  garbled-text detection, title fixes, priority coverage) plus syntax and
  regression-marker checks over the whole package.
- `tests/test_unit_text.py` — Arabic normalisation / language detection.
- `tests/test_unit_knowledge.py` — static knowledge + config sanity.
- `tests/golden_questions.json` + `scripts/evaluate_answers.py` — content-level
  evaluation of sources, required facts, forbidden claims and abstention.
- `tests/test_bot_internal.py` — end-to-end agent test, **skipped unless**
  `NOVACORTEX_RUN_INTEGRATION=1` (needs a working environment + network).

## Scripts

All scripts are run from the project root (`python scripts/<name>.py`) and
anchor their data paths to the repository root:

| Script | Purpose |
|---|---|
| `yu_scraper.py` | full YU site scraper → `data/yarmouk_data_new.json` |
| `update_facts_db.py` | incremental facts/scraper merge |
| `yarmouk_missing_scraper.py` | targeted PDF/plan backfill (`--input/--only`) |
| `clean_data.py` / `safe_data_cleaner.py` | noise removal pipeline |
| `Generate synthetic questions.py` | LLM-generated training questions |
| `dashboard.py` | Plotly/HTML analytics dashboard → `dashboard_output/` |

## Documentation

See [`docs/architecture.md`](docs/architecture.md) for the module map, the
request flow through the pipeline, and notes on the lazy-import design.
