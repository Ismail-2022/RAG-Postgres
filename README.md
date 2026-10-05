# PGLens

A RAG assistant for PostgreSQL administration and performance. It answers
questions about indexing, query planning, performance tuning, VACUUM and
maintenance, monitoring, and concurrency, and cites the official
PostgreSQL documentation for every answer.

Built as a portfolio project that demonstrates MLOps and CI/CD practice,
at zero cost.

> The original planning handoff is in [docs/PROJECT_BRIEF.md](docs/PROJECT_BRIEF.md).

## Goals

- **Production-grade**: CI/CD, automated evaluation gates, containerised, hosted.
- **MLOps-demonstrating**: experiment tracking, model registry, data versioning,
  and a measurable fine-tuning result.
- **Free to build and run**: no paid tiers anywhere.
- **Runs locally on an Apple Silicon Mac** for development and training.

## Scope and decisions

- **Corpus**: 19 hand-curated PostgreSQL 17 documentation pages, listed in
  [config/sources.yaml](config/sources.yaml). Editing that file is how pages
  are added or removed.
- **Version pinning**: always `docs/17`, never `docs/current`, so re-runs are
  reproducible.
- **Ingestion**: scripted fetch of static HTML. Raw HTML is cached untouched,
  then converted to markdown with YAML frontmatter (source URL, content hash,
  docs version, fetch time) so every citation can be traced back to its source.
- **Local inference**: Ollama runs natively on macOS, not in Docker, because
  Docker Desktop cannot use the Metal GPU.
- **Image portability**: images are built multi-arch with `docker buildx`
  (arm64 for local, amd64 for Hugging Face Spaces).
- **Rejected**: GOV.UK content (licensing and attribution complexity, weak fit)
  and general org-wide scraping (replaced by the hand-curated list).

## Stack

| Layer | Local (dev) | Hosted (prod) |
|---|---|---|
| LLM | Ollama (Qwen2.5 7B or Llama 3.2 3B) | Groq or Gemini free tier |
| Provider switching | LiteLLM | same |
| Orchestration | LangGraph: retrieve → rerank → generate → cite-check | same |
| Embeddings | fastembed (`BAAI/bge-small-en-v1.5`), later fine-tuned | CPU in the container |
| Vector DB | Qdrant, embedded local mode | Qdrant Cloud free cluster |
| API / UI | FastAPI + lightweight front end | same |
| Experiment tracking | MLflow (Docker Compose) | DagsHub free MLflow server |
| Data versioning | DVC | DagsHub storage remote |
| Evaluation | RAGAS / DeepEval, ~50-question golden set | GitHub Actions |
| CI/CD | GitHub Actions | GitHub Container Registry |
| Hosting | — | Hugging Face Spaces (Docker SDK) |

## Repository layout

```
Rag_PostGres/
├── README.md
├── pyproject.toml                 # dependencies and tool settings
├── config/
│   └── sources.yaml               # curated corpus list (human-edited)
├── data/                          # generated corpus (DVC-tracked, see below)
│   ├── raw.dvc, processed.dvc     # pointers to the real files
│   └── qdrant/                    # vector index, rebuilt on demand (not versioned)
├── docs/
│   └── PROJECT_BRIEF.md           # original planning handoff
├── src/pglens/
│   ├── config.py                  # all filesystem paths, anchored to the project root
│   ├── ingest/fetch.py            # fetch, cache, and convert doc pages
│   ├── chunking/chunker.py        # heading-aware chunking of processed markdown
│   └── retrieval/indexer.py       # Qdrant indexing and search
└── tests/
    ├── fixtures/                  # saved real pages used as test input
    ├── test_ingest.py
    ├── test_chunker.py
    └── test_indexer.py
```

Planned, not yet present: `src/pglens/graph/` (LangGraph pipeline),
`src/pglens/api/` (FastAPI), `src/pglens/eval/` (RAGAS harness),
`training/` (embedder fine-tuning), `docker/`, `.github/workflows/`, and
`data/eval/golden_set.jsonl`.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"   # runtime dependencies and pytest
```

## Usage

Run these from the project root. Paths are anchored to the project, so the
location of your shell does not matter.

```bash
# 1. Fetch the pages from postgresql.org into data/raw and data/processed
.venv/bin/python -m pglens.ingest.fetch

# 2. Build the vector index from data/processed
.venv/bin/python -m pglens.retrieval.indexer

# 3. Score retrieval against the golden set (data/eval/golden_set.jsonl)
.venv/bin/python -m pglens.eval.retrieval_eval
```

Step 1 contacts postgresql.org, so run it only when the corpus needs
refreshing. Step 2 uses the local model cache after its first download.

## Testing

```bash
.venv/bin/pytest
```

The tests run offline. They use saved pages in `tests/fixtures/` as real-world
input, and a fake embedder for the indexer, so they never download the model.

## Data versioning

`data/raw` and `data/processed` are tracked with DVC. Git stores only the
pointer files (`data/raw.dvc`, `data/processed.dvc`). The files themselves
live in the DVC cache and the configured remote.

## Status

- [x] Project brief and corpus list
- [x] Repo created (local git, `main` branch)
- [x] Ingestion: all 19 slugs resolve on PostgreSQL 17
- [x] Markdown output checked: links absolute, permalinks and non-breaking spaces removed
- [x] Heading-aware chunking (197 chunks from the current corpus)
- [x] Qdrant indexing and search (embedded local mode)
- [x] Offline unit tests with pytest and saved page fixtures
- [x] Data versioned with DVC (local remote at `~/dvc-store/pglens`)
- [ ] Pre-commit hooks (ruff, tests)
- [ ] Type checking with mypy (add when wanted)
- [x] Retrieval evaluation baseline: 23 questions, recall@5 1.000, MRR 0.928
  (questions were written from the section headings, so this is an optimistic
  upper bound; harder paraphrased questions are next)
- [ ] LangGraph pipeline with citations, `/ask` endpoint
- [ ] Evaluation harness and golden set
- [ ] Embedder fine-tuning with MLflow tracking
- [ ] CI/CD workflows and Hugging Face deployment
- [ ] Production polish: logging, `/health`, rate limiting

## Open questions

- Unified memory on the M5 MacBook: determines the 7B vs 3B local model and
  the fine-tuning batch size. Not yet answered.
- Golden set size and the split between hand-written and synthetic questions.
  The plan is to start with 15–20 hand-written questions.
- Ingestion sets a placeholder `User-Agent` (`github.com/<you>/pglens`). Replace
  it with the real repository URL before public use.
