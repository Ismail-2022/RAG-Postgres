# PGLens

A RAG assistant for PostgreSQL administration and performance. It answers
questions about indexing, query planning, performance tuning, VACUUM and
maintenance, monitoring, and concurrency, and cites the official
PostgreSQL documentation for every answer.

Built as a portfolio project that demonstrates MLOps and CI/CD practice,
at zero cost.

> The original planning handoff is kept in [PROJECT_BRIEF.md](PROJECT_BRIEF.md).
> This file is the working overview and is kept up to date as the project moves.

## Goals

- **Production-grade**: CI/CD, automated evaluation gates, containerised, hosted.
- **MLOps-demonstrating**: experiment tracking, model registry, data versioning,
  and a measurable fine-tuning result.
- **Free to build and run**: no paid tiers anywhere.
- **Runs locally on an Apple Silicon Mac** for development and training.

## Scope and decisions

- **Corpus**: 19 hand-curated PostgreSQL 17 documentation pages, listed
  in [sources.yaml](sources.yaml). Editing that file is how pages are added or
  removed.
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
| Embeddings | sentence-transformers, later fine-tuned | CPU in the container |
| Vector DB | Qdrant (Docker Compose) | Qdrant Cloud free cluster |
| API / UI | FastAPI + lightweight front end | same |
| Experiment tracking | MLflow (Docker Compose) | DagsHub free MLflow server |
| Data versioning | DVC | DagsHub storage remote |
| Evaluation | RAGAS / DeepEval, ~50-question golden set | GitHub Actions |
| CI/CD | GitHub Actions | GitHub Container Registry |
| Hosting | — | Hugging Face Spaces (Docker SDK) |

## Repository layout

Target layout. Items marked *(planned)* do not exist yet.

```
pglens/
├── sources.yaml                  # curated corpus list (human-edited)
├── fetch_postgres_docs.py        # ingestion; moves to src/pglens/ingest/ (planned)
├── src/pglens/                   # (planned)
│   ├── ingest/                   # fetch + markdown conversion
│   ├── chunking/                 # heading-aware chunking of processed/*.md
│   ├── retrieval/                # Qdrant indexing and query
│   ├── graph/                    # LangGraph pipeline
│   ├── api/                      # FastAPI app
│   └── eval/                     # RAGAS harness, golden set loader
├── training/                     # embedder fine-tuning (MPS) (planned)
├── data/
│   ├── raw/                      # DVC-tracked, generated
│   ├── processed/                # DVC-tracked, generated
│   └── eval/golden_set.jsonl     # (planned)
├── docker/                       # Compose for Qdrant + MLflow, Dockerfile (planned)
├── .github/workflows/            # ci, deploy, nightly-eval (planned)
└── tests/                        # (planned)
```

## Status

- [x] Project brief and corpus list
- [x] Ingestion script drafted
- [x] Repo created (local git, `main` branch)
- [x] Verify all `sources.yaml` slugs resolve on PostgreSQL 17 (19 of 19)
- [x] Run ingestion and spot-check markdown output (links absolute, permalinks and non-breaking spaces removed)
- [x] Offline unit tests with pytest (`tests/`, saved page fixtures)
- [x] Version `data/` with DVC (local remote at `~/dvc-store/pglens`)
- [ ] Repo scaffold: `pyproject.toml` (pip), ruff, pre-commit
- [ ] Type checking with mypy (add when wanted)
- [ ] Chunking, Qdrant indexing, LangGraph pipeline, `/ask` endpoint
- [ ] Evaluation harness and golden set
- [ ] Embedder fine-tuning with MLflow tracking
- [ ] CI/CD workflows and Hugging Face deployment
- [ ] Production polish: logging, `/health`, rate limiting, README results

## Running ingestion

```bash
python -m venv .venv
.venv/bin/pip install -e .                # runtime dependencies from pyproject.toml
.venv/bin/python fetch_postgres_docs.py   # run from the project root
```

Output: `data/raw/<slug>.html` (untouched HTML) and
`data/processed/<slug>.md` (markdown with frontmatter).

## Running tests

```bash
.venv/bin/pip install -e ".[dev]"   # adds pytest
.venv/bin/pytest
```

Tests run offline. Saved pages in `tests/fixtures/` are used as real-world input.

## Open questions

- Unified memory on the M5 MacBook: determines 7B vs 3B local model and the
  fine-tuning batch size. Not yet answered.
- Golden set size and the split between hand-written and synthetic questions.
  The plan is to start with 15–20 hand-written questions.
