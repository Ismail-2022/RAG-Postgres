# PGLens — Project Brief & Handoff

A production-grade RAG portfolio project: a DBA/DevOps copilot that answers
questions about PostgreSQL performance, indexing, and administration, with
citations back to the official docs. Built to demonstrate MLOps and CI/CD
practices on a resume, at zero cost.

This file is the handoff from a planning conversation into Claude Code.
Read it fully before writing code — it captures decisions already made so
they aren't re-litigated.

---

## 1. Goal

Build and ship an end-to-end RAG system that is:
- **Production-grade**: CI/CD, automated eval gates, containerised, hosted.
- **MLOps-demonstrating**: experiment tracking, model registry, data
  versioning, a measurable fine-tuning result.
- **Free to build and run.** No paid tiers anywhere.
- **Runnable on an M5 MacBook** for local dev/training.

Target resume line (the bar to hit):
> "Built and deployed a RAG system over PostgreSQL documentation with a
> fine-tuned embedding model (+X% retrieval accuracy), MLflow model
> registry, and a GitHub Actions pipeline that blocks merges on
> LLM-evaluation regressions; containerised and continuously deployed to
> Hugging Face Spaces."

## 2. Decisions already made (do not re-litigate)

- **Domain**: PostgreSQL official documentation (docs.postgresql.org),
  **version 17 pinned** (never `/docs/current/` — it drifts).
- **Scope**: ~20 curated pages across indexing, query planning, performance
  tuning, VACUUM/maintenance, monitoring, and concurrency. Full list is in
  `sources.yaml` (below). This was deliberately chosen small — the
  project's rigor is the point, not corpus size.
- **Rejected alternatives**: GOV.UK content (legal/attribution complexity,
  weak business case, unsupported Search API, redirect handling pain).
  General "org-wide" scraping was rejected in favour of a hand-curated
  `sources.yaml` list.
- **Business framing**: internal engineering knowledge-base copilot — the
  same shape as a real enterprise support-deflection or onboarding tool.
- **Ingestion approach**: scripted fetch of static HTML pages (not an API),
  cache raw HTML untouched, convert to clean markdown with YAML
  frontmatter (source_url, content_hash, docs_version, fetched_at) for
  citation integrity. Manual copy-paste was explicitly rejected — no
  provenance, not reproducible, breaks the citation feature.
- **Local inference**: Ollama run **natively on macOS**, not inside Docker
  — Docker Desktop on Mac cannot access the Metal GPU, so containerised
  inference would be far too slow for dev iteration.
- **Image portability**: M5 Mac builds arm64 images; Hugging Face Spaces
  runs amd64. Must use `docker buildx` multi-arch builds in CI, or the
  image that passes locally won't run when deployed.

## 3. Stack (all free tiers — verify current limits before signing up)

| Layer | Local (dev) | Hosted (prod) |
|---|---|---|
| LLM | Ollama (Qwen2.5 7B or Llama 3.2 3B — pick based on RAM) | Groq or Gemini free API tier |
| Provider switching | LiteLLM (one env var swaps local ↔ hosted) | same |
| Orchestration | LangGraph: retrieve → rerank → generate → cite-check | same |
| Embeddings | sentence-transformers, later fine-tuned | runs on CPU in the container |
| Vector DB | Qdrant, Docker Compose | Qdrant Cloud free cluster |
| API / UI | FastAPI + lightweight front end | same |
| Experiment tracking + registry | MLflow, Docker Compose | DagsHub free MLflow server |
| Data versioning | DVC | DagsHub storage remote |
| Evaluation | RAGAS / DeepEval, ~50-question golden set | runs in GitHub Actions |
| CI/CD | GitHub Actions (free, public repo) | — |
| Image registry | — | GitHub Container Registry (free for public images) |
| Hosting | — | Hugging Face Spaces, Docker SDK, free CPU tier |

## 4. Data pipeline (already drafted — see attached files)

- `sources.yaml`: hand-curated list of ~20 doc slugs + topic tags, pinned
  to docs version 17. This file is the one human-curated artifact; editing
  it is how you add/remove corpus pages.
- `fetch_postgres_docs.py`: fetches each page's static HTML, caches it
  untouched under `data/raw/`, extracts the main content div (strips
  nav/header/footer chrome), converts to markdown via `markdownify`, and
  writes `data/processed/<slug>.md` with a YAML frontmatter block
  (title, slug, topic, source_url, docs_version, license, fetched_at,
  content_hash).
- **Not yet verified**: slugs in `sources.yaml` have not been confirmed
  live against docs v17 (the planning session had no internet access to
  postgresql.org). First task in Claude Code: run the fetch script, fix
  any `FAILED` slugs (Postgres renames pages between major versions), and
  spot-check 2-3 output `.md` files for markdown-conversion artifacts
  (code blocks, tables, escaped pipes are the usual culprits).
- Next steps after a clean fetch: `dvc add data/raw data/processed` and
  commit — this is the "data is versioned" milestone.

## 5. Repo shape (target)

```
pglens/
├── sources.yaml
├── fetch_postgres_docs.py       # becomes src/pglens/ingest/fetch.py
├── src/pglens/
│   ├── ingest/                  # fetch + markdown conversion (above)
│   ├── chunking/                # heading-aware chunking of processed/*.md
│   ├── retrieval/                # Qdrant indexing + query
│   ├── graph/                    # LangGraph pipeline (retrieve/rerank/generate/cite)
│   ├── api/                      # FastAPI app
│   └── eval/                     # RAGAS harness, golden set loader
├── training/                     # embedder fine-tuning (MPS)
├── data/
│   ├── raw/                      # DVC-tracked
│   ├── processed/                # DVC-tracked
│   └── eval/golden_set.jsonl
├── docker/
│   ├── docker-compose.yml        # Qdrant + MLflow for local dev
│   └── Dockerfile
├── .github/workflows/
│   ├── ci.yml                    # lint, type-check, unit tests, PR eval gate
│   ├── deploy.yml                # build multi-arch image, push GHCR, deploy HF Spaces
│   └── nightly-eval.yml          # re-run full eval on schedule, open issue on regression
└── tests/
```

## 6. Delivery phases (roughly 7 weeks part-time)

0. **Foundations** — repo scaffold, `uv`, ruff, mypy, pre-commit, Docker
   Compose for Qdrant+MLflow, first CI workflow (lint/type/test green).
1. **Baseline RAG** — finish ingestion (this is where we are now), chunk,
   index into Qdrant with an off-the-shelf embedder, LangGraph pipeline
   with citations, FastAPI `/ask` endpoint answering via local Ollama.
2. **Evaluation harness** — hand-write ~15-20 golden questions, generate
   the rest synthetically and review, log RAGAS metrics (faithfulness,
   context precision, answer relevancy) + retrieval metrics to MLflow.
   This phase's output gates everything after it.
3. **The ML loop** — synthetic training pairs, fine-tune the embedder on
   MPS, log runs to MLflow, compare vs baseline, register the winner under
   a `champion` alias, version data with DVC.
4. **CI/CD** — PR: tests + image scan + fast eval gate (fails merge if
   faithfulness drops below threshold). Merge to main: multi-arch build,
   push GHCR, deploy to HF Spaces. Weekly cron: re-check doc pages for
   changes, re-run full eval, auto-open an issue on regression.
5. **Production polish** — structured logging, `/health` endpoint, rate
   limiting, MLflow traces per request, README with architecture diagram
   and results table, live demo link, short demo video.

## 7. Open questions for Claude Code to raise or resolve

- M5 MacBook's unified memory amount — decides 7B vs 3B local model choice
  and fine-tuning batch size (was asked, not yet answered).
- Final confirmation that all 20 slugs in `sources.yaml` resolve on docs
  v17; adjust the list if any have been renamed/removed.
- Exact golden-set question count and split between hand-written and
  synthetic (planning session suggested starting with ~15-20 hand-written).
