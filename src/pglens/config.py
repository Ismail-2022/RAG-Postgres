"""Filesystem locations used across PGLens.

Paths are anchored to the project root, so commands work from any directory.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"
SOURCES_FILE = CONFIG_DIR / "sources.yaml"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # DVC-tracked: untouched HTML from postgresql.org
PROCESSED_DIR = DATA_DIR / "processed"  # DVC-tracked: markdown with frontmatter
INDEX_DIR = DATA_DIR / "qdrant"  # generated, not versioned: rebuilt from processed/
EVAL_DIR = DATA_DIR / "eval"  # hand-written golden set, versioned in git
GOLDEN_SET_FILE = EVAL_DIR / "golden_set.jsonl"
