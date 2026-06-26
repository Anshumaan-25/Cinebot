"""Embed the corpus chunks and load them into the Chroma vector store (Part 2).

    python scripts/build_index.py

Requires GOOGLE_API_KEY (text-embedding-004) and a built corpus
(data/processed/chunks.jsonl from scripts/build_corpus.py). Embeddings are
cached, so re-runs only re-embed new/changed chunks.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.data.schema import Chunk  # noqa: E402
from app.llm.factory import get_embedder  # noqa: E402
from app.logging_config import setup_logging  # noqa: E402
from app.rag.factory import get_vector_store  # noqa: E402


def main() -> int:
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()

    if not settings.has_gemini:
        print("ERROR: GOOGLE_API_KEY is not set (needed for text-embedding-004).", file=sys.stderr)
        return 1
    if not settings.chunks_path.exists():
        print(
            f"ERROR: {settings.chunks_path} not found. Run scripts/build_corpus.py first.",
            file=sys.stderr,
        )
        return 1

    lines = [l for l in settings.chunks_path.read_text("utf-8").splitlines() if l.strip()]
    chunks = [Chunk.model_validate_json(l) for l in lines]
    print(f"Loaded {len(chunks)} chunks. Embedding with {settings.embedding_model} ...")

    embedder = get_embedder()
    started = time.monotonic()
    vectors = embedder.embed_documents([c.text for c in chunks])
    embed_s = time.monotonic() - started

    store = get_vector_store()
    store.recreate()
    store.add(chunks, vectors)
    total_s = time.monotonic() - started

    print("\n" + "=" * 60)
    print("  INDEX BUILD SUMMARY")
    print("=" * 60)
    print(f"  Chunks embedded     : {len(chunks)}")
    print(f"  Embedding model/dim : {settings.embedding_model} / {settings.embedding_dim}")
    print(f"  Collection          : {settings.chroma_collection} (count={store.count()})")
    print(f"  Embed time          : {embed_s:.1f}s")
    print(f"  Total time          : {total_s:.1f}s")
    print(f"  Chroma path         : {settings.chroma_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
