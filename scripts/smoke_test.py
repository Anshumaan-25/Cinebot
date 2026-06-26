"""Live smoke test for the Part 0 LLM/embedding clients.

Verifies each configured provider with a tiny real request. Providers without a
key are skipped (not failed), so you can run this with partial configuration.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys

from app.config import get_settings
from app.logging_config import setup_logging


def main() -> int:
    setup_logging()
    settings = get_settings()
    settings.ensure_dirs()
    ok = True

    # --- Gemini generation ---
    if settings.has_gemini:
        try:
            from app.llm.factory import get_llm

            reply = get_llm().generate("Reply with exactly: PONG")
            print(f"[gemini ] OK  -> {reply.strip()[:80]!r}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[gemini ] FAIL -> {exc}")
    else:
        print("[gemini ] SKIP (GOOGLE_API_KEY not set)")

    # --- Groq routing ---
    if settings.has_groq:
        try:
            from app.llm.factory import get_router_llm

            reply = get_router_llm().generate("Reply with exactly: PONG")
            print(f"[groq   ] OK  -> {reply.strip()[:80]!r}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[groq   ] FAIL -> {exc}")
    else:
        print("[groq   ] SKIP (GROQ_API_KEY not set)")

    # --- Embeddings (text-embedding-004) ---
    if settings.has_gemini:
        try:
            from app.llm.factory import get_embedder

            embedder = get_embedder()
            vecs = embedder.embed_documents(["The Matrix is a 1999 sci-fi film."])
            q = embedder.embed_query("matrix movie")
            print(f"[embed  ] OK  -> doc dim={vecs[0].shape[0]}, query dim={q.shape[0]}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"[embed  ] FAIL -> {exc}")
    else:
        print("[embed  ] SKIP (GOOGLE_API_KEY not set)")

    print("\nResult:", "ALL GOOD" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
