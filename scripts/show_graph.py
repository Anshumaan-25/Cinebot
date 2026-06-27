"""Print the LangGraph orchestration structure (Part 6) — for submission docs.

    python scripts/show_graph.py

Renders the node/edge structure as ASCII (falls back to Mermaid). Uses stub
dependencies, so it needs no API keys or services — just the graph topology.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.orchestration.graph import build_chat_graph  # noqa: E402


def main() -> int:
    # Structure only — nodes are not executed when rendering, so stub deps are fine.
    graph = build_chat_graph(memory=None, router_llm=None, hybrid=None, gen_llm=None).get_graph()
    try:
        graph.print_ascii()
    except Exception:  # grandalf not available
        print(graph.draw_mermaid())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
